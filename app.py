from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import io
import numpy as np
import json
import torch
import datetime 
from decimal import Decimal

# main.py'dan taşınan/import edilenler
from ultralytics import YOLO
import cv2
import boto3 
from sort.sort import *
# util.py'dan sadece kullanacağımız fonksiyonları import ediyoruz
from util import get_car, read_license_plate


# ----------------------------------------------------
# 2. AWS Fonksiyon Tanımlamaları (main.py'dan kopyalandı)
# ----------------------------------------------------

def upload_to_s3(image_data, plate_text, car_id, s3, S3_BUCKET_NAME, REGION_NAME):
    """Kesilen plaka görüntüsünü S3'e yükler."""
    
    # Görüntüyü JPEG formatında bir buffer'a sıkıştırma
    # Flask'ta 'image_data' zaten np.array formatında
    is_success, buffer = cv2.imencode(".jpg", image_data)
    if not is_success:
        print("S3 yüklemesi için görüntü sıkıştırma hatası.")
        return None

    # S3 dosya adı oluşturma: Plaka_ZamanDamgası_AracID.jpg (KareID'ye burada gerek yok)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{plate_text}_{timestamp}_car{car_id}.jpg"
    
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=buffer.tobytes(),
            ContentType='image/jpeg'
        )
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{filename}"
        return s3_url
    except Exception as e:
        print(f"S3 yükleme hatası: {e}")
        return None

def write_to_dynamodb(car_id, plate_text, confidence, s3_url, table):
    """Plaka meta verilerini DynamoDB'ye kaydeder."""
    
    record_id = str(datetime.datetime.now().timestamp())
    
    try:
        table.put_item(
            Item={
                'RecordID': record_id,
                'LicensePlate': plate_text,
                'CarID': str(int(car_id)),
                'Confidence': Decimal(str(confidence)),
                'Timestamp': datetime.datetime.now().isoformat(),
                'S3ImagePath': s3_url
            }
        )
    except Exception as e:
        print(f"DynamoDB yazma hatası: {e}")
        
# ----------------------------------------------------
# 3. GLOBAL AYARLAR, MODELLER VE AWS BAĞLANTILARI (main.py'dan kopyalandı)
# ----------------------------------------------------

# Cihaz ayarları (main.py'dan kopyalandı)
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'    

print(f"YOLO Modeli '{device}' cihazında çalıştırılıyor...")

# Modellerin ve Tracker'ın Yüklenmesi (main.py'dan kopyalandı)
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
mot_tracker = Sort() # API'de CarID takibini basit tutmak için her istekte yeniden başlatılır.

vehicles = [2, 3, 5, 7] # Araç sınıf ID'leri

# AWS İstemci Tanımlamaları (main.py'dan kopyalandı)
REGION_NAME = 'eu-central-1' 
S3_BUCKET_NAME = 'anpr-plakalar-simalgokcu' 
DYNAMODB_TABLE_NAME = 'ANPR_Metadata' 

s3 = boto3.client('s3', region_name=REGION_NAME)
dynamodb = boto3.resource('dynamodb', region_name=REGION_NAME)
table = dynamodb.Table(DYNAMODB_TABLE_NAME)


# ----------------------------------------------------
# 4. API ANA İŞLEM FONKSİYONU (main.py Mantığının Adaptasyonu)
# ----------------------------------------------------

# Bu fonksiyon, Flask'tan gelen görüntüyü alıp tüm ANPR/AWS mantığını çalıştırır.
def process_single_image_and_save(image_data):
    """Gelen görüntüyü işler, S3/DynamoDB'ye kaydeder ve sonuçları döndürür."""

    # Gelen bayt verisini (image_data) NumPy/OpenCV matrisine dönüştür
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {"plate": "HATA", "confidence": "0", "car_id": "0", "s3_url": "Görüntü okunamadı"}

    # 1. Araç Tespiti (main.py döngü mantığından alındı)
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # 2. Araç Takibi (mot_tracker her istekte sıfırlansa da, ilk tespiti sağlar)
    track_ids = mot_tracker.update(np.asarray(detections_))

    # 3. Plaka Tespiti (main.py döngü mantığından alındı)
    license_plates = license_plate_detector(frame)[0]
    
    # API'de genellikle sadece en yüksek skorlu plakayı işleriz.
    if license_plates.boxes.data.tolist():
        # Sadece ilk (ve muhtemelen en yüksek skorlu) plakayı al
        license_plate = license_plates.boxes.data.tolist()[0]
        x1, y1, x2, y2, score, class_id = license_plate

        # Plakayı araca ata
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        # Eğer araç tespit edilebildiyse devam et
        if car_id != -1:
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            
            # 4. OCR İşlemi
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

            if license_plate_text is not None:
                print(f"API İşlemi: Plaka {license_plate_text} bulundu, AWS'e gönderiliyor...")

                # 5. AWS Fonksiyon Çağrıları (main.py'dan kopyalandı)
                s3_image_url = upload_to_s3(
                    license_plate_crop, 
                    license_plate_text, 
                    car_id,
                    s3,
                    S3_BUCKET_NAME,
                    REGION_NAME
                )

                if s3_image_url:
                    write_to_dynamodb(
                        car_id=car_id, 
                        plate_text=license_plate_text, 
                        confidence=license_plate_text_score, 
                        s3_url=s3_image_url,
                        table=table
                    )
                
                # Sonuçları Flask'a döndür
                return {
                    "plate": license_plate_text,
                    "confidence": f"{license_plate_text_score:.4f}",
                    "car_id": str(int(car_id)),
                    "s3_url": s3_image_url if s3_image_url else "S3 yükleme başarısız."
                }
    
    # Plaka bulunamazsa
    return {"plate": "PLAKA TESPİT EDİLEMEDİ", "confidence": "0", "car_id": "0", "s3_url": "Yok"}


# ----------------------------------------------------
# 5. FLASK UYGULAMASI VE ROTASI
# ----------------------------------------------------

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    results = None
    if request.method == 'POST':
        if 'file' not in request.files:
            # Dosya seçilmedi
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            # Dosya adı boş veya uzantı uygun değil
            return redirect(request.url)

        if file:
            # Görüntüyü bellekten oku (image_data artık bayt verisi)
            image_data = file.read()
            
            # ANA İŞLEMİ ÇAĞIRMA
            results = process_single_image_and_save(image_data)
            
            # Sonuçları Terminal'e bas (Hata ayıklama için)
            print(f"Sonuç: {results}")

    # Sonuçları index.html'e göndererek göster
    return render_template('index.html', results=results)


# ----------------------------------------------------
# 6. UYGULAMAYI ÇALIŞTIRMA
# ----------------------------------------------------

if __name__ == '__main__':
    # Flask uygulamasını varsayılan portta (5000) çalıştır
    app.run(debug=True)