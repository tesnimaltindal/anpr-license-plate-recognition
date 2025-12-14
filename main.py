from ultralytics import YOLO
import cv2
import util
import boto3 
import datetime 
from sort.sort import *
from util import get_car, read_license_plate, write_csv, get_json_data
import json
import torch
from decimal import Decimal

# -------------------------------------------------------------------
# ⭐ BÖLÜM A: AWS Fonksiyon Tanımlamaları ⭐
# (Importlardan hemen sonra tanımlanır)
# -------------------------------------------------------------------

def upload_to_s3(image_data, plate_text, frame_id, car_id, s3, S3_BUCKET_NAME, REGION_NAME):
    """Kesilen plaka görüntüsünü S3'e yükler."""
    
    # Görüntüyü JPEG formatında bir buffer'a sıkıştırma
    is_success, buffer = cv2.imencode(".jpg", image_data)
    if not is_success:
        print("S3 yüklemesi için görüntü sıkıştırma hatası.")
        return None

    # S3 dosya adı oluşturma: Plaka_ZamanDamgası_AracID_KareID.jpg
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = f"{plate_text}_{timestamp}_car{car_id}_frame{frame_id}.jpg"
    
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=buffer.tobytes(),
            ContentType='image/jpeg'
        )
        # S3 URL'sini döndürme
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{REGION_NAME}.amazonaws.com/{filename}"
        return s3_url
    except Exception as e:
        print(f"S3 yükleme hatası: {e}")
        return None

def write_to_dynamodb(car_id, plate_text, confidence, s3_url, table):
    """Plaka meta verilerini DynamoDB'ye kaydeder."""
    
    # Birincil Anahtar (RecordID) için benzersiz bir timestamp ID oluşturma
    record_id = str(datetime.datetime.now().timestamp())
    
    try:
        table.put_item(
            Item={
                # DynamoDB, Birincil Anahtarı (Partition Key) kullanarak düşük gecikme sağlar.
                'RecordID': record_id,
                'LicensePlate': plate_text,
                'CarID': str(int(car_id)), # DynamoDB'de CarID'yi string olarak saklamak daha iyi
                'Confidence': Decimal(str(confidence)),
                'Timestamp': datetime.datetime.now().isoformat(),
                'S3ImagePath': s3_url
            }
        )
    except Exception as e:
        print(f"DynamoDB yazma hatası: {e}")
        
# -------------------------------------------------------------------


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0' #  RTX 4060
elif torch.backends.mps.is_available():
    device = 'mps'    # Mac 

print(f"YOLO Modeli '{device}' cihazında çalıştırılıyor...")

results = {}
mot_tracker = Sort()

# Modeller
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')


# ----------------------------------------------------
# ⭐ BÖLÜM B: AWS İstemci Tanımlamaları ⭐
# (Modeller yüklendikten hemen sonra ve döngüden önce başlatılır)
# ----------------------------------------------------

# Kendi AWS bölgenizle, S3 kova adınızla ve DynamoDB tablo adınızla DEĞİŞTİRİN!
REGION_NAME = 'eu-central-1' 
S3_BUCKET_NAME = 'anpr-plakalar-simalgokcu' 
DYNAMODB_TABLE_NAME = 'ANPR_Metadata' 

# Boto3 istemcilerini başlatma
s3 = boto3.client('s3', region_name=REGION_NAME)
dynamodb = boto3.resource('dynamodb', region_name=REGION_NAME)
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

# ----------------------------------------------------


# Video kaynağı (Canlı akış için 'videos/sample.mp4' yerine 0 veya RTSP linki yazılabilir)
cap = cv2.VideoCapture('sample.mp4') 

vehicles = [2, 3, 5, 7] # YOLOv8'deki araç sınıf ID'leri

frame_nmr = -1
ret = True

# --- HIZ AYARI ---
FRAME_SKIP = 5 
# -----------------

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    
    if ret:
        # HIZLANDIRMA ADIMI: Her 5 karede bir işlem yap
        if frame_nmr % FRAME_SKIP != 0:
            continue

        results[frame_nmr] = {}
        
        # 1. Araç Tespiti
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # 2. Araç Takibi (SORT)
        track_ids = mot_tracker.update(np.asarray(detections_))

        # 3. Plaka Tespiti
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Plakayı araca ata
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Plakayı Kırp
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # 4. OCR İşlemi
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                if license_plate_text is not None:
                    
                    # Sonuçları yerel sonuçlar sözlüğüne kaydet
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
                    
                    # ----------------------------------------------------
                    # ⭐ BÖLÜM C: AWS Fonksiyon Çağrıları ⭐
                    # ----------------------------------------------------
                    
                    # 1. S3'e Yükleme
                    s3_image_url = upload_to_s3(
                        license_plate_crop, 
                        license_plate_text, 
                        frame_nmr, 
                        car_id,
                        s3,
                        S3_BUCKET_NAME,
                        REGION_NAME
                    )

                    # 2. DynamoDB'ye Kaydetme
                    if s3_image_url:
                        write_to_dynamodb(
                            car_id=car_id, 
                            plate_text=license_plate_text, 
                            confidence=license_plate_text_score, 
                            s3_url=s3_image_url,
                            table=table
                        )
                    
                    # ----------------------------------------------------
                    
                    # Canlı İzleme İçin Konsola Bas
                    print(f"Frame: {frame_nmr} | Car: {car_id} | Plate: {license_plate_text} | Conf: {license_plate_text_score}")

# Döngü bittiğinde JSON çıktısını hazırla
data_to_send = get_json_data(results)
print("\n--- İŞLEM BİTTİ. YEREL JSON ÇIKTISI ---")
print(json.dumps(data_to_send, indent=4))