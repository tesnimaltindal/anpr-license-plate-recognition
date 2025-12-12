from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv, get_json_data
import json
import torch

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0' #  RTX 4060
elif torch.backends.mps.is_available():
    device = 'mps'    # Mac 

print(f"YOLO Modeli '{device}' cihazında çalıştırılıyor...")

# Modeli yüklerken cihazı belirtmek zorunda değilsin, YOLO bunu anlar.
# Ancak .to(device) ile zorlayabilirsin. Şimdilik standart bırakmak en iyisi:
results = {}
mot_tracker = Sort()

# Modeller
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# Video kaynağı (Canlı akış için 'videos/sample.mp4' yerine 0 veya RTSP linki yazılabilir)
cap = cv2.VideoCapture('videos/sample.mp4') 

vehicles = [2, 3, 5, 7]

frame_nmr = -1
ret = True

# --- HIZ AYARI ---
# Her kaç karede bir işlem yapılacağını belirler.
# 1 = Her kareyi işle (En yavaş, en hassas)
# 5 = 5 karede bir işle (Çok hızlı, canlı akış için ideal)
FRAME_SKIP = 5 
# -----------------

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    
    if ret:
        # HIZLANDIRMA ADIMI: Eğer sıradaki kare, hedef kare değilse atla
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
        # Not: SORT normalde her karede çalışmalı daha iyi sonuç verir ama
        # performans için burada da atlatıyoruz.
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

                # Görüntü İşleme kısmını SİLİYORUZ. 
                # Çünkü artık bu işi util.py içindeki 'preprocess_image' fonksiyonu daha iyi yapıyor.
                # OCR fonksiyonuna doğrudan renkli (crop) görüntüyü veriyoruz.

                # 4. OCR İşlemi (EN YAVAŞ KISIM BURASI)
                # Buraya bir kontrol ekleyebilirsin: Eğer bu car_id daha önce yüksek skorla okunduysa tekrar okuma!
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
                    
                    # Canlı İzleme İçin Konsola Bas (Opsiyonel)
                    print(f"Frame: {frame_nmr} | Car: {car_id} | Plate: {license_plate_text} | Conf: {license_plate_text_score}")

# Döngü bittiğinde JSON çıktısını hazırla
data_to_send = get_json_data(results)
print("\n--- AZURE İÇİN JSON ÇIKTISI ---")
print(json.dumps(data_to_send, indent=4))