import cv2
import numpy as np
import pandas as pd
import easyocr
import string
import ast
import sys
from ultralytics import YOLO

# --- 1. AYARLAR ---
VIDEO_PATH = 'sample.mp4'         # Video dosyanın adı
LICENSE_MODEL = 'best.pt'         # Plaka modelin (Senin eğittiğin veya indirdiğin)
COCO_MODEL = 'yolov8n.pt'         # Araba modeli
OUTPUT_CSV = 'sonuc.csv'          # Kaydedilecek veriler
OUTPUT_VIDEO = 'sonuc_videosu.mp4'# Çıktı videosu

# Windows için 'lap' kütüphanesi yaması (Hata almamak için)
try:
    import lapx
    sys.modules['lap'] = lapx
except ImportError:
    pass

# --- 2. YARDIMCI FONKSİYONLAR (UTIL) ---
# Dışarıdan dosya çağırmamak için fonksiyonları buraya gömdük.

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'text_score'))
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['car']['bbox']),
                                                            '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['license_plate']['bbox']),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score']))

def license_complies_format(text):
    # Basit kural: Plaka en az 4, en fazla 8 karakter olmalı (Gürültüyü önler)
    if len(text) < 4 or len(text) > 9:
        return False
    return True

def format_license(text):
    # Harf/Rakam düzeltmesi (Örn: '0' yerine 'O' okunmuşsa düzeltir)
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in range(len(text)):
        if j in mapping and text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def read_license_plate(license_plate_crop, reader):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        # Temizle (Sadece Alfanümerik)
        import re
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        if license_complies_format(text):
            return text, score # format_license(text) fonksiyonunu opsiyonel bıraktım
    return None, None

def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        # Plaka arabanın içinde mi?
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle_track_ids[j]
    return -1, -1, -1, -1, -1

# --- 3. ANA İŞLEM (TESPİT VE OKUMA) ---
print("🚀 Modeller Yükleniyor...")
coco_model = YOLO(COCO_MODEL)
license_plate_detector = YOLO(LICENSE_MODEL)
reader = easyocr.Reader(['en'], gpu=True) # GPU Kullanımı

cap = cv2.VideoCapture(VIDEO_PATH)
vehicles = [2, 3, 5, 7] # Araba, Motor, Otobüs, Kamyon
results = {}
frame_nmr = -1
ret = True

print("🎬 Video Analizi Başladı! (Lütfen Bekleyin, Çıktı vermeyebilir)...")

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        if frame_nmr % 20 == 0: print(f"⏳ İşlenen Kare: {frame_nmr}")

        # A. Araçları Takip Et (YOLOv8 Native Tracking - SORT Yerine)
        detections = coco_model.track(frame, persist=True, classes=vehicles, verbose=False)
        track_ids = []
        
        if detections[0].boxes.id is not None:
            boxes = detections[0].boxes.xyxy.cpu().numpy()
            ids = detections[0].boxes.id.int().cpu().numpy()
            for box, car_id in zip(boxes, ids):
                track_ids.append([box[0], box[1], box[2], box[3], car_id])

        # B. Plakaları Bul
        license_plates = license_plate_detector(frame, verbose=False)[0]
        
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # C. Eşleştirme
            xcar1, ycar1, xcar2, ycar2, car_id = get_car([x1, y1, x2, y2, score, class_id], track_ids)

            if car_id != -1:
                # D. Plakayı İşle ve Oku
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                
                # Görüntü İyileştirme
                gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                license_plate_text, license_plate_text_score = read_license_plate(thresh, reader)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

# Sonuçları Kaydet
write_csv(results, OUTPUT_CSV)
cap.release()
print(f"✅ Analiz Bitti. Veriler '{OUTPUT_CSV}' dosyasına kaydedildi.")

# --- 4. GÖRSELLEŞTİRME VE VİDEO OLUŞTURMA ---
print("🎥 Final Videosu Oluşturuluyor...")

results_df = pd.read_csv(OUTPUT_CSV)
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# En iyi plaka tahminlerini belirle (En yüksek skora sahip olanı al)
license_plate_map = {}
for car_id in np.unique(results_df['car_id']):
    max_score = np.amax(results_df[results_df['car_id'] == car_id]['license_number_score'])
    best_row = results_df[(results_df['car_id'] == car_id) & (results_df['license_number_score'] == max_score)].iloc[0]
    license_plate_map[car_id] = best_row['license_number']

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    frame_nmr += 1
    if not ret: break

    df_ = results_df[results_df['frame_nmr'] == frame_nmr]
    
    for row_indx in range(len(df_)):
        # Araba Kutusu
        car_bbox = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        cv2.rectangle(frame, (int(car_bbox[0]), int(car_bbox[1])), (int(car_bbox[2]), int(car_bbox[3])), (0, 255, 0), 4)

        # Plaka Kutusu
        plate_bbox = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        cv2.rectangle(frame, (int(plate_bbox[0]), int(plate_bbox[1])), (int(plate_bbox[2]), int(plate_bbox[3])), (0, 0, 255), 4)

        # Yazı
        car_id = df_.iloc[row_indx]['car_id']
        if car_id in license_plate_map:
            text = license_plate_map[car_id]
            
            # Arka plan kutusu
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
            cv2.rectangle(frame, (int(plate_bbox[0]), int(plate_bbox[1]) - text_h - 20), (int(plate_bbox[0]) + text_w, int(plate_bbox[1])), (0, 0, 0), -1)
            # Metin
            cv2.putText(frame, text, (int(plate_bbox[0]), int(plate_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    out.write(frame)

out.release()
cap.release()

print(f"🎉 TEBRİKLER! Proje tamamlandı.")
print(f"📄 Veriler: {OUTPUT_CSV}")
print(f"📺 Video: {OUTPUT_VIDEO}")