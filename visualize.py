import ast
import cv2
import numpy as np
import pandas as pd
import os

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

# 1. DOSYA YOLLARINI KONTROL ET
# Interpolasyon yapılmış veriyi okuyoruz (boşluksuz veri için)
csv_path = 'csv/test_interpolated.csv'
video_path = 'videos/sample.mp4' # Videonun tam yolu veya doğru klasörü

if not os.path.exists(csv_path):
    print(f"HATA: CSV dosyası bulunamadı: {csv_path}")
    print("Lütfen önce interpolate kodunu çalıştırdığından emin ol.")
    exit()

results = pd.read_csv(csv_path)

# 2. EN İYİ PLAKALARI HAFIZAYA AL (Global Best Match)
# Her araç ID'si için, videodaki en yüksek skorlu plakayı bulup sabitliyoruz.
vehicle_best_plates = {}
for car_id in np.unique(results['car_id']):
    car_data = results[results['car_id'] == car_id]
    
    # Bu aracın tüm kayıtları içinde en yüksek skorlu olanı bul
    max_score = np.amax(car_data['license_number_score'])
    
    # En iyi skora sahip satırı çek
    best_row = car_data[car_data['license_number_score'] == max_score].iloc[0]
    
    # Sözlüğe kaydet: {ArabaID: '34ABC123'}
    vehicle_best_plates[car_id] = best_row['license_number']

# Video yükleme
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Çıktı klasörü yoksa oluştur
if not os.path.exists('outputs'):
    os.makedirs('outputs')
out = cv2.VideoWriter('outputs/out_stabilized.mp4', fourcc, fps, (width, height))

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Video işleniyor... Lütfen bekleyin.")

ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    
    if ret:
        df_ = results[results['frame_nmr'] == frame_nmr]
        
        for row_indx in range(len(df_)):
            row = df_.iloc[row_indx]
            car_id = row['car_id']
            
            # --- ÇİZİM İŞLEMLERİ ---
            
            # 1. Arabayı Çiz
            # Veri string formatındaysa listeye çevir (interpolate çıktısı bazen string yapar)
            car_bbox = row['car_bbox']
            if isinstance(car_bbox, str):
                # Köşeli parantez düzeltmesi ve temizleme
                car_bbox = car_bbox.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(car_bbox)
            else:
                car_x1, car_y1, car_x2, car_y2 = car_bbox
                
            draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 15, line_length_x=200, line_length_y=200)

            # 2. Plaka Kutusunu Çiz
            plate_bbox = row['license_plate_bbox']
            if isinstance(plate_bbox, str):
                plate_bbox = plate_bbox.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
                px1, py1, px2, py2 = ast.literal_eval(plate_bbox)
            else:
                px1, py1, px2, py2 = plate_bbox
                
            cv2.rectangle(frame, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 10)

            # 3. YAZIYI YAZ (STABİL KISIM)
            # O anki karede ne okunduğuna bakma, hafızadaki EN İYİ plakayı yaz
            if car_id in vehicle_best_plates:
                text = vehicle_best_plates[car_id]
                
                # Eğer plaka '0' veya boş değilse yaz
                if text != '0' and text != 0:
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 6)
                    
                    # Yazıyı arabanın üzerine ortalayarak koy
                    text_x = int((car_x2 + car_x1 - text_width) / 2)
                    text_y = int(car_y1 - 50)
                    
                    # Arka plan kutusu (daha okunaklı olması için)
                    cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10), (text_x + text_width + 10, text_y + 20), (255, 255, 255), -1)
                    
                    # Yazı
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6, cv2.LINE_AA)

        out.write(frame)
    else:
        break

out.release()
cap.release()
print("Tamamlandı! Video 'outputs/out_stabilized.mp4' dosyasına kaydedildi.")