import string
import easyocr
import json
import cv2
import numpy as np

# Initialize the OCR reader
# Mac kullanıyorsan gpu=False yapabilirsin, NVIDIA kartın varsa True kalsın.
reader = easyocr.Reader(['en'], gpu=True)

# --- 1. MAPPING SÖZLÜKLERİ (GENİŞLETİLMİŞ) ---

# Harf okunup Sayı olması gereken yerler (Örn: Plakanın ortasında D -> 0)
dict_char_to_int = {
    'O': '0',
    'I': '1', 'L': '1', 'J': '3',
    'A': '4',
    'S': '5',
    'G': '6', 'b': '6',
    'T': '7', 'Y': '7',
    'B': '8', 'R': '8',
    'D': '0', 'Q': '0', 'U': '0', # D, Q ve U sıklıkla 0 ile karışır
    'Z': '2'
}

# Sayı okunup Harf olması gereken yerler (Örn: Başta 0 -> O)
dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S',
    '8': 'B',
    '2': 'Z',
    '7': 'T'
}

# Harf - Harf Karışıklıkları (Özel Durumlar)
# Eğer W sürekli yanlışlıkla M okunuyorsa (veya tam tersi) buraya ekle
dict_visual_fix = {
    'W': 'M', # W okunursa M yap
    'Q': 'O', # Q okunursa O yap
    # 'E': 'F', 
}

def write_csv(results, output_path):
    """
    Sonuçları CSV dosyasına yazar.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )

def license_complies_format(text):
    """
    Plaka formatını kontrol eder (UK Formatı: 2 Harf, 2 Rakam, 3 Harf -> LL NN LLL)
    Toplam 7 hane.
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in string.digits or text[2] in dict_char_to_int.keys()) and \
       (text[3] in string.digits or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False

def format_license(text):
    """
    Formatı zorlar: Karakterin konumuna göre harf mi sayı mı olacağını belirler ve değiştirir.
    """
    license_plate_ = ''
    
    # 0,1,4,5,6 indisleri HARF olmalı (Rakam varsa harfe çevir)
    # 2,3 indisleri RAKAM olmalı (Harf varsa rakama çevir)
    mapping = {
        0: dict_int_to_char, 
        1: dict_int_to_char, 
        4: dict_int_to_char, 
        5: dict_int_to_char, 
        6: dict_int_to_char,
        2: dict_char_to_int, 
        3: dict_char_to_int
    }
    
    for j in range(7):
        char = text[j]
        
        # 1. Önce Görsel Hata Düzeltmesi (W -> M gibi)
        if char in dict_visual_fix:
            char = dict_visual_fix[char]
            
        # 2. Sonra Konum Bazlı Düzeltme (Format Zorlama)
        if char in mapping[j].keys():
            license_plate_ += mapping[j][char]
        else:
            license_plate_ += char

    return license_plate_

def preprocess_image(img):
    """
    OCR başarısını artırmak için görüntü işleme adımları
    """
    # 1. Görüntüyü biraz büyüt (Küçük plakalar için çok etkili)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Gri tonlamaya çevir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Gürültü azaltma (Bulanıklaştırma) - Opsiyonel, bazen yararlı bazen zararlı
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 4. Adaptive Thresholding veya Otsu
    # Otsu genelde en iyisidir çünkü otomatik eşik değeri belirler
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. İnceltme/Kalınlaştırma (Erosion/Dilation)
    # Harfler birbirine yapışıksa 'erode' kullanın. Harfler kopuksa 'dilate' kullanın.
    # W ve M karışıklığı genelde harfler kalınlaşıp birbirine yapıştığı için olur -> Erode deneyelim.
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    return binary

def read_license_plate(license_plate_crop):
    """
    Plakayı okur ve formatlar.
    """
    
    # Görüntüyü iyileştir
    processed_image = preprocess_image(license_plate_crop)

    detections = reader.readtext(processed_image)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')
        
        # Eğer okunan metin çok kirliyse ve alakasız karakterler varsa temizle
        # (Sadece harf ve rakamları bırak)
        text = ''.join(e for e in text if e.isalnum())

        if license_complies_format(text):
            return format_license(text), score

    return None, None

def get_car(license_plate, vehicle_track_ids):
    """
    Plaka koordinatlarına göre araç ID'sini bulur.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1

def get_json_data(results):
    json_output = []
    for frame_nmr in results.keys():
        for car_id in results[frame_nmr].keys():
            data = results[frame_nmr][car_id]
            if 'license_plate' in data and 'text' in data['license_plate']:
                record = {
                    "frame_id": frame_nmr,
                    "car_id": car_id,
                    "license_plate": data['license_plate']['text'],
                    "confidence": data['license_plate']['text_score'],
                    "bbox": data['license_plate']['bbox']
                }
                json_output.append(record)
    return json_output