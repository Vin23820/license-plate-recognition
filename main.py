import os
import glob
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_and_ocr_license_plate(image_path):
    image = cv2.imread(image_path)
    original_image = image.copy()  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plate_rects = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            
            aspect_ratio = w / float(h)
            if aspect_ratio > 2.5 and aspect_ratio < 6.0 and w > 100 and h > 20:
                plate_rects.append((x, y, w, h))

    if plate_rects:
        plate_rects = sorted(plate_rects, key=lambda x: x[0])
        
        x, y, w, h = plate_rects[-1]
        
        plate_img = gray[y:y+h, x:x+w]
        
        custom_config = r'--oem 3 --psm 6'
        plate_number = pytesseract.image_to_string(plate_img, config=custom_config)
        
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original_image, plate_number.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return plate_number.strip(), original_image
    
    else:
        return None, None

print("Welcome to the Number Plate Detection System.\n")
    
dir_path = os.path.dirname(__file__)
images_dir = os.path.join(dir_path, "Images")
    
for img_path in glob.glob(os.path.join(images_dir, "*.jpeg")):
    detected_plate, annotated_image = detect_and_ocr_license_plate(img_path)
        
    if detected_plate:
        print(f"Detected number plate in {img_path}: {detected_plate}")
        cv2.imshow("Detected Number Plate", annotated_image)
        cv2.waitKey(1000)  
        cv2.destroyAllWindows()
    else:
        print(f"No number plate detected in {img_path}")
