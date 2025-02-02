import cv2
import time
import pyautogui
from ultralytics import YOLO

# YOLOv8 modelinizi yükleyin
model = YOLO("rock.pt")

# Webcam'den görüntü almak için VideoCapture nesnesi
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı. Lütfen kamera bağlantısını kontrol edin.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera'dan görüntü alınamadı.")
            break

        # Modelin tahmini (inference)
        # conf parametresi ile minimum güven skorunu ayarlayabilirsiniz (ör. conf=0.5)
        results = model.predict(frame, conf=0.5)

        # Sonuçların çizimi için kopya alalım
        draw_frame = frame.copy()

        # Tespitler üzerinde dolaş
        for result in results:
            for box in result.boxes:
                # box.xyxy -> x1, y1, x2, y2
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cls_id = int(box.cls[0].item())   # Sınıf ID
                conf_score = float(box.conf[0].item())  # Güven skoru
                class_name = model.names[cls_id]  # Sınıf ismi ("Paper", "Rock", "Scissors")

                # Tespit edilen sınıfa göre ekranda gösterilecek metin
                if class_name == "Paper":
                    label_text = f"El Acik ({conf_score:.2f})"
                elif class_name == "Rock":
                    label_text = f"Ses Azaltiliyor... ({conf_score:.2f})"
                elif class_name == "Scissors":
                    label_text = f"Ses Artiriliyor... ({conf_score:.2f})"


                # Ekrana çizim (kutu & etiket)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (139, 69, 19), 2)
                cv2.putText(draw_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (139, 26, 26), 2)

                # Tespit edilen sınıfa göre tuş/aksiyon
                if class_name == "Scissors":
                    print("Scissors tespit edildi -> Ses Azaltiliyor...")
                    pyautogui.press("volumeup")
                    time.sleep(0.1)

                elif class_name == "Rock":
                    print("Rock tespit edildi -> Ses Artiriliyor...")
                    pyautogui.press("volumedown")
                    time.sleep(0.1)

                # Paper için sadece "El Açık" metni gösteriliyor; ek aksiyon yok.

        # İşlenmiş görüntüyü ekranda göster
        cv2.imshow("YOLOv8 Detection", draw_frame)

        # 'q' tuşuna basılırsa döngüyü kır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Kullanıcı tarafından durduruldu.")
finally:
    cap.release()
    cv2.destroyAllWindows()