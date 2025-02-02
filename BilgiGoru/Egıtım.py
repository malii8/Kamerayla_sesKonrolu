from ultralytics import YOLO
import torch

def train_yolov8():
    # 1. Modeli Yükleme
    model = YOLO('yolov8n.pt')  # Önceden eğitilmiş modelin yüklenmesi (nano boyut)

    # 2. Veri Seti ve Eğitim Parametrelerini Ayarlama
    data_path = r'C:/Users/Mehmet/PycharmProjects/BilgiGoru/rock-paper-scissors.v1i.yolov8/data.yaml'
    epochs = 100  # Eğitim yapılacak epoch sayısı
    batch_size = 16  # Batch boyutu
    img_size = 640  # Giriş görüntü boyutu


    # 3. Cihazı Belirleme (GPU varsa GPU kullan, yoksa CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Eğitim için kullanılacak cihaz: {device}")

    # 4. Modeli Eğitme
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        name='yolov8_training_s',  # Eğitim sonuçlarının kaydedileceği klasör adı
        device=device  # GPU kullanımı için 'cuda', CPU için 'cpu'

    )

    # 5. Modeli Değerlendirme
    results = model.val()

    # 6. Modeli Kaydetme
    model.save('EnİYİSONUC.pt')

if __name__ == '__main__':
    train_yolov8()
