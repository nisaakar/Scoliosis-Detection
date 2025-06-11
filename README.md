# Skolyoz Röntgen Görüntüleri ile Sınıflandırma Projesi

Bu proje, skolyoz hastalığının varlığını X-ray (röntgen) görüntüleri üzerinden sınıflandırmayı amaçlayan bir görüntü işleme ve derin öğrenme projesidir.

## 🔍 Proje Amacı

Amaç, skolyozlu ve normal omurgaya sahip bireylerin X-ray görüntülerini kullanarak bir sınıflandırma modeli geliştirmektir. Bu sayede doktorlara erken teşhis ve destekleyici bir otomatik araç sunulması hedeflenmiştir.

## 🗂️ Veri Kümesi

- `normal/`: Normal omurgaya sahip bireylerin X-ray görüntüleri (71 adet)
- `scol/`: Skolyoz hastalığına sahip bireylerin X-ray görüntüleri (188 adet)

Veri kümesi, X-ray görüntülerinden oluşmaktadır ve sınıf dengesizliğini göz önünde bulundurarak veri artırma (data augmentation) teknikleri uygulanmıştır.

## ⚙️ Kullanılan Teknolojiler

- Python
- OpenCV
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib / Seaborn (görselleştirme için)

## 📈 Model Eğitimi

- Model: Convolutional Neural Network (CNN)
- Giriş boyutu: (128, 128, 1) — gri tonlamalı görüntüler
- Veri Artırma: Yatay/Dikey çevirme, döndürme, parlaklık ayarı, bulanıklık
- Eğitim/Doğrulama oranı: %80 / %20
- Performans Metrikleri: Doğruluk (accuracy), Kayıp (loss), Confusion Matrix, ROC Curve

## 🧠 Alternatif Modeller

- RGB + manuel ön işleme (histogram eşitleme, keskinleştirme, kırpma)
- Gradyan görselleştirme (Grad-CAM) ile modelin karar verdiği alanların gösterimi

## 🧪 Örnek Çıktılar

- Eğitim doğruluğu: ~%85-90
- Test doğruluğu: ~%80+
- Grad-CAM örnekleriyle skolyozlu bölgelerin vurgulanması

## 📂 Klasör Yapısı

