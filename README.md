# Titanic Survival Prediction (Makine Öğrenmesi)

Bu proje, Veri Bilimi ve Makine Öğrenmesi alanındaki temel kavramları (Keşifsel Veri Analizi, Veri Ön İşleme ve Sınıflandırma) uygulamalı olarak göstermek amacıyla geliştirilmiştir. 

## Proje Amacı
Tarihi Titanic felaketindeki yolcu verilerini kullanarak, bir yolcunun hayatta kalma ihtimalini tahmin eden bir Yapay Zeka modeli geliştirmek.

## Kullanılan Teknolojiler
* **Dil:** Python
* **Veri Manipülasyonu:** Pandas
* **Makine Öğrenmesi:** Scikit-Learn (Random Forest Classifier)
* **Görselleştirme:** Matplotlib

## Model Başarısı ve Sonuçlar
* Veri seti %80 eğitim ve %20 test olarak ayrılmış, modelin hiç görmediği veriler üzerindeki testinde **%78.77 Doğruluk (Accuracy)** oranına ulaşılmıştır.
* **Feature Importance (Özellik Önemi)** analizine göre, modelin tahmin yaparken en çok dikkate aldığı kriterler sırasıyla:
  1. Ödenen Bilet Ücreti (Fare)
  2. Cinsiyet (Sex)
  3. Yaş (Age)
