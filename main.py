import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Veriyi Yükleme
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
df = pd.read_csv(url)

# 2. Veri Ön İşleme (Data Preprocessing)
df['age'] = df['age'].fillna(df['age'].mean()) # Boş yaşları ortalama ile doldur
df['sex'] = df['sex'].map({'female': 1, 'male': 0}) # Cinsiyetleri sayıya çevir

# 3. Model Hazırlığı
X = df[['pclass', 'sex', 'age', 'fare']]
y = df['survived']

# Veriyi %80 Eğitim, %20 Test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Modeli Eğitme ve Test Etme
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

tahminler = model.predict(X_test)
basari_orani = accuracy_score(y_test, tahminler)
print(f"Modelin Doğruluk (Accuracy) Oranı: % {basari_orani * 100:.2f}")

# 5. Özellik Önemi (Feature Importance) Görselleştirmesi
ozellik_onemleri = pd.Series(model.feature_importances_, index=X.columns)
ozellik_onemleri.sort_values().plot(kind='barh', color='teal', figsize=(8, 5))
plt.title('Yapay Zeka Karar Verirken Hangi Özellikleri Önemsedi?')
plt.xlabel('Önem Derecesi')
plt.show()