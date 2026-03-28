import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

path = kagglehub.dataset_download("ritesaluja/bank-note-authentication-uci-data")
df = pd.read_csv(f"{path}/BankNote_Authentication.csv")

print(f"Avg dla kazdego atrybutu: {df.mean()}")
print(f"Std dla kazdego atrybutu: {df.std()}")
print(f"Wariancja dla kazdego atrybutu: {df.var()}")
print(f"Korelacja między atrybutami: {df.cov()}")
for col in df.columns:
    col_min = df[col].min()
    col_max = df[col].max()
    rozstep = col_max - col_min
    print(f"Atrybut '{col}': min = {col_min:.4f}, max = {col_max:.4f}, rozstęp = {rozstep:.4f}")

features = df.drop(columns=['class'])
min_max_scaler = MinMaxScaler()
features_minmax = pd.DataFrame(min_max_scaler.fit_transform(features), columns=features.columns)

print(f"Min-Max Scaled Data:\n{features_minmax.head()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=features['variance'], y=features['entropy'], hue=df['class'], palette='Set1', ax=axes[0])
axes[0].set_title('Przed przeskalowaniem (Oryginał)')
sns.scatterplot(x=features_minmax['variance'], y=features_minmax['entropy'], hue=df['class'], palette='Set1', ax=axes[1])
axes[1].set_title('Po przeskalowaniu (Min-Max [0, 1])')
plt.tight_layout()
plt.show()

standard_scaler = StandardScaler()
features_standard = pd.DataFrame(standard_scaler.fit_transform(features), columns=features.columns)

print(f"Średnia po standaryzacji: {features_standard.mean()}")
print(f"Odchylenie standardowe po standaryzacji: {features_standard.std()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(x=features['variance'], y=features['entropy'], hue=df['class'], palette='Set2', ax=axes[0])
axes[0].set_title('Przed przeskalowaniem (Oryginał)')
sns.scatterplot(x=features_standard['variance'], y=features_standard['entropy'], hue=df['class'], palette='Set2', ax=axes[1])
axes[1].set_title('Po standaryzacji (StandardScaler)')
plt.tight_layout()
plt.show()


norm_l1 = Normalizer(norm='l1')
norm_l2 = Normalizer(norm='l2')
features_norm_l1 = pd.DataFrame(norm_l1.fit_transform(features), columns=features.columns)
features_norm_l2 = pd.DataFrame(norm_l2.fit_transform(features), columns=features.columns)

print(f"Norma L1 (Suma wartości bezwzględnych dla każdego wiersza = 1):\n{features_norm_l1.head()}")
print(f"Norma L2 (Długość wektora euklidesowego dla każdego wiersza = 1):\n{features_norm_l2.head()}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(x=features['variance'], y=features['skewness'], hue=df['class'], palette='viridis', ax=axes[0])
axes[0].set_title('Dane oryginalne')
sns.scatterplot(x=features_norm_l1['variance'], y=features_norm_l1['skewness'], hue=df['class'], palette='viridis', ax=axes[1])
axes[1].set_title('Normalizacja L1 (Suma wartości bezwzględnych = 1)')
sns.scatterplot(x=features_norm_l2['variance'], y=features_norm_l2['skewness'], hue=df['class'], palette='viridis', ax=axes[2])
axes[2].set_title('Normalizacja L2 (Długość wektora euklidesowego = 1)')

plt.tight_layout()
plt.show()


"""
Cechy banknotów: charakteryzują się różną skalą. Największą zmienność wykazuje atrybut skewness gdzie najwyższa wariancja wynosiła 34.45 a rozstęp tej wynosił 26.72
Średnia: dla klasy decyzyjnej class wynosi ok. 0.44 co wskazuje na proporcję około 44% dla przypadków pozytywnych (1) do 56% negatywnych (0) w naszym zbiorze danych.
Przeskalowanie Min-Max: Skutecznie ujednoliciło zakresy wszystkich atrybutów do sztywnego przedziału
Standaryzacja: Zadziałała poprawnie
Normalizacja L1 i L2: potraktowały wiersze ako niezależne wektory, transformując je tak, by suma ich modułów była równa 1 (L1) lub by miały długość 1 (L2).
"""