import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.ensemble import IsolationForest

path = kagglehub.dataset_download("ritesaluja/bank-note-authentication-uci-data")
df = pd.read_csv(f"{path}/BankNote_Authentication.csv")


# Metoda 1: Z-score
z_scores = np.abs((df.drop(columns=['class']) - df.drop(columns=['class']).mean()) / df.drop(columns=['class']).std())
outliers_z_score = (z_scores > 3).any(axis=1)
print(f"Liczba outlierów wykrytych metodą Z-score: {outliers_z_score.sum()}")
# print(f"Outliery wykryte metodą Z-score:\n{df[outliers_z_score]}")

# Metoda 2: IQR
Q1 = df.drop(columns=['class']).quantile(0.25)
Q3 = df.drop(columns=['class']).quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((df.drop(columns=['class']) < (Q1 - 1.5 * IQR)) | (df.drop(columns=['class']) > (Q3 + 1.5 * IQR))).any(axis=1)
print(f"Liczba outlierów wykrytych metodą IQR: {outliers_iqr.sum()}")
# print(f"Outliery wykryte metodą IQR:\n{df[outliers_iqr]}")

# Metoda 3: Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers_iso_forest = iso_forest.fit_predict(df.drop(columns=['class'])) == -1
print(f"Liczba outlierów wykrytych metodą Isolation Forest: {outliers_iso_forest.sum()}")
# print(f"Outliery wykryte metodą Isolation Forest:\n{df[outliers_iso_forest]}")


# Porównanie wyników
outliers_summary = pd.DataFrame({
    'Metoda': ['Z-score', 'IQR', 'Isolation Forest'],
    'Liczba Outlierów': [outliers_z_score.sum(), outliers_iqr.sum(), outliers_iso_forest.sum()]
})
print("\nPodsumowanie wykrytych outlierów:")
print(outliers_summary)

# Wykres
sns.set_theme(style="whitegrid", context="talk")
fig = plt.figure(figsize=(22, 14))

gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, :])

feature_x = 'variance'
feature_y = 'skewness'

def plot_outliers_scatter(ax, outliers_mask, title, count, color):
    sns.scatterplot(data=df[~outliers_mask], x=feature_x, y=feature_y, 
                    color='#bdc3c7', s=30, ax=ax, label='W normie', alpha=0.5, edgecolor=None)
    sns.scatterplot(data=df[outliers_mask], x=feature_x, y=feature_y, 
                    color=color, s=80, ax=ax, label='Outliery', edgecolor='black', linewidth=1, marker='X')
    
    ax.set_title(f"{title}\n(Wykryto: {count})", fontsize=16, fontweight='bold', pad=10)
    ax.set_xlabel('Variance (Wariancja)', fontsize=13)
    ax.set_ylabel('Skewness (Skośność)', fontsize=13)
    ax.legend(loc='upper right', frameon=True, fontsize=11)

plot_outliers_scatter(ax1, outliers_z_score, 'Z-score', outliers_z_score.sum(), '#ff1900') # Czerwony
plot_outliers_scatter(ax2, outliers_iqr, 'IQR', outliers_iqr.sum(), '#ff9900') # Pomarańczowy
plot_outliers_scatter(ax3, outliers_iso_forest, 'Isolation Forest', outliers_iso_forest.sum(), '#b700ff') # Fioletowy

methods = outliers_summary['Metoda']
counts = outliers_summary['Liczba Outlierów']
colors = ["#ff1900", "#ff9900", "#b700ff"]

sns.barplot(x=methods, y=counts, ax=ax4, palette=colors)
ax4.set_title('Zestawienie ilościowe wykrytych anomalii', fontsize=18, fontweight='bold', pad=15)
ax4.set_ylabel('Liczba usuniętych próbek', fontsize=14)
ax4.set_xlabel('Metoda detekcji', fontsize=14)

ax4.set_ylim(0, max(counts) * 1.2) 

for i, count in enumerate(counts):
    ax4.text(i, count + (max(counts) * 0.02), str(count), 
             ha='center', va='bottom', fontsize=16, fontweight='bold', color='black')

plt.show()


"""
Wyraźnie widać, że metoda IQR i wycięła znacznie więcej próbek (92) niż Z-score (36) czy Isolation Forest (69).
Na wykresach punktowych widać, że punkty w IQR są dużo gęściej wokół głównego klastra danych.
Z-score i IQR opierając się na sztywnych granicach, przez co mogą usuwać całkiem poprawne rzadkie przypadki.

Isolation Forest wygrywa w tym zestawieniu
Nie tnie on wszystkiego tylko szuka wielowymiarowych anomalii, co jest bardziej elastyczne i lepiej dopasowane do rzeczywistych danych.
Dlatego Isolation Forest jest tutaj najbardziej trafnym i wiarygodnym wyborem do oczyszczenia tego konkretnego zbioru danych.
"""