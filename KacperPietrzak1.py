import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.impute import KNNImputer


temperatura = pd.DataFrame({
   'Miasto': ['Warszawa', 'Kraków', 'Gdańsk', 'Wrocław', 'Poznań', 'Łódź'],
   'Styczeń': [-2, -1, 0, -1, -2, -3],
   'Kwiecień': [9, 10, 8, 11, 10, 9],
   'Lipiec': [19, 20, 17, 21, 19, 20],
   'Październik': [9, 9, 10, 10, 9, 8]
})

print("--- Inspekcja DataFrame ---")
print(temperatura)

df_plot = temperatura.set_index('Miasto')
plt.figure(figsize=(10, 6))
sns.heatmap(df_plot, annot=True, cmap='coolwarm', center=10, cbar_kws={'label': 'Temperatura [°C]'})

plt.title('Średnie temperatury w wybranych miastach Polski')
plt.xlabel('Miesiąc')
plt.ylabel('Miasto')
plt.tight_layout()
plt.show()

months = ['Styczeń', 'Kwiecień', 'Lipiec', 'Październik']
mins = temperatura[months].min()
maxs = temperatura[months].max()

synthetic_data = np.random.uniform(low=mins, high=maxs, size=(len(temperatura), len(months)))
synthetic_df = pd.DataFrame(synthetic_data, columns=months, index=temperatura['Miasto'])
print("--- Syntetyczne dane ---")
print(synthetic_df)

plt.figure(figsize=(10, 6))
sns.heatmap(synthetic_df, annot=True, cmap='coolwarm', center=10, cbar_kws={'label': 'Temperatura [°C]'})
plt.title('Syntetyczne średnie temperatury w wybranych miastach Polski')
plt.xlabel('Miesiąc')
plt.ylabel('Miasto')
plt.tight_layout()
plt.show()

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv'
iris_df = pd.read_csv(url)
print("--- Dane z repozytorium ---")
print(iris_df)

sns.pairplot(iris_df, hue='species', palette='bright')
plt.suptitle('Relacje cech z podziałem na gatunki', y=1.02)
plt.show()


path = kagglehub.dataset_download("ritesaluja/bank-note-authentication-uci-data")
df = pd.read_csv(f"{path}/BankNote_Authentication.csv")
print("--- Dane z Kaggle ---")
print(df)
df.to_csv("data.csv", index=False)

df = pd.read_csv("data.csv")
print("--- Dane z pliku CSV ---")
print(df)

sns.pairplot(df, hue='class', palette='Set1',)
plt.suptitle('Relacje cech z podziałem na klasy', y=1.02)
plt.show()



#Zadanie zajęcia 22.03.2025
print("--- Zadanie: Zadanie zajęcia 22.03.2025 ---")

df_original = df.copy()
np.random.seed(42) 
missing_mask_var = np.random.rand(len(df)) < 0.25
missing_mask_ent = np.random.rand(len(df)) < 0.25

df.loc[missing_mask_var, 'variance'] = np.nan
df.loc[missing_mask_ent, 'entropy'] = np.nan

print(f"Liczba braków danych przed imputacją (wariancja): {df['variance'].isnull().sum()}")

k_values = [5, 20, 40]
imputed_dfs = {}

for k in k_values:
    knn_imputer = KNNImputer(n_neighbors=k)
    imputed_dfs[k] = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

fig = plt.figure(figsize=(16, 12))

plt.subplot(2, 1, 1)
sns.kdeplot(df_original['variance'], label='Oryginalne', color='black', linewidth=3)
sns.kdeplot(imputed_dfs[5]['variance'], label='KNN (k=5)', linestyle='--', color='#e74c3c', linewidth=2)
sns.kdeplot(imputed_dfs[20]['variance'], label='KNN (k=20)', linestyle='-.', color='#3498db', linewidth=2)
sns.kdeplot(imputed_dfs[40]['variance'], label='KNN (k=40)', linestyle=':', color='#2ecc71', linewidth=2)
plt.title('Porównanie gęstości rozkładu zmiennej Variance w zależności od hiperparametru k')
plt.xlabel('Variance (Wariancja)')
plt.ylabel('Estymator gęstości jądrowej (KDE)')
plt.legend()

colors = ['#e74c3c', '#3498db', '#2ecc71']

for idx, k in enumerate(k_values):
    plt.subplot(2, 3, 4 + idx)
    df_temp = imputed_dfs[k].copy()
    df_temp['Status obserwacji'] = np.where(missing_mask_var, 'Imputowana', 'Oryginalna')
    
    sns.scatterplot(
        data=df_temp, 
        x='skewness', 
        y='variance', 
        hue='Status obserwacji', 
        palette={'Oryginalna': '#bdc3c7', 'Imputowana': colors[idx]}, 
        alpha=0.7,
        s=30
    )
    plt.title(f'Rozkład punktów dla k={k}')
    plt.xlabel('Skewness')
    if idx == 0:
        plt.ylabel('Variance')
    else:
        plt.ylabel('')
    plt.legend(loc='lower right', fontsize='small')

plt.tight_layout()
plt.show()

"""
Wprowadzenie danych: Usuniecie 25% danych z kolumn 'variance' i 'entropy' w sposób losowy,
aby zasymulować brakujące dane. 
Imputacja KNN: Zastosowanie KNNImputer z różnymi wartościami k (5, 20, 40) do imputacji brakujących danych.
Ocena jakości imputacji: Porównanie gęstości rozkładu (KDE) dla oryginalnych danych z imputowanymi danymi dla każdej wartości k.
Dodatkowo, wizualizacja rozkładu punktów dla zmiennych 'skewness' i 'variance', z podziałem na obserwacje oryginalne i imputowane, aby ocenić,
jak dobrze imputacja zachowuje relacje między zmiennymi.
Celem jest zrozumienie wpływu wyboru hiperparametru k na jakość imputacji oraz na zachowanie struktury danych. 
"""

begin = math.floor(min(df['variance'].min(), df['entropy'].min()))
end = math.ceil(max(df['variance'].max(), df['entropy'].max()))
print(f"Zakres wartości dla imputacji: [{begin}, {end}]")

