import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter


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


