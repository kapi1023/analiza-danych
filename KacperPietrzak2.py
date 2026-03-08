
import pandas as pd

def basic_inspection(df):
    print("---PODSTAWOWE METODY INSPEKCJI DANYCH---")
    print(df.head(3))
    print(df.info())
    print(df.shape)
    print(df.columns)

def sort(df):
    print("---SORTOWANIE DANYCH---")
    print(df.sort_values(by='Lipiec', ascending=True)[['Miasto', 'Lipiec']])
    print(df.sort_values(by='Styczeń', ascending=False)[['Miasto', 'Styczeń']])
    print(df.sort_values(by=['Lipiec', 'Kwiecień'], ascending=[False, True])[['Miasto', 'Lipiec', 'Kwiecień']])

def select_columns(df):
    print("---WYBIERANIE KOLUMN---")
    print(df['Miasto'])
    print(df[['Miasto', 'Lipiec']])
    columns_to_select = ['Miasto', 'Styczeń', 'Lipiec']
    print(df[columns_to_select])

def filter_rows(df):
    print("---FILTROWANIE WIERSZY---")
    print(df[df['Lipiec'] > 19][['Miasto', 'Lipiec']])
    print(df[df['Styczeń'] >= -1][['Miasto', 'Styczeń']])
    print(df[df['Kwiecień'] == 10][['Miasto', 'Kwiecień']])
    print(df[(df['Lipiec'] == 20) | (df['Styczeń'] == -1)][['Miasto', 'Lipiec', 'Styczeń']])

def filter_rows_combined(df):
    print("---FILTROWANIE WIERSZY ZŁOŻONE---")
    print(df[(df['Lipiec'] > 18) & (df['Styczeń'] >= -2)][['Miasto', 'Lipiec', 'Styczeń']])
    print(df[df['Miasto'].isin(['Warszawa', 'Kraków', 'Gdańsk'])][['Miasto', 'Styczeń', 'Kwiecień', 'Lipiec', 'Październik']])
    print(df[(df['Lipiec'] >= 18) & (df['Lipiec'] <= 20)][['Miasto', 'Lipiec']])   

def add_columns(df):
    print("---DODAWANIE NOWYCH KOLUMN---")
    df['Średnia_roczna'] = df[['Styczeń', 'Kwiecień', 'Lipiec', 'Październik']].mean(axis=1)
    df['Amplituda'] = df['Lipiec'] - df['Styczeń']
    df['Ciepłe_miasto'] = df['Średnia_roczna'] > 11
    print(df[['Miasto', 'Średnia_roczna', 'Amplituda', 'Ciepłe_miasto']])

temperatura = pd.DataFrame({
   'Miasto': ['Warszawa', 'Kraków', 'Gdańsk', 'Wrocław', 'Poznań', 'Łódź'],
   'Styczeń': [-2, -1, 0, -1, -2, -3],
   'Kwiecień': [9, 10, 8, 11, 10, 9],
   'Lipiec': [19, 20, 17, 21, 19, 20],
   'Październik': [9, 9, 10, 10, 9, 8]
})

basic_inspection(temperatura)
sort(temperatura)
select_columns(temperatura)
filter_rows(temperatura)
filter_rows_combined(temperatura)
add_columns(temperatura)

