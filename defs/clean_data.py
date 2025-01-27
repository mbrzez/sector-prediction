import pandas as pd

# Sprawdzenie ilości danych per sektor
def clean_data(df: pd.DataFrame, threshold: int):
    # Lista do przechowywania kolumn do usunięcia
    columns_to_remove = []

    # Iteracja przez kolumny i liczenie niepustych wartości
    for column in df.columns:
        non_empty_count = df[column].count()  # Liczba niepustych wierszy w kolumnie
        if non_empty_count < threshold:
            columns_to_remove.append(column)  # Dodanie kolumny do listy

    # Usuwanie kolumn
    df = df.drop(columns=columns_to_remove, inplace=True)

    # Wyświetlenie informacji o usuniętych kolumnach
    if columns_to_remove:
        print(f'\nUsunięto następujące kolumny z niewystarczającą ilością danych (<{threshold} wierszy):')
        print(columns_to_remove)
    else:
        print('\nNie usunięto żadnych kolumn, wszystkie spełniają wymagania.')

    return df