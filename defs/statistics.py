import pandas as pd

# Statystyki dla DataFrame'a
def statistics(dataset: pd.DataFrame, name: str):

    # Statystyki opisowe pandas
    data_stats = dataset.describe(include='all').round(2)

    # Dodatkowe statystyki
    data_additional_info = pd.DataFrame({
        'Data Types': dataset.dtypes,
        'Missing Values': dataset.isnull().sum(),
        'Missing Values (%)': (dataset.isnull().sum() / len(dataset) * 100).round(1)
    })

    # Scalanie wszystkich statystyk
    all_stats = pd.concat([data_stats, data_additional_info.T])
    print('\nStatystyki dla danych: ', name)
    print(all_stats)