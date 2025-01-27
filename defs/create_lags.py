import pandas as pd

# Funkcja tworząca dane opóźnione
def create_lags(df: pd.DataFrame, lags: int, sector: str):
    lagged_data = []
    for column in df.columns:
        for lag in range(1, lags + 1):
            lagged_col = df[column].shift(lag)  # Zmienna opóźniona
            lagged_data.append(lagged_col.rename(f'{column}_lag_{lag}'))

    # Połączenie wszystkich Series w jeden DataFrame
    lagged_data_df = pd.concat(lagged_data, axis=1)

    # Zmienna Y nieopóźniona
    df_y = df[sector]

    # Połączenie
    df_lagged = pd.concat([df_y, lagged_data_df], axis=1)  # tylko zmienne opóźnione (nie bierzemy zmiennych z aktualnego dnia) + Y nieopoźniony (do prognozy)
    df_lagged = df_lagged.dropna()

    return df_lagged