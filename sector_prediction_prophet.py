# ================================ Importy i ustawienia ================================================================================

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import kagglehub

from sklearn.metrics import mean_squared_error
from prophet import Prophet

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 1000)
plt.rcParams.update({'font.size': 8})


# ================================ Zaimporotwanie stałych globalnych i funkcji ========================================================

# Import stałych globalnych
from resources.config import *

# Import funkcji z plików zewnętrznych
from defs.clean_data import clean_data
from defs.optuna_objective_prophet import objective
from defs.statistics import statistics


# ================================ Dane źródłowe z kaggle ================================
# Zestaw danych World Stock Prices

filename = 'World-Stock-Prices-Dataset.csv'
path = kagglehub.dataset_download('nelgiriyewithana/world-stock-prices-daily-updating')
fullpath = os.path.join(path, filename)
data = pd.read_csv(fullpath)


# ================================ Wstępna analiza danych ===================================================================

data['Date'] = pd.to_datetime(data['Date'], utc=True)   # Zmiana formatu daty aby móc analizować szeregi czasowe
data['Date'] = data['Date'].dt.tz_localize(None)       # Usunięcie strefy czasowej
newest_date = data.iloc[1]['Date'].date()               # Najnowsza data w pobranym zbiorze danych, tylko data bez godziny

print('\n~~~~~~~~~~~~~~~~~~~~ WITAJ W ANALIZIE DANYCH WORLD STOCK PRICES DATASET! ~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Najnowsze dane z dnia: ', newest_date, ' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# Połączenie niektórych kategorii w jedną z uwagi na podobieństwo i małą liczność spółek w danych grupach
data['Industry_Tag'] = data['Industry_Tag'].replace('financial services', 'finance')
data['Industry_Tag'] = data['Industry_Tag'].replace('food & beverage', 'food')
data['Industry_Tag'] = data['Industry_Tag'].replace('music', 'entertainment')

industries = data['Industry_Tag'].unique()

print('\n Sektory biorące udział w analizie: \n')

# Utworzenie tabelki z unikalnymi spółkami per sektor
data_industries = data.groupby('Industry_Tag')['Brand_Name']\
    .apply(lambda x: sorted(list(set(x))))\
    .reset_index()

# Zmiana nazwy kolumn dla czytelności
data_industries.columns = ['Industry_Tag', 'Brand_Names']

print(data_industries)


# Wybór sektora
while True:
    sektor_Y = input('Napisz nazwę sektora do analizy: ')
    if sektor_Y in industries:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sektor analizowany: ', sektor_Y, '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        break
    else:
        print('!!! Niepoprawne słowo !!! Spróbuj ponownie.')


# Wyczyszczenie danych duplikujących się:
data = data.drop_duplicates(subset=['Date', 'Brand_Name'])
statistics(data, 'dane źródłowe z csv')

print('\n~~~~~~~~~~~~~~~~~~~~ PRZYGOTOWANIE DANYCH PER SEKTOR ~~~~~~~~~~~~~~~~~~~~')

# Obliczenie średniej ważonej ceny zamknięcia dla każdego sektora (Industry_Tag)
data_industry = (
    data.assign(weighted_close=data['Close'] * data['Volume'])        # Dodanie kolumny z wagą
    .groupby(['Date', 'Industry_Tag'])[['weighted_close', 'Volume']]  # Grupowanie
    .sum()                                                            # Sumowanie wag i wolumenów w grupach
    .eval('weighted_close / Volume')                                  # Obliczenie średniej ważonej
    .unstack()                                                        # Przekształcenie grup w kolumny dla każdego 'Industry_Tag'
)

data_industry = data_industry.reset_index()

print('\nŚrednia ważona cen zamknięcia dla każdego sektora (dziennie) - najnowsze dane:')
print(data_industry.tail(10))

clean_data(data_industry,TRAIN_SIZE+VAL_SIZE+TEST_SIZE)  #Usunięcie sektorów z niewystarczającą ilością danych

statistics(data_industry, 'dzienne indeksy sektorowe')

if sektor_Y not in data_industry.columns:
    print('\nUwaga! Sektor wybrany do analizy nie posiada wystarczającej ilości danych. Wybierz inny sektor!')

# ================================= Analiza korelacji =================================

# Korelacja
numeric_data = data_industry.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()


plt.figure(figsize=(10, 8))
macierz = sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', fmt='.2f', linewidths=0.5,
                          annot_kws={'size': 9})

# Pogrubienie wybranej nazwy na osi X
for label in macierz.get_xticklabels():
    if label.get_text() == sektor_Y:
        label.set_weight('bold')
        label.set_fontsize(10)

# Pogrubienie wybranej nazwy na osi Y
for label in macierz.get_yticklabels():
    if label.get_text() == sektor_Y:
        label.set_weight('bold')
        label.set_fontsize(10)

plt.title(f'Macierz korelacji między sektorami')
plt.tight_layout()
plt.show()

# ============================= Wykres liniowy =============================

# Wykres liniowy tylko dla wybranego okna czasowego (train+test)
data_industry_shortened = data_industry.tail(TRAIN_SIZE + VAL_SIZE + TEST_SIZE)

plt.figure(figsize=(10, 8))
palette = sns.color_palette('tab20c', n_colors=len(data_industry_shortened.columns[1:]))

# Iterowanie po zmiennych i ich wykres
for i, column in enumerate(data_industry_shortened.columns[1:]):  # iterujemy po indeksie oraz po nazwie kolumny
    if column == sektor_Y:
        # Wyróżnienie koloru i grubości dla wybranej zmiennej
        plt.plot(data_industry_shortened['Date'], data_industry_shortened[column], label=column, color='black', linewidth=2.5)
    else:
        plt.plot(data_industry_shortened['Date'], data_industry_shortened[column], label=column, color=palette[i])

# Dostosowanie wykresu
plt.title('Indeksy sektorowe')
plt.xlabel('Data')
plt.ylabel('Wartość USD')
plt.legend(title='Sektory', loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.xticks(rotation=45)

# Wyświetlanie wykresu
plt.tight_layout()
plt.show()


#=============================== Wybór zmiennych objaśniających Lasso Regression (regresja z regularyzacją)===================================================================
# Lasso zmniejsza wagi mniej istotnych zmiennych do zera.
from sklearn.linear_model import LassoCV

# Przygotowanie danych
X = data_industry_shortened.drop(columns=[sektor_Y,'Date'])  # Wszystkie kolumny oprócz Y
y = data_industry_shortened[sektor_Y]

# Model Lasso
lasso = LassoCV(cv=5)
lasso.fit(X, y)

# Sprawdzenie ważności zmiennych
importance = lasso.coef_

importance_df = pd.DataFrame({
    'Feature': X.columns,       # Nazwy zmiennych
    'Coefficient': lasso.coef_  # Współczynniki
})


# Lista wybranych zmiennych
print(importance_df['Coefficient'])
selected_features =  importance_df[abs(importance_df['Coefficient']) >= 0.0001]
selected_feature_names = selected_features['Feature'].tolist()

columns_to_keep = ['Date', sektor_Y] + selected_feature_names

# ============================= Przygotowanie zbioru do modelowania ================================================================
X = data_industry_shortened[columns_to_keep]
X = X.rename(columns={'Date': 'ds'})
X = X.rename(columns={sektor_Y: 'y'})
statistics(X,'x')


#=============================== Podział na zbiór treningowy i testowy i optymalizacja modelu===================================================================
# Ze względu na analizę szeregu czasowego nie możemy podzielić zbioru w sposób losowy
# Rozmiary TRAIN_SIZE i TEST_SIZE podano na początku skryptu

# Zbiór treningowy
X_train = X.iloc[:TRAIN_SIZE]

# Zbiór walidacyjny
X_val = X.iloc[TRAIN_SIZE: TRAIN_SIZE + VAL_SIZE]

# Zbiór testowy
X_test = X.iloc[TRAIN_SIZE + VAL_SIZE: TRAIN_SIZE + VAL_SIZE + TEST_SIZE]


# Tworzenie wykresu
plt.figure(figsize=(12,6))
plt.plot(X['ds'], X['y'], label='Indeksy sektorowe', color='blue')

# Dodanie pionowych linii oddzielających zbiory
plt.axvline(X_train['ds'].iloc[-1], color='red', linestyle='dashed', label='Podział Train/Val')
plt.axvline(X_val['ds'].iloc[-1], color='green', linestyle='dashed', label='Podział Val/Test')

# Dodanie etykiet i legendy
plt.title("Podział na zbiory: Treningowy, Walidacyjny i Testowy")
plt.xlabel("Data")
plt.ylabel("Wartość USD")
plt.legend()
plt.grid(True)

# Wyświetlenie wykresu
plt.show()

# Uruchomienie Optuna
study = optuna.create_study(direction='minimize')  # Minimalizacja RMSE
study.optimize(lambda trial: objective(trial, X_train, X_val), n_trials=OPTUNA_ITERATIONS)

# Najlepsze hiperparametry
print("\nNajlepsze parametry:", study.best_params)

# Trenowanie finalnego modelu Prophet z najlepszymi hiperparametrami
best_params = study.best_params

# Finalny model Prophet
final_model = Prophet(
    interval_width=0.95,
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    holidays_prior_scale=best_params['holidays_prior_scale'],
    n_changepoints=best_params['n_changepoints'],
    weekly_seasonality=best_params['weekly_seasonality'],
    yearly_seasonality=best_params['yearly_seasonality']
)

# Lista regresorów - wszystkie kolumny poza 'ds' i 'y'
regressors = [col for col in X_train.columns if col not in ['ds', 'y']]

# Dodanie regresorów do modelu Prophet
for reg in regressors:
    final_model.add_regressor(reg)


final_model.fit(X_train)

# Prognoza na zbiorze walidacyjnym
future_val = X_val[['ds'] + regressors]
forecast_val = final_model.predict(future_val)

X_val = X_val.reset_index(drop=True)
forecast_val = forecast_val.reset_index(drop=True)

rmse_val = mean_squared_error(X_val['y'], forecast_val['yhat'])
mape_val=np.mean(np.abs((X_val['y'] - forecast_val['yhat']) / X_val['y'])) * 100
print(f"\nRMSE dla walidacji: {rmse_val}")
print(f"MAPE dla walidacji: {mape_val:.2f}%")

# Prognoza na zbiorze testowym
future_test = X_test[['ds'] + regressors]
forecast_test = final_model.predict(future_test)

X_test = X_test.reset_index(drop=True)
forecast_test = forecast_test.reset_index(drop=True)

rmse_test = mean_squared_error(X_test['y'], forecast_test['yhat'])
mape_test=np.mean(np.abs((X_test['y'] - forecast_test['yhat']) / X_test['y'])) * 100
print(f"\nRMSE dla testu: {rmse_test}")
print(f"MAPE dla testu: {mape_test:.2f}%")

# ====================================== Prognoza zmiennych objaśniających ======================================

X_for_regressors=X.iloc[-(TEST_SIZE + VAL_SIZE):]


# Lista zmiennych objaśniających (bez 'ds' i 'y')
regressors = [col for col in X_for_regressors.columns if col not in ['ds', 'y']]

# Słownik do przechowywania prognoz dla zmiennych objaśniających
regressor_forecasts = {}

# Iteracja po każdej zmiennej objaśniającej
for regressor in regressors:
    print(f"\nPrognozowanie zmiennej objaśniającej: {regressor}")

    # Tworzymy kopię danych tylko dla tej zmiennej
    df_regressor = X_for_regressors[['ds', regressor]].dropna()  # Usuwamy ewentualne brakujące wartości

    # Tworzenie i dopasowanie modelu Prophet
    model_regressor = Prophet(
        interval_width = 0.95,
        weekly_seasonality = True)
    model_regressor.fit(df_regressor.rename(columns={regressor: 'y'}))  # Prophet wymaga 'y' jako nazwy zmiennej

    # Tworzenie przyszłych dat
    future_regressor = model_regressor.make_future_dataframe(periods=FORECAST)
    # Usuwanie weekendów (soboty i niedziele)
    future_regressor = future_regressor[~future_regressor['ds'].dt.weekday.isin([5, 6])]

    # Prognoza
    forecast_regressor = model_regressor.predict(future_regressor)

    # Zapisujemy prognozy
    regressor_forecasts[regressor] = forecast_regressor[['ds', 'yhat']].rename(columns={'yhat': regressor})


# Tworzenie głównego zbioru przyszłych dat
future_final = list(regressor_forecasts.values())[0][['ds']].copy()  # Pobieramy kolumnę 'ds' z pierwszego forecastu

# Iterujemy przez wszystkie prognozy zmiennych objaśniających i dodajemy do future_final
for regressor, forecast_df in regressor_forecasts.items():
    future_final = future_final.merge(forecast_df, on='ds', how='left')

# Wykorzystanie przewidzianych wartości zmiennych objaśniających do prognozy finalnej
future_forecast = final_model.predict(future_final)

X_val_test=X.iloc[-(VAL_SIZE+TEST_SIZE):]

# Przygotowanie danych do wykresu (tylko ostatnie VAL+TEST SIZE + FORECAST)
X_dates=X_val_test['ds'].values
X_dates_pred = future_final['ds'].values
y_test_true = X_val_test['y'].values  # Rzeczywiste wartości
y_test_pred = future_forecast['yhat'].values  # Prognozy modelu
y_test_upper = future_forecast['yhat_upper'].values  # Górny przedział ufności
y_test_lower = future_forecast['yhat_lower'].values  # Dolny przedział ufności

X_dates_all=X['ds'].values
y_test_true_all = X['y'].values  # Rzeczywiste wartości (cały zakres)


# Tworzenie wykresu 1
plt.figure(figsize=(12, 6))
plt.plot(X_dates_all, y_test_true_all, label="Dane rzeczywiste", marker='o')
plt.plot(X_dates_pred, y_test_pred, label="Dane estymowane przez model", linestyle='--')
plt.fill_between(X_dates_pred, y_test_lower, y_test_upper, color='gray', alpha=0.2, label="Przedział ufności")
plt.title("Porównanie danych rzeczywistych i prognoz modelu")
plt.xlabel("Data")
plt.ylabel(f"Wartość dla sektora {sektor_Y}")
plt.legend()
plt.grid()
plt.show()


# Tworzenie wykresu 2
plt.figure(figsize=(12, 6))
plt.plot(X_dates, y_test_true, label="Dane rzeczywiste", marker='o')
plt.plot(X_dates_pred, y_test_pred, label="Dane estymowane przez model", linestyle='--')
plt.fill_between(X_dates_pred, y_test_lower, y_test_upper, color='gray', alpha=0.2, label="Przedział ufności")
plt.title("Porównanie danych rzeczywistych i prognoz modelu (najnowsze dane i prognoza)")
plt.xlabel("Data")
plt.ylabel(f"Wartość dla sektora {sektor_Y}")
plt.legend()
plt.grid()
plt.show()

# Tworzenie wykresu kolumnowego skumulowanego dla wpływu regresorów
import numpy as np

# Lista regresorów (filtrowanie odpowiednich kolumn)
regressors = [col for col in future_forecast.columns if col not in ['ds', 'yhat'] and 'lower' not in col and 'upper' not in col and 'extra_regressors' not in col and 'multiplicative_terms' not in col]

# Przygotowanie danych do wykresu skumulowanego
stacked_data = np.array([future_forecast[regressor].values for regressor in regressors])
stacked_data_cumsum = np.cumsum(stacked_data, axis=0)

# Tworzenie wykresu
plt.figure(figsize=(12, 6))

for i, regressor in enumerate(regressors):
    if i == 0:
        plt.bar(future_forecast['ds'], stacked_data[i], label=f'Wpływ {regressor}')
    else:
        plt.bar(future_forecast['ds'], stacked_data[i], bottom=stacked_data_cumsum[i - 1], label=f'Wpływ {regressor}')

plt.title("Wpływ regresorów na prognozę (skumulowany)")
plt.xlabel("Data")
plt.ylabel("Wpływ")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.show()