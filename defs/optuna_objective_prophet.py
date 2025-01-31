import pandas as pd
import numpy as np
from prophet import Prophet

from sklearn.metrics import mean_squared_error

# Funkcja celu dla Optuny do optymalizacji hiperparametrów
def objective(trial, df_train: pd.DataFrame, df_test: pd.DataFrame):
    # Elastyczność trendu
    changepoint_prior = trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True)

    # Waga sezonowości
    seasonality_prior = trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True)
    
    # Wpływ świąt
    holidays_prior = trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True)

    # Liczba punktów zmiany trendu
    n_changepoints = trial.suggest_int('n_changepoints', 5, 50)
    weekly_seasonality = trial.suggest_categorical('weekly_seasonality', [True, False])
    yearly_seasonality = trial.suggest_categorical('yearly_seasonality', [True, False])

    # Przestrzeń poszukiwań hiperparametrów
    # -------------------------------------
    param = {
        'interval_width': 0.95,
        'changepoint_prior_scale': changepoint_prior,
        'seasonality_prior_scale': seasonality_prior,
        'holidays_prior_scale': holidays_prior,
        'n_changepoints': n_changepoints,
        'weekly_seasonality': weekly_seasonality,
        'yearly_seasonality': yearly_seasonality
    }

    # Tworzenie modelu Prophet z optymalizowanymi hiperparametrami
    model = Prophet(**param)

    # # Tworzenie modelu Prophet z optymalizowanymi hiperparametrami
    # model = Prophet(**param)

    train_data = df_train

    # Dodanie zmiennych objaśniających
    for col in train_data.columns:
        if col not in ['ds', 'y']:
            model.add_regressor(col)

    # Trening modelu
    model.fit(train_data)

    # Pobranie listy wszystkich regresorów (czyli kolumn poza 'ds' i 'y')
    regressors = [col for col in df_train.columns if col not in ['ds', 'y']]

    # Prognoza na zbiorze walidacyjnym (musi zawierać wszystkie regresory!)
    future = df_test[['ds'] + regressors]
    forecast = model.predict(future)

    # Obliczenie RMSE jako metryki oceny
    rmse = np.sqrt(mean_squared_error(df_test['y'], forecast['yhat']))

    return rmse