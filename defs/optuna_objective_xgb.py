import numpy as np  
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error

# Funkcja celu dla Optuny do optymalizacji hiperparametrów
def objective(trial, X: pd.DataFrame, y: pd.DataFrame, train_size: int, val_size: int, test_size: int):

    # Przestrzeń poszukiwań hiperparametrów
    # -------------------------------------
    # objective         Funkcja kosztu dla regresji
    # eval_metric       RMSE do oceny modelu
    # max_depth         Max. głębokość drzewa decyzyjnego
    # learning_rate     Szybkość uczenia (od, do), LR wpływa na długość kroku iteracji, skala logarytmiczna
    # gamma             Minimalna redukcja kosztu, wymagana do podziału węzła, im wyższe tym mniej podziałów (jeśli GAIN<Gamma to nie idz dalej)
    # reg_alpha         Redukcja wag cech nieistotnych, pomaga w selekcji cech (wagi nieistotnych cech są redukowane)
    # reg_lambda        Ograniczenie wielkości wag, im lambda większa tym Similarity Score mniejszy, model się uogólnia (zapobiegamy overfittingowi)

    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 20,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'gamma': 1,
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 5)
    }

    rmse_scores = []

    for X_train, y_train, X_val, y_val, X_test, y_test in sliding_window_split(X, y, train_size, val_size, test_size):
        assert X_train.shape[0] > 0 and y_train.shape[0] > 0, 'Puste dane treningowe!'
        assert X_val.shape[0] > 0 and y_val.shape[0] > 0, 'Puste dane walidacyjne!'
        assert len(y_train.shape) == 1 and len(y_val.shape) == 1, 'Etykiety muszą być wektorami 1D!'

        model = xgb.XGBRegressor(**param, early_stopping_rounds=10)

        # Trening na zbiorze treningowym
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Predykcja na zbiorze testowym
        y_pred = model.predict(X_test)

        # Obliczanie RMSE na zbiorze testowym
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)

        # Średnia wartość RMSE ze wszystkich podziałów
        mean_rmse = np.mean(rmse_scores)

        return mean_rmse
    

# Dla Optuny i każdej iteracji zwraca zbiory treningowy, walidacyjny i testowy
def sliding_window_split(X: pd.DataFrame, y: pd.DataFrame, train_size: int, val_size: int, test_size: int):

    # Pętla przesuwająca się
    # Okno przesuwa się w każdej iteracji, w miarę jak start przesuwa się do przodu.
    for start in range(len(X) - train_size - val_size - test_size + 1):
        # Zbiór treningowy
        X_train = X[start: start + train_size]
        y_train = y[start: start + train_size]

        # Zbiór walidacyjny
        X_val = X[start + train_size: start + train_size + val_size]
        y_val = y[start + train_size: start + train_size + val_size]

        # Zbiór testowy
        X_test = X[start + train_size + val_size: start + train_size + val_size + test_size]
        y_test = y[start + train_size + val_size: start + train_size + val_size + test_size]

        # Zwraca kolejne wyniki, jeden po drugim
        yield X_train, y_train, X_val, y_val, X_test, y_test
