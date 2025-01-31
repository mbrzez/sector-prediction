# Stałe globalne dla projektu

LAG_WEEKS = 5               # Liczba tygodni wstecz, które mogą wpływać na prognozę
TRAIN_SIZE = 400            # Liczba próbek w oknie treningowym
VAL_SIZE = 50               # Liczba próbek w oknie walidacyjnym
TEST_SIZE = 50              # Liczba próbek w oknie testowym

OPTUNA_ITERATIONS = 100     # Liczba iteracji Optuny do optymalizacji hiperparametrów

FORECAST = 30               # Liczba dni prognozy (tylko dla modelu Prophet)