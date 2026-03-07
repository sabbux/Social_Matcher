import os
import sys

# Trova la cartella corrente (training) e risali a "xgboost"
current_dir = os.path.dirname(os.path.abspath(__file__))
xgboost_dir = os.path.dirname(current_dir)

# Risaliamo anche alla root del progetto nella cartella "data" globale
project_root = os.path.dirname(xgboost_dir)

# Diciamo a Python di considerare "xgboost" come punto di partenza per gli import
sys.path.insert(0, xgboost_dir)

# Importiamo normalmente Pandas, ecc.
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Importiamo i moduli partendo da dentro la cartella xgboost
from utils.preprocessing import build_xgb_preprocessor 
from model_config.xgboost_config import get_xgb_model 

resource_dir = os.path.join(xgboost_dir, "resources")
os.makedirs(resource_dir, exist_ok=True)

# Percorsi completi per output
MODEL_NAME = os.path.join(resource_dir, 'social_matcher_model.pkl')
X_test_path = os.path.join(resource_dir, "X_test.pkl")
y_test_path = os.path.join(resource_dir, "y_test.pkl")


# Caricamento dataset
print("Caricamento dataset...")
df = pd.read_csv(os.path.join(project_root, "data", "processed", "social_matcher.csv"))

# Separazione feature (X) e target (y)
X = df.drop(columns=['compatibility_score'])
y = df['compatibility_score']

# Divisione in Train e Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Salvataggio dati di test per la valutazione futura
joblib.dump(X_test, X_test_path)
joblib.dump(y_test, y_test_path)

# Creazione della Pipeline (Preprocessing + Modello XGBoost)
print("Configurazione della Pipeline in corso...")
pipeline = Pipeline(steps=[
    ('preprocessor', build_xgb_preprocessor()),
    ('model', get_xgb_model())
])

# Addestramento del modello (la pipeline fa il preprocessing in automatico!)
print("Addestramento XGBoost in corso...")
pipeline.fit(X_train, y_train)

# Salva l'intera pipeline (preprocessing + modello)
joblib.dump(pipeline, MODEL_NAME)
print(f"Modello e Preprocessing salvati con successo in: {MODEL_NAME}")