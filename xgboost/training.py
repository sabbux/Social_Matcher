import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost_config import get_xgb_model
from preprocessing import build_preprocessor 


base_dir = os.path.dirname(os.path.abspath(__file__))

resource_dir = os.path.join(base_dir, "resource")

os.makedirs(resource_dir, exist_ok=True)

# Percorsi completi per input e output
FILENAME = os.path.join(resource_dir, 'social_matcher.csv')
MODEL_NAME = os.path.join(resource_dir, 'social_matcher_model.pkl')
X_test_path = os.path.join(resource_dir, "X_test.pkl")
y_test_path = os.path.join(resource_dir, "y_test.pkl")


# Caricamento dataset
print("Caricamento dataset...")
df = pd.read_csv(FILENAME)

# Separazione feature (X) e target (y)
X = df.drop(columns=['compatibility_score'])
y = df['compatibility_score']

# Divisione in Train e Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Salvataggio dati di test per la valutazione futura
joblib.dump(X_test, X_test_path)
joblib.dump(y_test, y_test_path)

# 3. Creazione della Pipeline (Preprocessing + Modello XGBoost)
print("Configurazione della Pipeline in corso...")
pipeline = Pipeline(steps=[
    ('preprocessor', build_preprocessor()),
    ('model', get_xgb_model())
])

# 4. Addestramento del modello (la pipeline fa il preprocessing in automatico!)
print("Addestramento XGBoost in corso...")
pipeline.fit(X_train, y_train)

# Salva l'intera pipeline (preprocessing + modello)
joblib.dump(pipeline, MODEL_NAME)
print(f"Modello e Preprocessing salvati con successo in: {MODEL_NAME}")