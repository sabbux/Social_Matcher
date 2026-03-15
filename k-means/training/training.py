import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import os
import sys

# 1. Capiamo dove si trova questo script
cartella_script = os.path.dirname(os.path.abspath(__file__))

# 2. Facciamo UN PASSO INDIETRO (saliamo di un livello) per trovare la radice del progetto
cartella_principale = os.path.dirname(cartella_script)

# 3. Facciamo UN ALTRO PASSO INDIETRO per trovare la root principale
cartella_root = os.path.dirname(cartella_principale)

# 4. Aggiungiamo la radice al radar di Python per far funzionare l'import
if cartella_principale not in sys.path:
    sys.path.append(cartella_principale)

# --- IMPORT CORRETTO DALLA CARTELLA UTILS ---
from utils.preprocessing import build_preprocessor
# --------------------------------------------

def train_and_save_model(data_filename, model_filename, n_clusters=4):
    
    # --- PERCORSI AGGIORNATI ---
    # Ora il CSV è in data/processed, il modello rimane in resources
    data_path = os.path.join(cartella_root, 'data', 'processed', data_filename)
    model_save_path = os.path.join(cartella_principale, 'resources', model_filename)
    # ---------------------------

    print(f"1. Caricamento del dataset pulito da: {data_path}...")
    df_users = pd.read_csv(data_path)
    
    print("2. Creazione della Pipeline (Preprocessing + K-Means)...")
    pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('clusterer', KMeans(n_clusters=n_clusters, random_state=42, n_init=10))
    ])
    
    print("3. Addestramento in corso...")
    pipeline.fit(df_users)
    
    print(f"4. Salvataggio del modello in:\n{model_save_path} ...")
    joblib.dump(pipeline, model_save_path)
    print("Finito! Il modello K-Means è addestrato e salvato in resources.")

if __name__ == "__main__":
    train_and_save_model('adapted_dataset.csv', 'kmeans_model.pkl')