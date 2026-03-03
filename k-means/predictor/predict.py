import pandas as pd
import joblib
import os
import sys

# 1. Capiamo dove si trova questo script
cartella_script = os.path.dirname(os.path.abspath(__file__))

# 2. Facciamo UN PASSO INDIETRO per trovare la radice del progetto
cartella_principale = os.path.dirname(cartella_script)

# 3. Aggiungiamo la radice al radar di Python (serve a joblib per trovare 'utils.preprocessing' di nascosto)
if cartella_principale not in sys.path:
    sys.path.append(cartella_principale)

def assegna_cluster(nuovo_utente_df, model_filename='kmeans_model.pkl'):
    """
    Carica il modello salvato e prevede a quale cluster appartiene il nuovo utente.
    """
    # --- FIX: Andiamo a pescare il modello dentro 'resources' ---
    model_path = os.path.join(cartella_principale, 'resources', model_filename)
    
    # Carichiamo la pipeline
    pipeline = joblib.load(model_path)
    
    # Eseguiamo la classificazione
    cluster_assegnato = pipeline.predict(nuovo_utente_df)
    return cluster_assegnato[0]

if __name__ == "__main__":
    # Esempio di un nuovo utente "inviato" dall'app/sito
    nuovi_dati_utente = pd.DataFrame([{
        'age': '18-30',
        'education': 4,
        'career_field': 'Tech',
        'career_ambition': 0.85,
        'openness': 0.80,
        'extraversion': 0.70,
        'agreeableness': 0.60,
        'conscientiousness': 0.75,
        'chronotype': 0.40,
        'spontaneity': 0.50,
        'communication_style': 'Shared Experiences',
        'emotional_intelligence': 0.65
    }])
    
    print("Classificazione del nuovo utente in corso...")
    
    # Richiamiamo la funzione
    risultato = assegna_cluster(nuovi_dati_utente)
    
    print(f"-> Il nuovo utente è stato assegnato al CLUSTER {risultato}")