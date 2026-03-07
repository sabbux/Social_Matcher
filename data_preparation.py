import pandas as pd
import os 

# 1. Trova la cartella corrente
project_root = os.path.dirname(os.path.abspath(__file__))

# 2. Definisci i percorsi basati sulla nuova struttura centralizzata "data/"
percorso_dataset = os.path.join(project_root, 'data', 'raw', 'cupid_dataset.csv')
cartella_output = os.path.join(project_root, 'data', 'processed')
percorso_output = os.path.join(cartella_output, 'social_matcher.csv')

# Assicurati che la cartella 'data/processed' esista (la crea se manca)
os.makedirs(cartella_output, exist_ok=True)

# 3. Carica il dataset
print(f"Caricamento dati grezzi da: {percorso_dataset}")
df = pd.read_csv(percorso_dataset)

# Rimozione delle colonne non necessarie
colonne_da_rimuovere = [
    'pair_id', 
    'a_location', 
    'b_location', 
    'compatible', 
    'relationship_longevity_months'
]
df = df.drop(columns=colonne_da_rimuovere)

# Trasformazione dell'età in range
bins = [17, 30, 43, 55] 
labels = ['18-30', '31-43', '44-55']

# pd.cut trasforma i numeri continui nelle categorie (labels) che abbiamo definito
df['a_age'] = pd.cut(df['a_age'], bins=bins, labels=labels)
df['b_age'] = pd.cut(df['b_age'], bins=bins, labels=labels)

# Rinomina le colonne, cambiando i nomi delle colonne per la persona A e la persona B
df = df.rename(columns={
    'a_love_language': 'a_communication_style',
    'b_love_language': 'b_communication_style',
    'a_emotional_expressiveness': 'a_emotional_intelligence',
    'b_emotional_expressiveness': 'b_emotional_intelligence'
})


# Crea un dizionario per tradurre i "Linguaggi dell'Amore" in "Stili di Comunicazione"
mappa_stili = {
    'Words of Affirmation': 'Verbal Support',     # Bisogno di feedback positivi/riconoscimento
    'Quality Time': 'Shared Experiences',          # Lavoro a stretto contatto / Pair programming
    'Acts of Service': 'Practical Reliability',        # Aiuto pratico nei task
    'Receiving Gifts': 'Thoughtful Gestures',       # Motivato da bonus o benefit tangibili
    'Physical Touch': 'Physical Warmth'           # Preferenza per il lavoro in ufficio (no smart working)
}

# Applica la mappa ai valori delle nuove colonne
df['a_communication_style'] = df['a_communication_style'].map(mappa_stili)
df['b_communication_style'] = df['b_communication_style'].map(mappa_stili)


# Gestione dei Valori Nulli (Imputazione)
# Calcolo dei parametri
moda_a_age = df['a_age'].mode()[0]
media_a_emot = round(df['a_emotional_intelligence'].mean(), 2) # Arrotondato a 2 decimali
moda_b_love = df['b_communication_style'].mode()[0]

# Stampa dei valori calcolati per l'imputazione
print("--- Valori calcolati per l'imputazione ---")
print(f"Età (Moda): {moda_a_age}")
print(f"Intelligenza Emotiva (Media arrotondata): {media_a_emot}")
print(f"Stile di Comunicazione (Moda): {moda_b_love}")
print("------------------------------------------")
    
    # Applicazione dell'imputazione
df['a_age'] = df['a_age'].fillna(moda_a_age)
df['a_emotional_intelligence'] = df['a_emotional_intelligence'].fillna(media_a_emot)
df['b_communication_style'] = df['b_communication_style'].fillna(moda_b_love)
    
    # Riempiamo anche eventuali altri nulli generici rimasti 
    # (mediana per i numeri, moda per i testi)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())
        
for col in df.select_dtypes(include=['object', 'category']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Salva il DataFrame aggiornato in un nuovo file CSV
df.to_csv(percorso_output, index=False)
print(f"Dataset pulito e salvato con successo in:\n{percorso_output}")