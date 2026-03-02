import pandas as pd
import os 


cartella_script = os.path.dirname(os.path.abspath(__file__))

cartella_principale = os.path.dirname(cartella_script)

percorso_dataset = os.path.join(cartella_principale, 'cupid_algorithm_dataset.csv')

# Carica il dataset
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


# Salva il DataFrame aggiornato in un nuovo file CSV
cartella_resource = os.path.join(cartella_script, 'resource')

if not os.path.exists(cartella_resource):
    os.makedirs(cartella_resource)

percorso_output = os.path.join(cartella_resource, 'social_matcher.csv')

df.to_csv(percorso_output, index=False)