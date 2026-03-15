import pandas as pd
import os

# 1. Trova il percorso assoluto della cartella dove si trova questo script (.py)
cartella_script = os.path.dirname(os.path.abspath(__file__))

# 2. PRIMO PASSO INDIETRO: usciamo dalla cartella dello script per arrivare a 'k-means'
cartella_kmeans = os.path.dirname(cartella_script)

# 3. SECONDO PASSO INDIETRO: usciamo da 'k-means' per arrivare alla ROOT principale
cartella_root = os.path.dirname(cartella_kmeans)

# 4. Assegniamo i percorsi corretti
input_csv = os.path.join(cartella_root, 'data', 'raw', 'cupid_dataset.csv')
output_csv = os.path.join(cartella_root, 'data', 'processed', 'adapted_dataset.csv')

# 5. Leggiamo il file csv
df = pd.read_csv(input_csv)

# 6. Estrai il blocco della "Persona A" e togli il prefisso 'a_'
a_cols = [col for col in df.columns if col.startswith('a_')]
df_a = df[a_cols].copy()
df_a.columns = [col.replace('a_', '') for col in df_a.columns]

# 7. Estrai il blocco della "Persona B" e togli il prefisso 'b_'
b_cols = [col for col in df.columns if col.startswith('b_')]
df_b = df[b_cols].copy()
df_b.columns = [col.replace('b_', '') for col in df_b.columns]

# 8. Unisci i due blocchi uno sotto l'altro (Concatenazione verticale)
df_utenti_completo = pd.concat([df_a, df_b], axis=0, ignore_index=True)

# 9. Rimuovi i duplicati
df_utenti_unici = df_utenti_completo.drop_duplicates().reset_index(drop=True)

# 10. Salviamo il nuovo dataset pulito, senza colonna target e con tutti gli utenti
df_utenti_unici.to_csv(output_csv, index=False)

# Stampiamo un recap per controllare che i conti tornino
print(f"Utenti 'A' estratti: {len(df_a)}")
print(f"Utenti 'B' estratti: {len(df_b)}")
print(f"Totale utenti dopo l'unione: {len(df_utenti_completo)}")
print(f"Totale utenti UNICI pronti per il K-Means: {len(df_utenti_unici)}")