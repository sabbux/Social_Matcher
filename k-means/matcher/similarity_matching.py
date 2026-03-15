import pandas as pd
import os
import sys

# 1. Capiamo dove si trova questo script
cartella_script = os.path.dirname(os.path.abspath(__file__))

# 2. Facciamo UN PASSO INDIETRO per trovare la radice del progetto
cartella_principale = os.path.dirname(cartella_script)

# 3. Facciamo UN ALTRO PASSO INDIETRO per trovare la root principale
cartella_root = os.path.dirname(cartella_principale)

# 4. Aggiungiamo la radice al radar di Python
if cartella_principale not in sys.path:
    sys.path.append(cartella_principale)

def trova_match_per_omofilia(cluster_target, numero_match_desiderati=3):
    """
    Simula una query al database: trova N profili che appartengono 
    esattamente allo stesso cluster dell'utente attivo.
    """
    # Andiamo a pescare il "database" aggiornato con i cluster
    db_path = os.path.join(cartella_root, 'data', 'processed', 'clustered_dataset.csv')
    
    if not os.path.exists(db_path):
        print("Errore: Il file 'dataset_con_cluster.csv' non esiste in resources.")
        print("Assicurati di aver scommentato il salvataggio nel Jupyter Notebook!")
        return None
        
    df_database = pd.read_csv(db_path)
    
    # --- LA VERA LOGICA DI MATCHING (OMOFILIA) ---
    # Filtriamo l'intero dataset tenendo SOLO le persone con lo stesso cluster
    utenti_compatibili = df_database[df_database['cluster'] == cluster_target]
    
    # Controlliamo quanti utenti compatibili abbiamo trovato realmente
    totale_disponibili = len(utenti_compatibili)
    
    if totale_disponibili == 0:
        print("Nessun utente compatibile trovato in questo cluster.")
        return None
        
    # Se chiediamo più match di quelli che esistono, ci accontentiamo di quelli che ci sono
    n_estratti = min(numero_match_desiderati, totale_disponibili)
    
    # Estraiamo casualmente N profili dal bacino dei compatibili
    # random_state rimosso per avere match diversi a ogni avvio!
    match_selezionati = utenti_compatibili.sample(n=n_estratti)
    
    return match_selezionati

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" MOTORE DI MATCHING - IPOTESI DELL'OMOFILIA ")
    print("="*50)
    
    # Immaginiamo che il nostro file "predict.py" abbia appena stabilito 
    # che l'utente che ha aperto l'app (Utente Attivo) appartiene al Cluster 1
    mio_cluster = 1
    
    print(f"\nL'utente attivo appartiene al CLUSTER {mio_cluster}.")
    print("Ricerca di profili altamente compatibili in corso...\n")
    
    # Eseguiamo la funzione per trovare 3 potenziali partner/colleghi
    risultati = trova_match_per_omofilia(cluster_target=mio_cluster, numero_match_desiderati=3)
    
    if risultati is not None:
        print(f"-> Trovati {len(risultati)} match perfetti!")
        print("-" * 50)
        
        # Mostriamo solo alcune colonne chiave per verificare visivamente l'affinità
        colonne_da_mostrare = ['age', 'career_field', 'communication_style', 'education']
        
        # Stampiamo in modo leggibile iterando sui risultati
        for indice, row in risultati.iterrows():
            print(f"Profilo ID: {indice}")
            print(f"  - Settore: {row['career_field']}")
            print(f"  - Stile Com.: {row['communication_style']}")
            print(f"  - Età: {row['age']}")
            print("-" * 50)