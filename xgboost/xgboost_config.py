# model_config.py
from xgboost import XGBRegressor

def get_xgb_model():

    """Modello xgboost bilanciato per raggiungere punteggi realistici e ampi (es. 40-80), senza rischiare l'overfitting."""
    return XGBRegressor(
        # --- PARAMETRI DI BASE ---
        n_estimators=250,       # Numero di alberi bilanciato per garantire un ottimo apprendimento senza allungare troppo i tempi.
        learning_rate=0.05,     # Passo di apprendimento stabile, ideale per migliorare gradualmente senza sbalzi.
        
        # --- MOTORE (Profondità e gestione dei casi rari) ---
        max_depth=6,            # Permette incroci complessi (es. Età + Lavoro + Ambizione + Estroversione) per scovare le dinamiche che portano a profili da 80 o da 40.
        min_child_weight=2,     # Il compromesso perfetto: permette di esplorare i picchi, ma esige che ci siano almeno 2 coppie simili nel dataset per confermare la regola (evita le anomalie).
        gamma=0.0,              # Nessun blocco alla ramificazione, lasciando libertà all'albero di crescere e trovare dettagli.
        
        # --- FIDUCIA NEI DATI (Certezze e robustezza) ---
        subsample=0.85,         # L'albero usa l'85% dei dati. Avendo una visione molto ampia e sicura, oserà restituire punteggi più decisi, pur mantenendo un 15% di casualità.
        colsample_bytree=0.85,  # Valuta l'85% delle caratteristiche per ogni decisione, evitando di fissarsi solo su una singola variabile dominante.
        
        # --- I FRENI MATEMATICI (Bilanciati per non schiacciare le previsioni verso la media) ---
        reg_alpha=0.1,          # Penalità L1 leggera: sufficiente per spegnere il "rumore" irrilevante senza soffocare il modello.
        reg_lambda=1.0,         # Valore standard (neutro) di XGBoost: non "taglia" artificialmente i punteggi estremi, permettendo alla previsione di spingersi liberamente verso gli alti e i bassi.
        
        # --- IMPOSTAZIONI DI SISTEMA ---
        n_jobs=-1,              # Usa tutti i processori disponibili per velocizzare l'elaborazione.
        random_state=42,        # Rende i risultati identici e riproducibili ad ogni avvio.
        eval_metric='rmse',     # Misuratore di errore ideale per la regressione lineare.
        enable_categorical=True # Permette a XGBoost di leggere nativamente i testi (es. "Tech", "Science").
    )