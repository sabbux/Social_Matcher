# model_config.py
from xgboost import XGBRegressor

def get_xgb_model():

    """Restituisce il modello XGBoost non addestrato con gli iperparametri configurati."""

    return XGBRegressor(
        
    n_estimators=200,       # Numero di alberi decisionali da costruire (né troppi, né pochi).
    learning_rate=0.05,     # Velocità di apprendimento: basso = impara a piccoli passi, ma meglio.
    max_depth=4,            # Profondità degli alberi: ridotta per non imparare a memoria i dati.
    min_child_weight=5,     # Peso minimo per creare un ramo: alzato per ignorare casi troppo specifici.
    gamma=0.1,              # Crea un nuovo nodo solo se riduce l'errore almeno di questo valore.
    subsample=0.7,          # Usa il 70% dei dati (scelti a caso) per albero, aumentando la robustezza.
    colsample_bytree=0.7,   # Usa il 70% delle colonne per albero, per non dipendere da singole feature.
    reg_alpha=0.1,          # Regolarizzazione L1: aiuta a ignorare del tutto le feature poco utili.
    reg_lambda=1.0,         # Regolarizzazione L2: evita che i "pesi" matematici diventino troppo estremi.
    n_jobs=-1,              # Usa tutti i core del computer per velocizzare l'addestramento.
    random_state=42,        # Fissa la casualità per avere risultati identici se si ripete il codice.
    eval_metric='rmse'      # Metrica di valutazione (Root Mean Squared Error) adatta alla regressione.
)