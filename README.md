# Social Matcher - Calcolatore di Affinità IA 🔮

**Social Matcher** è un progetto di Machine Learning progettato per analizzare profili utente e calcolare le affinità tra le persone, simulando un algoritmo di matching sociale intelligente.

Il progetto esplora e confronta due approcci di Machine Learning:
- **Algoritmi Supervisionati (XGBoost):** Per prevedere un punteggio di compatibilità diretto (da 0 a 100) tra due profili sulla base di dati storici di successo delle coppie.
- **Algoritmi Non Supervisionati (K-Means):** Per raggruppare (clusterizzare) gli utenti in base a caratteristiche simili (personalità, interessi, aspirazioni lavorative, ecc.) senza un target definito, individuando somiglianze latenti.

---

## 📁 Struttura del Progetto

Il progetto è suddiviso in due moduli principali, ciascuno dedicato a un modello specifico:

### 1. Modulo XGBoost (`/xgboost`)
Questo modulo addestra un modello predittivo per calcolare in tempo reale la percentuale di affinità tra due profili ("Persona A" e "Persona B").

- `app.py`: Un'applicazione intuitiva sviluppata in **Tkinter** che permette agli utenti di inserire manualmente i parametri di due profili (età, livello di istruzione, carriera, stili comunicativi e tratti di personalità tra cui apertura, estroversione, cordialità, coscienziosità) e ottenere la stima percentuale di affinità calcolata dal modello addestrato.
- `training.py`: Lo script principale per addestrare il modello. Carica i dati, li divide in Train e Test (80/20) ed esegue l'addestramento costruendo una pipeline completa che integra preprocessing e modello XGBoost. Salva l'intera pipeline in `social_matcher_model.pkl`. Vengono inoltre generati i file `X_test.pkl` e `y_test.pkl`, che contengono rispettivamente i dati di test e i target di addestramento.
- `xgboost_config.py` & `preprocessing.py`: Componenti che gestiscono la configurazione dei parametri di XGBoost e le tecniche per la trasformazione dei dati (come Scaling e One-Hot Encoding per le variabili categoriche).
- `benchmark.ipynb`: Notebook Jupyter che calcola le metriche di performance del modello XGBoost.

### 2. Modulo K-Means (`/k-means`)
Questo modulo implementa l'approccio non supervisionato applicando il clustering per trovare gruppi di utenti affini.

- `dataset_adapter/dataset_adapter.py`: Script che si occupa di separare i profili della "Persona A" e "Persona B" presenti nel dataset originale e li unisce eliminando i duplicati, creando così un set di utenti unici pronti per la clusterizzazione.
- Supporta diverse altre cartelle (`matcher`, `predictor`, `results_evaluation`, `training`, ecc.) dedicate a compiti come l'allenamento del modello K-Means e la valutazione dei cluster generati.

### I Dataset
- `cupid_algorithm_dataset.csv`: Il dataset principale adoperato nel progetto. Presenta una raccolta di caratteristiche delle copie degli utenti e un "compatibility_score" risultante.
- `social_matcher.csv`: Dataset risultante da un processo di pulizia e preprocessing del dataset originale, utilizzato per l'addestramento del modello XGBoost.

---

## 🚀 Installazione e Requisiti

Assicurati di avere **Python** installato sul sistema. Per avviare i modelli e l'interfaccia grafica, occorre installare diverse librerie standard impiegate per il Data Science:

```bash
pip install pandas scikit-learn xgboost numpy joblib
```

> **Nota per l'applicazione grafica (`app.py`):** L'interfaccia utente è scritta con la libreria integrata di Python `tkinter`, che non dovrebbe richiedere l'installazione tramite pip ed è generalmente già inclusa con Python standard.

---

## 🛠️ Come usare il progetto?

### Passo 1: Addestramento del Modello (XGBoost)
Prima di poter utilizzare l'interfaccia utente, è necessario creare un file di modello funzionante (`.pkl`). Spostati nella cartella `xgboost` e avvia lo script di training:
```bash
cd xgboost
python training.py
```
*Verrà costruita una directory `resources` all'interno della quale saranno salvati il dataset d'addestramento diviso ed il modello addestrato `social_matcher_model.pkl`.*

### Passo 2: Calcolo delle Affinità (GUI)
Una volta ottenuto il file `.pkl`, sarà possibile valutare interattivamente nuovi match usando la dashboard. Nello stesso percorso, lancia l'app:
```bash
python app.py
```
Inserisci i dati per il Profilo A e per il Profilo B, compila ogni campo e clicca su **"Calcola Affinità"** per ottenere la previsione calcolata dal modello (un numero da 0 a 100%).

### Passo 3: Clusterizzazione K-Means
Se vuoi esplorare i cluster degli utenti singoli, spingiti nel folder `k-means`. Puoi ad esempio iniziare preparando la versione adatatta ('unfolded') del dataset:
```bash
cd ../k-means/dataset_adapter
python dataset_adapter.py
```