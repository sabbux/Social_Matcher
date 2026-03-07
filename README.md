# Social Matcher - Calcolatore di Affinità IA 🔮

**Social Matcher** è un progetto di Machine Learning progettato per analizzare profili utente e calcolare le affinità tra le persone, simulando un algoritmo di matching sociale intelligente.

Il progetto esplora e confronta due approcci di Machine Learning:
- **Algoritmi Supervisionati (XGBoost):** Per prevedere un punteggio di compatibilità diretto (da 0 a 100) tra due profili sulla base di dati storici di successo delle coppie.
- **Algoritmi Non Supervisionati (K-Means):** Per raggruppare (clusterizzare) gli utenti in base a caratteristiche simili (personalità, interessi, aspirazioni lavorative, ecc.) senza un target definito, individuando somiglianze latenti e applicando il principio sociologico dell'omofilia.
---

## 📁 Struttura del Progetto

Il progetto è suddiviso in due moduli principali, ciascuno dedicato a un modello specifico:

### 1. Modulo XGBoost (`/xgboost`)
Questo modulo addestra un modello predittivo per calcolare in tempo reale la percentuale di affinità tra due profili ("Persona A" e "Persona B").

- `predictor/`: Modulo dedicato all'interfaccia utente.
  - `app.py`: Un'applicazione intuitiva sviluppata in **Tkinter** che permette agli utenti di inserire manualmente i parametri di due profili (età, livello di istruzione, carriera, stili comunicativi e tratti di personalità tra cui apertura, estroversione, cordialità, coscienziosità) e ottenere la stima percentuale di affinità calcolata dal modello addestrato.
- `training/`: Modulo dedicato all'addestramento.
  - `training.py`: Lo script principale per addestrare il modello. Carica i dati, li divide in Train e Test (80/20) ed esegue l'addestramento costruendo una pipeline completa che integra preprocessing e modello XGBoost. Salva l'intera pipeline in `social_matcher_model.pkl`. Vengono inoltre generati i file `X_test.pkl` e `y_test.pkl`, che contengono rispettivamente i dati di test e i target di addestramento.
- `model_config/` & `utils/`: Componenti di configurazione parametri XGBoost e trasformazione dati (`xgboost_config.py` e `preprocessing.py`).
- `results_evaluation/`:
  - `benchmark.ipynb`: Notebook Jupyter che calcola le metriche di performance del modello XGBoost.

### 2. Modulo K-Means (`/k-means`)
Questo modulo implementa l'approccio non supervisionato per estrarre 4 macro-profili psicologici (centroidi) e suggerire affinità basate sulla compatibilità caratteriale.

- `dataset_adapter/`: Contiene lo script di destrutturazione che separa le "coppie" del dataset originale, trasformandole in un database di individui singoli e indipendenti (`adapted_dataset.csv`).
- `utils/`: Contiene il file `preprocessing.py` dedicato alla standardizzazione semantica, al Feature Scaling e al One-Hot Encoding per preparare matematicamente i dati all'algoritmo spaziale.
- `training/`: Contiene il file di training per addestrare il modulo non supervisionato e renderlo pronto all'uso.
- `results_evaluation/`: Cartella principale dedicata alla validazione e interpretazione dei risultati. Al suo interno comprende:
  - `elbow_point.ipynb` e `silhouette_score.ipynb`: File e analisi per la determinazione oggettiva del numero ottimale di cluster, tramite la minimizzazione dell'Inerzia (Elbow Method) e la validazione della densità dei gruppi (Silhouette Score calcolato su un campione di 50k istanze).
  - `cluster_profiling.ipynb`: Analisi approfondita dei centroidi estratti. Traduce le coordinate matematiche in profili psicologici reali analizzando le medie dei tratti continui e le mode delle variabili categoriali. Questo step si occupa inoltre di generare il database finale (`clustered_dataset.csv`) assegnando le etichette di cluster ai singoli utenti.
- `resources/`: Cartella di destinazione per gli artefatti generati dal training, contenente il modello serializzato (`kmeans_model.pkl`) e i database `adapted_dataset.csv` `clustered_dataset.csv`.
- `matcher/`: Contiene `similarity_matching.py`, il motore logico che applica il principio di omofilia per interrogare il database ed estrarre i profili più affini.
- `predictor/`: Modulo dedicato all'inferenza e all'interfaccia utente. Contiene:
  - `predict.py`: Assegna i nuovi utenti al cluster corretto calcolando la distanza matematica dai centroidi.
  - `app.py`: Un'applicazione web interattiva sviluppata in **Streamlit** che offre due funzionalità:
    - **Confronto di Gruppo:** Permette di inserire fino a 5 persone e verificare istantaneamente se appartengono allo stesso cluster psicologico.
    - **Trova Profili Affini:** Ricerca dinamica nel database per estrarre i 3 profili perfetti per l'utente.
   
### I Dataset
- `cupid_algorithm_dataset.csv`: Il dataset principale adoperato nel progetto. Presenta una raccolta di caratteristiche delle coppie di utenti e un "compatibility_score" risultante.
- `social_matcher.csv`: Dataset risultante da un processo di pulizia del dataset originale, utilizzato per l'addestramento del modello XGBoost.
- `adapted_dataset.csv` e `clustered_dataset.csv`: Dataset derivati per l'addestramento del K-Means e per il motore di ricerca dell'applicazione Streamlit.

---

## 🚀 Installazione e Requisiti

Assicurati di avere **Python 3.11.9** installato sul sistema. Per avviare i modelli, l'interfaccia grafica Tkinter e la web app Streamlit, è sufficiente installare le dipendenze elencate nel file `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🛠️ Come usare il progetto?

### Passo 1: Addestramento del Modello (XGBoost)
Prima di poter utilizzare l'interfaccia utente, è necessario creare un file di modello funzionante (`.pkl`). Spostati nella cartella `xgboost/training` e avvia lo script di training:
```bash
cd xgboost/training
python training.py
```
*Verrà costruita una directory `resources` all'interno della quale saranno salvati il dataset d'addestramento diviso ed il modello addestrato `social_matcher_model.pkl`.*

### Passo 2: Calcolo delle Affinità (GUI)
Una volta ottenuto il file `.pkl`, sarà possibile valutare interattivamente nuovi match usando la dashboard. Spostati nella cartella `xgboost/predictor` e lancia l'app:
```bash
cd xgboost/predictor
python app.py
```
Inserisci i dati per il Profilo A e per il Profilo B, compila ogni campo e clicca su **"Calcola Affinità"** per ottenere la previsione calcolata dal modello (un numero da 0 a 100%).

### Passo 3: Preparazione Dati e Addestramento (K-Means)
Per testare il modulo non supervisionato e far funzionare l'applicazione web, è necessario preparare i dati, addestrare il modello e popolare il database. Segui questi sotto-passaggi in ordine:

1. **Adattamento del Dataset:**
   Spostati nella cartella `k-means` ed esegui lo script che destruttura il dataset originale (che conteneva coppie) per creare un bacino di utenti singoli:
   ```bash
   cd k-means/dataset_adapter
   python dataset_adapter.py
   ```
   *(Verrà generato il file `adapted_dataset.csv` pronto per l'elaborazione).*

2. **Addestramento del Modello:**
   Esegui gli script presenti nella cartella `training/`. Questa fase si occupa specificamente di addestrare l'algoritmo K-Means, applicare il preprocessing e assegnare i cluster. L'addestramento popolerà automaticamente la cartella `resources/` con il file fondamentale per l'app:
   - `kmeans_model.pkl` (il modello serializzato).

3. **Valutazione e Profilazione (Analisi Indipendente):**
   I file per la validazione matematica e psicologica del modello si trovano nella cartella `results_evaluation/`. Puoi eseguirli separatamente per:
   - Verificare le metriche di densità e inerzia (tramite gli script `elbow_point.ipynb` e `silhouette_score.ipynb`).
   - Analizzare nel dettaglio i tratti psicologici dei centroidi ( `cluster_profiling.ipynb`). Quest'ultimo genererà anche `clustered_dataset.csv` che verrà usato dall'applicazione.

### Passo 4: Avvio della Web App (Streamlit)
Una volta che l'addestramento ha generato il modello e hai creato il database storicizzato, sei pronto per lanciare la piattaforma interattiva di Social Matching.

1. Posizionati nella cartella del predittore:
   ```bash
   cd predictor
   ```

2. Avvia l'interfaccia web tramite Streamlit:
   ```bash
   streamlit run app.py
   ```

3. **Esplora la Piattaforma:** Si aprirà automaticamente una finestra nel tuo browser. Usa i tab in alto per navigare tra le funzionalità:
   - **👥 Confronto di Gruppo:** Compila i dati per un gruppo ristretto (fino a 5 amici o colleghi) per verificare istantaneamente se condividono la stessa struttura psicologica.
   - **🔍 Trova Profili Affini:** Compila solo il tuo profilo personale. Il motore logico calcolerà la tua distanza dai centroidi e pescherà dal database i 3 partner ideali basati sul principio di omofilia!

## 👥 Team

Il progetto è stato sviluppato da:
* [Francesco Sabetta](https://github.com/sabbux)
* [Gloria Scarallo](https://github.com/gloriascarallo)

---
Questa repository contiene la completa implementazione del progetto accademico del corso di `Machine Learning` di Unisa.   