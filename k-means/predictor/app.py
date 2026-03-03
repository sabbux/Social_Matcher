import streamlit as st
import pandas as pd
import os
import sys

# 1. Capiamo dove si trova esattamente questo file app.py
cartella_script = os.path.dirname(os.path.abspath(__file__))
if cartella_script not in sys.path:
    sys.path.append(cartella_script)

# 2. Facciamo UN PASSO INDIETRO per arrivare alla cartella 'k-means'
cartella_kmeans = os.path.dirname(cartella_script)
if cartella_kmeans not in sys.path:
    sys.path.append(cartella_kmeans)

# 3. Importiamo la funzione dal file predict
from predict import assegna_cluster

# --- CONFIGURAZIONE DELLA PAGINA ---
st.set_page_config(page_title="Social Matcher AI", page_icon="🧬", layout="wide")

st.title("🧬 Social Matcher AI")
st.markdown("Scopri le affinità tra le persone usando l'Intelligenza Artificiale (K-Means Clustering).")

# --- SEZIONE 1: SCELTA DEL NUMERO DI PERSONE ---
n_persone = st.number_input("Quante persone vuoi confrontare?", min_value=2, max_value=5, value=2)

st.divider()

# --- DIZIONARI E LISTE OPZIONI ---
eta_options = ['18-30', '31-43', '44-55', '56+']
lavoro_options = [
    'Tech', 'Healthcare', 'Finance', 'Education', 'Creative Arts', 
    'Law', 'Engineering', 'Marketing', 'Science', 'Entrepreneurship'
]
stile_options = [
    'Physical Warmth', 'Thoughtful Gestures', 'Practical Reliability', 
    'Shared Experiences', 'Verbal Support'
]

# Dizionario per l'istruzione (Testo -> Numero per il modello)
mappa_istruzione = {
    "Diploma Superiore (High School)": 1,
    "Diploma Universitario (Associate's)": 2,
    "Laurea Triennale (Bachelor's)": 3,
    "Laurea Magistrale (Master's)": 4,
    "Dottorato di Ricerca (Doctorate)": 5
}
opzioni_istruzione = list(mappa_istruzione.keys())

# Dizionario dei testi personalizzati per i Cluster
testi_cluster = {
    0: {
        "titolo": "Gli Strutturati / I Tradizionalisti",
        "descrizione": "Questo match è basato su una solida organizzazione. Entrambi amate la routine, l'affidabilità e la pianificazione. Insieme formate una coppia (o un team) estremamente stabile e concreta, dove non c'è spazio per il caos!"
    },
    1: {
        "titolo": "I Leader Dinamici",
        "descrizione": "Scintille garantite! Siete entrambi estremamente socievoli, emotivamente intelligenti e con una forte propensione alla leadership. Questo match è perfetto per conquistare il mondo insieme, con grande energia e comunicazione eccellente."
    },
    2: {
        "titolo": "Gli Spiriti Liberi / I Rilassati",
        "descrizione": "Regole? Quali regole? Avete un'altissima affinità basata sulla spontaneità e sulla flessibilità. Questo è un match all'insegna dell'improvvisazione, dell'avventura e della creatività senza confini, lontani dallo stress della perfezione."
    },
    3: {
        "titolo": "Gli Intellettuali Introversi",
        "descrizione": "Una profonda connessione mentale. Siete entrambi altamente istruiti, ambiziosi ma riservati. Questo match prospererà su conversazioni profonde, un forte rispetto dei reciproci spazi e il supporto verso grandi obiettivi personali e professionali."
    }
}

dati_inseriti = [] 

# --- SEZIONE 2: FORM DINAMICO PER OGNI PERSONA ---
tabs = st.tabs([f"Persona {i+1}" for i in range(n_persone)])

for i, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Profilo della Persona {i+1}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Dati Demografici e Lavoro**")
            eta = st.selectbox("Fascia d'età", eta_options, key=f"eta_{i}")
            # Menu a tendina per l'istruzione (di default impostato sulla Laurea Triennale, indice 2)
            istruzione_scelta = st.selectbox("Livello Istruzione", opzioni_istruzione, index=2, key=f"edu_label_{i}")
            # Convertiamo il testo scelto nel numero corrispondente
            istruzione_num = mappa_istruzione[istruzione_scelta]
            
            lavoro = st.selectbox("Settore Lavorativo", lavoro_options, key=f"lav_{i}")
            stile = st.selectbox("Stile Comunicativo", stile_options, key=f"stile_{i}")
            
        with col2:
            st.markdown("**Tratti della Personalità**")
            apertura = st.slider("Apertura Mentale (0-1)", 0.0, 1.0, 0.5, key=f"ope_{i}")
            estroversione = st.slider("Estroversione (0-1)", 0.0, 1.0, 0.5, key=f"ext_{i}")
            amicalita = st.slider("Amicalità / Empatia (0-1)", 0.0, 1.0, 0.5, key=f"agr_{i}")
            coscienziosita = st.slider("Coscienziosità / Ordine (0-1)", 0.0, 1.0, 0.5, key=f"con_{i}")
            
        with col3:
            st.markdown("**Stile di Vita e Attitudini**")
            ambizione = st.slider("Ambizione Lavorativa (0-1)", 0.0, 1.0, 0.5, key=f"amb_{i}")
            cronotipo = st.slider("Cronotipo (0=Mattiniero, 1=Notturna)", 0.0, 1.0, 0.5, key=f"chr_{i}")
            spontaneita = st.slider("Spontaneità (0-1)", 0.0, 1.0, 0.5, key=f"spo_{i}")
            intelligenza_emotiva = st.slider("Intelligenza Emotiva (0-1)", 0.0, 1.0, 0.5, key=f"emo_{i}")
        
        # Salviamo il profilo usando il numero (istruzione_num) per l'istruzione
        profilo_df = pd.DataFrame([{
            'age': eta, 
            'education': istruzione_num, 
            'career_field': lavoro, 
            'career_ambition': ambizione, 
            'openness': apertura, 
            'extraversion': estroversione, 
            'agreeableness': amicalita, 
            'conscientiousness': coscienziosita, 
            'chronotype': cronotipo, 
            'spontaneity': spontaneita, 
            'communication_style': stile, 
            'emotional_intelligence': intelligenza_emotiva
        }])
        
        dati_inseriti.append(profilo_df)

st.divider()

# --- SEZIONE 3: IL MOTORE DI MATCHING ---
if st.button("Trova i Match! 💘", use_container_width=True):
    with st.spinner("L'Intelligenza Artificiale sta analizzando i profili..."):
        
        risultati_cluster = {}
        
        for i, df_persona in enumerate(dati_inseriti):
            cluster = assegna_cluster(df_persona)
            
            if cluster not in risultati_cluster:
                risultati_cluster[cluster] = []
            risultati_cluster[cluster].append(f"Persona {i+1}")
            
        st.success("Analisi completata!")
        st.header("🎯 Risultati del Matching")
        
        match_trovati = False
        
        for cluster, persone in risultati_cluster.items():
            if len(persone) > 1:
                match_trovati = True
                st.balloons()
                
                # Recuperiamo i testi personalizzati per il cluster specifico
                info_cluster = testi_cluster.get(cluster, {"titolo": "Profilo Sconosciuto", "descrizione": "Affinità rilevata."})
                
                st.write(f"### ✨ MATCH PERFETTO: {info_cluster['titolo']}!")
                st.write(f"C'è altissima affinità tra: **{', '.join(persone)}** (Cluster {cluster})")
                st.info(info_cluster['descrizione'])
                
        if not match_trovati:
            st.warning("Nessun match perfetto trovato. Tutte le persone appartengono a profili psicologici molto diversi (Cluster differenti).")