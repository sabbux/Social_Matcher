import streamlit as st
import pandas as pd
import os
import sys

# --- 1. CONFIGURAZIONE PERCORSI ---
cartella_script = os.path.dirname(os.path.abspath(__file__))
if cartella_script not in sys.path:
    sys.path.append(cartella_script)

cartella_kmeans = os.path.dirname(cartella_script)
if cartella_kmeans not in sys.path:
    sys.path.append(cartella_kmeans)

# Importiamo le funzioni
from predict import assegna_cluster
from matcher.similarity_matching import trova_match_per_omofilia

# --- 2. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Social Matcher AI", page_icon="🧬", layout="wide")

st.title("🧬 Social Matcher AI")
st.markdown("Analisi della personalità e matching basato sull'ipotesi dell'omofilia.")

# --- 3. COSTANTI E DIZIONARI ---
eta_options = ['18-30', '31-43', '44-55']
lavoro_options = [
    'Tech', 'Healthcare', 'Finance', 'Education', 'Creative Arts', 
    'Law', 'Engineering', 'Marketing', 'Science', 'Entrepreneurship'
]
stile_options = [
    'Physical Warmth', 'Thoughtful Gestures', 'Practical Reliability', 
    'Shared Experiences', 'Verbal Support'
]
mappa_istruzione = {
    "Diploma Superiore (High School)": 1,
    "Diploma Universitario (Associate's)": 2,
    "Laurea Triennale (Bachelor's)": 3,
    "Laurea Magistrale (Master's)": 4,
    "Dottorato di Ricerca (Doctorate)": 5
}
opzioni_istruzione = list(mappa_istruzione.keys())

# Dizionario inverso per tradurre il numero estratto dal DB di nuovo in testo
mappa_istruzione_inversa = {v: k for k, v in mappa_istruzione.items()}

testi_cluster = {
    0: {"titolo": "Gli Strutturati / I Tradizionalisti", "descrizione": "Ami l'organizzazione, la stabilità e la pianificazione. Cerchi rapporti solidi e concreti."},
    1: {"titolo": "I Leader Dinamici", "descrizione": "Sei socievole, ambizioso e carismatico. Cerchi persone con cui conquistare grandi obiettivi."},
    2: {"titolo": "Gli Spiriti Liberi / I Rilassati", "descrizione": "Ami la spontaneità e la flessibilità. Cerchi connessioni autentiche senza troppe regole."},
    3: {"titolo": "Gli Intellettuali Introversi", "descrizione": "Ami le conversazioni profonde e il rispetto dei tuoi spazi. Cerchi una connessione mentale d'alto livello."}
}

# --- 4. CREAZIONE DELLE TAB ---
tab1, tab2 = st.tabs(["👥 Confronto di Gruppo", "🔍 Trova Profili Affini"])

# =========================================================
# TAB 1: MATCH DI GRUPPO
# =========================================================
with tab1:
    st.header("Confronta tra loro un gruppo di persone")
    n_persone = st.number_input("Quante persone vuoi confrontare?", min_value=2, max_value=5, value=2, key="n_persone_tab1")

    dati_inseriti = [] 
    tabs_persone = st.tabs([f"Persona {i+1}" for i in range(n_persone)])

    for i, tab in enumerate(tabs_persone):
        with tab:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Dati e Lavoro**")
                eta = st.selectbox("Fascia d'età", eta_options, key=f"eta_{i}")
                istruzione_scelta = st.selectbox("Livello Istruzione", opzioni_istruzione, index=2, key=f"edu_label_{i}")
                istruzione_num = mappa_istruzione[istruzione_scelta]
                lavoro = st.selectbox("Settore Lavorativo", lavoro_options, key=f"lav_{i}")
                stile = st.selectbox("Stile Comunicativo", stile_options, key=f"stile_{i}")
            with col2:
                st.markdown("**Personalità**")
                apertura = st.slider("Apertura Mentale", 0.0, 1.0, 0.5, key=f"ope_{i}")
                estroversione = st.slider("Estroversione", 0.0, 1.0, 0.5, key=f"ext_{i}")
                amicalita = st.slider("Amicalità", 0.0, 1.0, 0.5, key=f"agr_{i}")
                coscienziosita = st.slider("Coscienziosità", 0.0, 1.0, 0.5, key=f"con_{i}")
            with col3:
                st.markdown("**Attitudini**")
                ambizione = st.slider("Ambizione", 0.0, 1.0, 0.5, key=f"amb_{i}")
                cronotipo = st.slider("Cronotipo", 0.0, 1.0, 0.5, key=f"chr_{i}")
                spontaneita = st.slider("Spontaneità", 0.0, 1.0, 0.5, key=f"spo_{i}")
                intelligenza_emotiva = st.slider("Intelligenza Emotiva", 0.0, 1.0, 0.5, key=f"emo_{i}")
            
            dati_inseriti.append(pd.DataFrame([{
                'age': eta, 'education': istruzione_num, 'career_field': lavoro, 'career_ambition': ambizione,
                'openness': apertura, 'extraversion': estroversione, 'agreeableness': amicalita,
                'conscientiousness': coscienziosita, 'chronotype': cronotipo, 'spontaneity': spontaneita,
                'communication_style': stile, 'emotional_intelligence': intelligenza_emotiva
            }]))

    if st.button("Analizza Affinità di Gruppo 🤝", use_container_width=True):
        risultati_cluster = {}
        for i, df_persona in enumerate(dati_inseriti):
            c = assegna_cluster(df_persona)
            if c not in risultati_cluster: risultati_cluster[c] = []
            risultati_cluster[c].append(f"Persona {i+1}")
        
        st.header("🎯 Risultati")
        match_trovati = False
        for cluster, persone in risultati_cluster.items():
            if len(persone) > 1:
                match_trovati = True
                st.balloons()
                info = testi_cluster.get(cluster, {"titolo": "Match Rilevato", "descrizione": ""})
                st.success(f"### ✨ MATCH: {info['titolo']}!")
                st.write(f"Affinità rilevata tra: **{', '.join(persone)}**")
                st.info(info['descrizione'])
        if not match_trovati:
            st.warning("Nessun match perfetto. I profili appartengono a gruppi psicologici differenti.")

# =========================================================
# TAB 2: TROVA PROFILI AFFINI
# =========================================================
with tab2:
    st.header("🔍 Trova persone compatibili con te")
    st.write("Inserisci i tuoi dati: l'IA cercherà nel database 3 profili con la tua stessa struttura psicologica.")
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Chi sei?**")
        e_u = st.selectbox("Tua fascia d'età", eta_options, key="u_eta")
        i_u = st.selectbox("Tua istruzione", opzioni_istruzione, index=2, key="u_edu")
        i_n_u = mappa_istruzione[i_u]
        l_u = st.selectbox("Tuo settore", lavoro_options, key="u_lav")
        s_u = st.selectbox("Tuo stile comunicativo", stile_options, key="u_stile")
    with colB:
        st.markdown("**Personalità**")
        o_u = st.slider("Apertura Mentale", 0.0, 1.0, 0.5, key="u_ope")
        ex_u = st.slider("Estroversione", 0.0, 1.0, 0.5, key="u_ext")
        ag_u = st.slider("Amicalità", 0.0, 1.0, 0.5, key="u_agr")
        co_u = st.slider("Coscienziosità", 0.0, 1.0, 0.5, key="u_con")
    with colC:
        st.markdown("**Stile di vita**")
        am_u = st.slider("Ambizione", 0.0, 1.0, 0.5, key="u_amb")
        ch_u = st.slider("Cronotipo", 0.0, 1.0, 0.5, key="u_chr")
        sp_u = st.slider("Spontaneità", 0.0, 1.0, 0.5, key="u_spo")
        ei_u = st.slider("Intelligenza Emotiva", 0.0, 1.0, 0.5, key="u_emo")

    if st.button("Cerca nel Database 🚀", use_container_width=True):
        with st.spinner("Interrogazione del database in corso..."):
            
            df_u = pd.DataFrame([{'age': e_u, 'education': i_n_u, 'career_field': l_u, 'career_ambition': am_u,
                                 'openness': o_u, 'extraversion': ex_u, 'agreeableness': ag_u,
                                 'conscientiousness': co_u, 'chronotype': ch_u, 'spontaneity': sp_u,
                                 'communication_style': s_u, 'emotional_intelligence': ei_u}])
            
            cluster_utente = assegna_cluster(df_u)
            
            risultati = trova_match_per_omofilia(cluster_target=cluster_utente, numero_match_desiderati=3)
            
            if risultati is not None and not risultati.empty:
                st.success(f"Analisi completata! Appartieni al **Cluster {cluster_utente}**.")
                info_c = testi_cluster.get(cluster_utente, {"titolo": "", "descrizione": ""})
                st.markdown(f"**Identikit:** {info_c['titolo']}")
                st.divider()
                st.subheader("👥 Ecco i profili più compatibili con te:")
                
                # Mostriamo i risultati completi in card espandibili
                for _, row in risultati.iterrows():
                    # Traduciamo il numero dell'istruzione di nuovo in testo leggibile
                    istruzione_testo = mappa_istruzione_inversa.get(row['education'], "Non specificato")
                    
                    with st.expander(f"🧩 Profilo Affine (Settore: {row['career_field']})"):
                        c1, c2, c3 = st.columns(3)
                        
                        with c1:
                            st.write("**Anagrafica e Lavoro**")
                            st.write(f"• Età: {row['age']}")
                            st.write(f"• Istruzione: {istruzione_testo}")
                            st.write(f"• Stile Com.: {row['communication_style']}")
                        
                        with c2:
                            st.write("**Personalità**")
                            st.write(f"• Apertura: {row['openness']:.2f}")
                            st.write(f"• Estroversione: {row['extraversion']:.2f}")
                            st.write(f"• Amicalità: {row['agreeableness']:.2f}")
                            st.write(f"• Coscienziosità: {row['conscientiousness']:.2f}")
                            
                        with c3:
                            st.write("**Attitudini**")
                            st.write(f"• Ambizione: {row['career_ambition']:.2f}")
                            st.write(f"• Cronotipo: {row['chronotype']:.2f}")
                            st.write(f"• Spontaneità: {row['spontaneity']:.2f}")
                            st.write(f"• Int. Emotiva: {row['emotional_intelligence']:.2f}")
            else:
                st.error("Spiacenti, non abbiamo trovato profili compatibili nel database al momento.")