import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

resource_dir = os.path.join(base_dir, "resources")

os.makedirs(resource_dir, exist_ok=True)

MODEL_FILE = os.path.join(resource_dir, 'social_matcher_model.pkl')

def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

model = load_model()

# 2. Configurazione della Finestra Principale
root = tk.Tk()
root.title("Social Matcher - Calcolatore di Affinità IA")
root.geometry("750x750")
root.configure(padx=20, pady=20)

# Mappatura dei livelli di istruzione
mappa_istruzione = {
    1: "Diploma Superiore (High School)",
    2: "Diploma Universitario (Associate's)",
    3: "Laurea Triennale (Bachelor's)",
    4: "Laurea Magistrale (Master's)",
    5: "Dottorato di Ricerca (Doctorate)"
}
# Creiamo anche il dizionario inverso per tradurre velocemente dal testo al numero
istruzione_inversa = {v: k for k, v in mappa_istruzione.items()}

# Elenco delle variabili per separare i tipi di dato
text_features = ['age', 'career_field', 'communication_style']
numeric_features = ['education', 'career_ambition', 'openness', 'extraversion', 
                    'agreeableness', 'conscientiousness', 'chronotype', 
                    'spontaneity', 'emotional_intelligence']

# Valori esatti per i menu a tendina
dropdown_options = {
    'age': ['18-30', '31-43', '44-55'],
    'education': list(mappa_istruzione.values()), # Inseriamo i testi descrittivi
    'career_field': ['Creative Arts', 'Education', 'Engineering', 'Entrepreneurship', 
                     'Finance', 'Healthcare', 'Law', 'Marketing', 'Science', 'Tech'],
    'communication_style': ['Physical Warmth', 'Practical Reliability', 'Shared Experiences', 
                            'Thoughtful Gestures', 'Verbal Support']
}

# Dizionario per salvare i riferimenti a tutti i widget
entries = {}

# 3. Funzione per creare dinamicamente le colonne
def create_column(parent, prefix, title, col_idx):
    frame = tk.LabelFrame(parent, text=title, font=("Arial", 12, "bold"), padx=15, pady=15)
    frame.grid(row=0, column=col_idx, padx=10, sticky="nsew")

    all_features = text_features + numeric_features

    for i, feature_name in enumerate(all_features):
        full_col_name = f"{prefix}_{feature_name}"
        
        # Etichetta del campo (estetica)
        label_text = feature_name.replace('_', ' ').title() + ":"
        tk.Label(frame, text=label_text).grid(row=i, column=0, sticky="e", pady=5)
        
        # Menu a tendina
        if feature_name in dropdown_options:
            widget = ttk.Combobox(frame, values=dropdown_options[feature_name], width=26, state="readonly")
            widget.grid(row=i, column=1, pady=5, padx=5)
            # Mettiamo un default sensato (es. Laurea Triennale)
            if feature_name == 'education':
                widget.set(mappa_istruzione[3])
            else:
                widget.current(0)
            
        # Casella di testo classica per i numeri continui
        else:
            widget = tk.Entry(frame, width=28)
            widget.grid(row=i, column=1, pady=5, padx=5)
            widget.insert(0, "0.5") 
            
        entries[full_col_name] = widget

# Creiamo le due colonne
create_column(root, "a", "👤 Profilo A", 0)
create_column(root, "b", "👤 Profilo B", 1)

# 4. Funzione che scatta quando clicchi il pulsante
def calcola_affinita():
    if model is None:
        messagebox.showerror("Errore Critico", f"Modello '{MODEL_FILE}' non trovato!\nDevi prima addestrarlo eseguendo 'train.py'.")
        return

    data = {}
    try:
        # Legge tutti i valori dalla GUI e li converte
        for key, widget in entries.items():
            val = widget.get()
            
            # --- TRADUZIONE DELL'ISTRUZIONE ---
            # Se stiamo leggendo il campo education (sia A che B), convertiamo il testo in numero
            if 'education' in key:
                val = istruzione_inversa[val]

            # Se la variabile è di testo, la teniamo come stringa
            if any(t in key for t in text_features):
                data[key] = [val] 
            # Se è numerica, la convertiamo in float
            else:
                data[key] = [float(val)] 

        # Trasforma i dati in un DataFrame
        df_input = pd.DataFrame(data)

        # Converte le colonne testuali in 'category' per XGBoost
        for col in df_input.columns:
            if any(t in col for t in text_features):
                df_input[col] = df_input[col].astype('category')

        # Fa la previsione
        prediction = model.predict(df_input)[0]
        prediction = np.clip(prediction, 0, 100) # Forza tra 0 e 100

        # Mostra il risultato a schermo
        lbl_result.config(text=f"🎯 Affinità Calcolata: {prediction:.1f} %", fg="#2E7D32")

    except ValueError:
        messagebox.showerror("Errore di Inserimento", "Assicurati di aver inserito numeri validi (es. 0.5, 0.8) nei campi liberi!")
    except Exception as e:
        messagebox.showerror("Errore Sconosciuto", f"Si è verificato un errore:\n{e}")

# 5. Pulsante e Testo del Risultato
btn_calcola = tk.Button(root, text="🔮 Calcola Affinità", command=calcola_affinita, 
                        font=("Arial", 14, "bold"), bg="#1976D2", fg="white", padx=20, pady=10)
btn_calcola.grid(row=1, column=0, columnspan=2, pady=25)

lbl_result = tk.Label(root, text="Compila i dati e premi calcola", font=("Arial", 16, "bold"), fg="#424242")
lbl_result.grid(row=2, column=0, columnspan=2, pady=5)

# Avvia l'interfaccia grafica
if __name__ == "__main__":
    if model is None:
        messagebox.showwarning("Avviso", f"Attenzione: '{MODEL_FILE}' non trovato.")
    root.mainloop()