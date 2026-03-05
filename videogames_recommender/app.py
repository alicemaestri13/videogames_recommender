import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import os
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Steam Game Recommender", page_icon="🎮", layout="wide")

st.title("🎮 Steam Game Recommender")
st.write("Trova il tuo prossimo gioco preferito grazie all'Intelligenza Artificiale e ai tag della community!")

# --- 1. CARICAMENTO DATI E MODELLI ---
# Usiamo os per assicurarci che Streamlit trovi i file ovunque si trovi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    knn = joblib.load(os.path.join(BASE_DIR, 'knn_steam_model.pkl'))
    vect = joblib.load(os.path.join(BASE_DIR, 'vectorizer_steam.pkl'))
    return knn, vect

@st.cache_data
def load_dataframe():
    df = pd.read_pickle(os.path.join(BASE_DIR, 'steam_pulito.pkl'))
    # Creiamo una colonna con solo l'anno per il clustering
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    return df

knn_model, vectorizer = load_models()
df = load_dataframe()

# --- 2. MENU DI NAVIGAZIONE A BOTTONI ---
scelta = option_menu(
    menu_title=None,
    options=["Esplorazione Dati", "Trova Giochi Simili", "Clustering K-Means", "Come Funziona"],
    icons=["bar-chart-line", "controller", "pie-chart", "lightbulb"],
    default_index=1,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f2f6"},
        "icon": {"color": "#ff4b4b", "font-size": "25px"}, 
        "nav-link": {"font-size": "18px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
    }
)

st.write("---")

# --- 3. CONTENUTI DELLE SCHEDE ---

if scelta == "Esplorazione Dati":
    st.header("Esplorazione del Dataset Steam")
    st.info(f"""
    **🎮 Il Dataset in pillole:**
    Per questo progetto abbiamo utilizzato il famoso dataset **Steam Store Games**. 
    Dopo un'accurata pulizia (filtrando i giochi con meno di 20.000 copie stimate per mantenere alta la qualità), 
    il nostro motore analizza **{len(df)} videogiochi unici**. 
    """)
    st.write("Ecco un'anteprima dei dati su cui lavora l'algoritmo:")
    st.dataframe(df[['name', 'release_date', 'developer', 'steamspy_tags', 'min_owners']].head(15))

elif scelta == "Trova Giochi Simili":
    st.header("Il tuo Personal Shopper Videoludico")
    
    lista_giochi = sorted(df['name'].unique())
    gioco_scelto = st.selectbox(
        "Scegli un gioco che hai amato (inizia a digitare):", 
        lista_giochi,
        index=None,
        placeholder="Es. Portal 2, The Elder Scrolls V: Skyrim, Terraria..."
    )
    
    if st.button("Raccomandami giochi simili!", type="primary"):
        if gioco_scelto is None:
            st.warning("⚠️ Per favore, seleziona un gioco prima di cliccare!")
        else:
            # 1. Recuperiamo i tag del gioco scelto
            tags_gioco = df[df['name'] == gioco_scelto]['tags_clean'].values[0]
            
            # 2. Trasformiamo le parole in numeri (Vettorizzazione)
            vettore_gioco = vectorizer.transform([tags_gioco])
            
            # 3. L'algoritmo calcola le distanze e trova i vicini
            distanze, indici = knn_model.kneighbors(vettore_gioco)
            
            st.success(f"Ottima scelta! Se ti è piaciuto **{gioco_scelto}**, ecco 5 titoli con tag molto simili:")
            
            # Mostriamo i risultati
            for i in range(1, len(indici[0])):
                indice_vicino = indici[0][i]
                nome = df.iloc[indice_vicino]['name']
                sviluppatore = df.iloc[indice_vicino]['developer']
                tag_originali = df.iloc[indice_vicino]['steamspy_tags']
                
                # Sostituiamo i punti e virgola con una virgola per renderli più leggibili
                tag_belli = tag_originali.replace(';', ', ')
                
                st.info(f"**{i}. {nome}** | Sviluppatore: {sviluppatore} | Tag: {tag_belli}")

elif scelta == "Clustering K-Means":
    st.header("Analisi dei Gruppi (K-Means Clustering)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configurazione")
        n_giochi = st.slider("Giochi da visualizzare", min_value=100, max_value=2000, value=500, step=100)
        seed_camp = st.slider("Seed campionamento", min_value=0, max_value=100, value=42)
        
        st.write("---")
        st.subheader("Caratteristiche")
        usa_voti = st.checkbox("Recensioni Positive", value=True)
        usa_vendite = st.checkbox("Vendite Stimate", value=True)
        usa_anno = st.checkbox("Anno di Uscita", value=False)
        
        st.write("---")
        st.subheader("Parametri K-Means")
        k = st.slider("Numero di cluster (k)", min_value=2, max_value=8, value=4)

    with col2:
        # Peschiamo un campione casuale
        df_cluster = df.sample(n=n_giochi, random_state=seed_camp).copy()
        
        feature_cols = []
        if usa_voti: feature_cols.append('positive_ratings')
        if usa_vendite: feature_cols.append('min_owners')
        if usa_anno: feature_cols.append('year')
        
        if len(feature_cols) < 2:
            st.warning("⚠️ Seleziona almeno due caratteristiche dalla barra laterale per generare il grafico.")
        else:
            # Rimuoviamo eventuali giochi senza anno (se selezionato)
            df_cluster = df_cluster.dropna(subset=feature_cols)
            X = df_cluster[feature_cols]
            
            # Standardizziamo i dati
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Applichiamo K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            df_cluster['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)
            
            # Creiamo il grafico
            fig = px.scatter(
                df_cluster, 
                x=feature_cols[0], 
                y=feature_cols[1], 
                color="Cluster",
                hover_name="name", 
                hover_data=["developer", "year"],
                title=f"Distribuzione Videogiochi: {feature_cols[0]} vs {feature_cols[1]}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)") 
            st.plotly_chart(fig, use_container_width=True)

elif scelta == "Come Funziona":
    st.header("Dietro le quinte: Content-Based Filtering")
    st.write("""
    Questo sistema di raccomandazione rappresenta un'evoluzione rispetto al filtraggio classico.
    
    Invece di basarsi solo su un singolo genere o anno di uscita, l'algoritmo analizza il linguaggio naturale (**NLP**) dei **Tag della Community di Steam** (es. *'Open World', 'Story Rich', 'Gore'*).
    
    ### I Passaggi:
    1. **Vettorizzazione (`CountVectorizer`):** Trasformiamo le etichette testuali di ogni gioco in una matrice matematica di 0 e 1.
    2. **Calcolo della Distanza:** Utilizziamo l'algoritmo **K-Nearest Neighbors (KNN)** con la metrica della **Cosine Similarity** (Similarità del Coseno).
    
    La Similarità del Coseno calcola l'angolo tra i "vettori" dei giochi: più due giochi condividono tag specifici, più i loro vettori sono vicini nello spazio multidimensionale, permettendoci di offrirti raccomandazioni incredibilmente precise!
    """)
