import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu
import os
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configurazione iniziale della pagina
st.set_page_config(page_title="Steam Recommender", page_icon="🎮", layout="wide")

st.title("🎮 Steam Game Recommender")
st.write("Trova il tuo prossimo gioco preferito analizzando i tag della community!")

# --- 1. CARICAMENTO DATI E MODELLI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    percorso_knn = os.path.join(BASE_DIR, 'knn_steam_model.pkl')
    percorso_vect = os.path.join(BASE_DIR, 'vectorizer_steam.pkl')
    knn = joblib.load(percorso_knn)
    vect = joblib.load(percorso_vect)
    return knn, vect

@st.cache_data
def load_dataframe():
    percorso_df = os.path.join(BASE_DIR, 'steam_pulito.pkl')
    df = pd.read_pickle(percorso_df)
    # Estraiamo l'anno per usarlo nel clustering
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    return df

knn_model, vectorizer = load_models()
df = load_dataframe()

# --- 2. MENU DI NAVIGAZIONE ---
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
    Stiamo usando il famoso dataset **Steam Store Games**. 
    Dopo aver filtrato i giochi meno conosciuti (sotto le 20.000 copie stimate), 
    analizziamo **{len(df)} videogiochi unici**. Il punto di forza? Analizziamo i **Tag della Community**!
    """)
    st.write("Anteprima dei dati:")
    st.dataframe(df[['name', 'release_date', 'developer', 'steamspy_tags', 'owners']].head(10))

elif scelta == "Trova Giochi Simili":
    st.header("Il tuo Personal Shopper Videoludico")
    lista_giochi = sorted(df['name'].unique())
    gioco_scelto = st.selectbox(
        "Scegli un gioco che hai amato:",
        lista_giochi,
        index=None,
        placeholder="Es. Portal 2, The Witcher 3, Terraria..."
    )
    
    if st.button("Raccomandami giochi simili!", type="primary"):
        if gioco_scelto is None:
            st.warning("⚠️ Seleziona un gioco prima di cliccare!")
        else:
            # Troviamo i tag del gioco scelto
            tags_gioco = df[df['name'] == gioco_scelto]['tags_clean'].values[0]
            
            # Trasformiamo i tag in numeri usando il nostro vectorizer
            vettore_gioco = vectorizer.transform([tags_gioco])
            
            # Troviamo i vicini
            distanze, indici = knn_model.kneighbors(vettore_gioco)
            
            st.success(f"Se ti è piaciuto **{gioco_scelto}**, ecco 5 giochi con tag molto simili:")
            for i in range(1, len(indici[0])):
                indice_vicino = indici[0][i]
                nome = df.iloc[indice_vicino]['name']
                sviluppatore = df.iloc[indice_vicino]['developer']
                tag_originali = df.iloc[indice_vicino]['steamspy_tags']
                st.info(f"**{i}. {nome}** | Sviluppatore: {sviluppatore} | Tag: {tag_originali}")

elif scelta == "Clustering K-Means":
    st.header("Analisi dei Gruppi (K-Means Clustering)")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configurazione")
        n_giochi = st.slider("Numero di giochi da visualizzare", min_value=100, max_value=2000, value=500, step=100)
        seed_campionamento = st.slider("Seed per campionamento", min_value=0, max_value=100, value=42)
        
        st.write("---")
        st.subheader("Caratteristiche")
        usa_recensioni = st.checkbox("Recensioni Positive", value=True)
        usa_vendite = st.checkbox("Vendite (Minime Stimate)", value=True)
        usa_anno = st.checkbox("Anno di Uscita", value=False)
        
        st.write("---")
        k = st.slider("Numero di cluster (k)", min_value=2, max_value=8, value=4)

    with col2:
        df_cluster = df.sample(n=n_giochi, random_state=seed_campionamento).copy()
        
        feature_cols = []
        if usa_recensioni: feature_cols.append('positive_ratings')
        if usa_vendite: feature_cols.append('min_owners')
        if usa_anno: feature_cols.append('year')
        
        if len(feature_cols) < 2:
            st.warning("⚠️ Seleziona almeno due caratteristiche per visualizzare il grafico.")
        else:
            # Gestione dei NaN per l'anno (se presente)
            df_cluster = df_cluster.dropna(subset=feature_cols)
            X = df_cluster[feature_cols]
            
            scaler_kmeans = StandardScaler()
            X_scaled = scaler_kmeans.fit_transform(X)
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            df_cluster['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)
            
            fig = px.scatter(
                df_cluster, x=feature_cols[0], y=feature_cols[1],
                color="Cluster", hover_name="name", hover_data=["developer"],
                title=f"Distribuzione Videogiochi: {feature_cols[0]} vs {feature_cols[1]}",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

elif scelta == "Come Funziona":
    st.header("Dietro le quinte")
    st.write("""
    Abbiamo migliorato l'app passando a un sistema **Content-Based Filtering**.
    Invece di guardare solo il genere principale, l'algoritmo legge tutti i **Tag della Community** assegnati a un gioco (es. 'Open World', 'Gore', 'Story Rich').
    
    Usiamo un `CountVectorizer` per trasformare queste parole in una matrice matematica. Poi, tramite il **K-Nearest Neighbors** calcoliamo la *Cosine Similarity* (Similarità del Coseno) per trovare i giochi che condividono il maggior numero di tag specifici con il tuo preferito!
    """)
