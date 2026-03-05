import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

print("Inizio addestramento del modello sui Tag di Steam...")

# 1. Carica i dati puliti
df = pd.read_pickle('steam_pulito.pkl')

# 2. Trasformiamo i tag di testo in una matrice di numeri (Vettorizzazione)
vectorizer = CountVectorizer()
tag_matrix = vectorizer.fit_transform(df['tags_clean'])

# 3. Addestriamo il modello KNN usando la "cosine similarity"
knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
knn_model.fit(tag_matrix)

# 4. SALVATAGGIO CON JOBLIB
joblib.dump(knn_model, 'knn_steam_model.pkl')
# IMPORTANTE: Dobbiamo salvare anche il vectorizer, perché Streamlit ne avrà bisogno!
joblib.dump(vectorizer, 'vectorizer_steam.pkl')

print("Modello basato sui tag addestrato e salvato con successo!")
