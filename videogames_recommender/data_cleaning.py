import pandas as pd

print("Inizio pulizia dati Steam...")

# 1. Carichiamo il dataset
df = pd.read_csv('steam.csv')

# 2. Teniamo le colonne che ci servono davvero
df = df[['name', 'release_date', 'developer', 'steamspy_tags', 'owners', 'positive_ratings']]
df = df.dropna()

# 3. Trasformiamo le vendite stimate in numeri (prendiamo il numero prima del trattino)
# Es. da "20000-50000" diventa 20000
df['min_owners'] = df['owners'].str.split('-').str[0].astype(int)

# 4. Prepariamo i tag: sostituiamo il ';' con uno spazio (' ')
df['tags_clean'] = df['steamspy_tags'].str.replace(';', ' ')

# 5. Filtro opzionale ma consigliato: teniamo solo i giochi con almeno 20.000 possessori.
# Steam è pieno di "giochi spazzatura" o amatoriali, questo filtro alleggerisce il modello e migliora le raccomandazioni!
df = df[df['min_owners'] >= 20000]

# Resettiamo l'indice
df = df.reset_index(drop=True)

# Salviamo il file
df.to_pickle('steam_pulito.pkl')

print(f"Dati puliti e salvati! Sono rimasti {len(df)} giochi validi nel database.")
