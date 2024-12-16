import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('final_happy.csv', sep=',')

numeric_columns = [
    "Life Ladder", "Log GDP per capita", "Social support",
    "Healthy life expectancy at birth", "Freedom to make life choices",
    "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect"
]

# Supprimer les lignes avec des valeurs manquantes
data_clean = data.dropna(subset=numeric_columns)

# Standardiser les données
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_clean[numeric_columns])

# Réaliser l'Analyse en Composantes Principales (ACP)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Afficher la variance expliquée
print("Variance expliquée par les composantes principales :", pca.explained_variance_ratio_)

# Appliquer K-Means sur les données réduites
kmeans = KMeans(n_clusters=3, random_state=42) #k=3
clusters = kmeans.fit_predict(pca_data)

# Ajouter les clusters au jeu de données
data_clean['Cluster'] = clusters

plt.figure(figsize=(10, 7))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("Visualisation des clusters après ACP")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.colorbar(label="Cluster")
plt.show()

#Enregistrer les résultats
data_clean.to_csv("happiness_with_clusters.csv", index=False)
print("Les résultats ont été sauvegardés dans 'happiness_with_clusters.csv'")

# Calculer les moyennes des variables par cluster
cluster_analysis = data_clean.groupby('Cluster')[numeric_columns].mean()

print("Caractéristiques moyennes par cluster :")
print(cluster_analysis)



# Filtrer les pays pour un bon niveau de bonheur et de soutien social
happy_countries = data_clean[
    (data_clean['Life Ladder'] > 6.5) &
    (data_clean['Social support'] > 0.8)
].sort_values(by='Life Ladder', ascending=False)

# Calculer les moyennes des variables par cluster
cluster_analysis = data_clean.groupby('Cluster')[numeric_columns].mean()

print("Caractéristiques moyennes par cluster :")
print(cluster_analysis)

# Afficher les recommandations
print("Pays recommandés pour un haut niveau de bonheur :")
print(happy_countries[['Country name', 'Life Ladder', 'Social support']].head(10))

# Filtrer les pays pour une bonne espérance de vie et une faible corruption
long_life_countries = data_clean[
    (data_clean['Healthy life expectancy at birth'] > 70) &
    (data_clean['Perceptions of corruption'] < 0.3)
].sort_values(by='Healthy life expectancy at birth', ascending=False)

# Afficher les recommandations
print("Pays recommandés pour une bonne espérance de vie :")
print(long_life_countries[['Country name', 'Healthy life expectancy at birth', 'Perceptions of corruption']].head(10))

# Filtrer les pays pour une grande liberté individuelle et une générosité élevée
free_countries = data_clean[
    (data_clean['Freedom to make life choices'] > 0.8) &
    (data_clean['Generosity'] > 0.2)
].sort_values(by='Freedom to make life choices', ascending=False)

# Afficher les recommandations
print("Pays recommandés pour une grande liberté individuelle :")
print(free_countries[['Country name', 'Freedom to make life choices', 'Generosity']].head(10))

# Ajouter un score de sécurité basé sur la perception de corruption (moins de corruption = plus de sécurité)
data_clean['Security Score'] = 1 - data_clean['Perceptions of corruption']

# Trier les pays par le score de sécurité (du plus sécurisé au moins sécurisé)
secure_countries = data_clean.sort_values(by='Security Score', ascending=False)

# Afficher les 10 pays les plus sécurisés
print("Top 10 des pays les plus sécurisés :")
print(secure_countries[['Country name', 'Security Score', 'Perceptions of corruption']].head(10))


# Sauvegarder les objets pour Flask
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(pca, open("pca.pkl", "wb"))
pickle.dump(kmeans, open("kmeans.pkl", "wb"))
