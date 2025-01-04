import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import networkx as nx

# Example data: List of documents with keywords and scores
original_keywords = [
    [
        ("visit chichester", 0.4869),
        ("many tourists", 0.4692),
        ("on chichester", 0.4426),
        ("tourists", 0.4312),
        ("chichester", 0.4257),
    ],
    [
        ("recycled water", 0.4744),
        ("waste water", 0.4664),
        ("waster water", 0.4465),
        ("water treatment", 0.4201),
        ("homes should", 0.3973),
    ],
    [
        ("upgrading water", 0.4929),
        ("water treatment", 0.4535),
        ("second homes", 0.4039),
        ("homes will", 0.3876),
        ("the tax", 0.3805),
    ],
    [
        ("water authorities", 0.508),
        ("our harbours", 0.4942),
        ("harbours", 0.471),
        ("the beachbuoy", 0.4645),
        ("beachbuoy", 0.4506),
    ],
    [
        ("oyster population", 0.6412),
        ("the oyster", 0.4952),
        ("oyster", 0.4828),
        ("itchenor channel", 0.4592),
        ("how healthy", 0.4192),
    ],
    [
        ("fibreglass boats", 0.7266),
        ("fibreglass", 0.6118),
        ("used fibreglass", 0.6085),
        ("boats", 0.5453),
        ("boats in", 0.5326),
    ],
    [
        ("chichester area", 0.4948),
        ("chichester is", 0.4308),
        ("chichester", 0.4246),
        ("of chichester", 0.4244),
        ("crested newts", 0.4078),
    ],
    [
        ("chichester harbour", 0.4377),
        ("household water", 0.4181),
        ("flood risk", 0.4061),
        ("in chichester", 0.3966),
        ("water use", 0.392),
    ],
    [
        ("water authorities", 0.6156),
        ("storm overflows", 0.5592),
        ("authorities infringes", 0.4471),
        ("the legislation", 0.4248),
        ("storm", 0.4191),
    ],
    [
        ("species soil", 0.4389),
        ("water contamination", 0.4329),
        ("flood risk", 0.4127),
        ("of microplastic", 0.4059),
        ("microplastic", 0.4017),
    ],
]

# Extract all keywords
all_keywords = [keyword for keyword, _ in [k for doc in original_keywords for k in doc]]

# Step 1: Co-occurrence Matrix from Network Graph
G_full = nx.Graph()

# Build the co-occurrence graph
for doc in original_keywords:
    keywords = [kw[0] for kw in doc]
    for i, kw1 in enumerate(keywords):
        for kw2 in keywords[i + 1 :]:
            if G_full.has_edge(kw1, kw2):
                G_full[kw1][kw2]["weight"] += 1
            else:
                G_full.add_edge(kw1, kw2, weight=1)

# Create co-occurrence matrix
co_occurrence_matrix = np.zeros((len(all_keywords), len(all_keywords)))
for u, v, data in G_full.edges(data=True):
    if u in all_keywords and v in all_keywords:
        i, j = all_keywords.index(u), all_keywords.index(v)
        co_occurrence_matrix[i][j] = data["weight"]
        co_occurrence_matrix[j][i] = data["weight"]

# Step 2: Semantic Similarity Matrix (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_keywords)
semantic_similarity_matrix = cosine_similarity(X)

# Step 3: Normalize and Combine Matrices
scaler = MinMaxScaler()
co_occurrence_matrix_norm = scaler.fit_transform(co_occurrence_matrix)
semantic_similarity_matrix_norm = scaler.fit_transform(semantic_similarity_matrix)

# Combine the two matrices with equal weighting
combined_matrix = (co_occurrence_matrix_norm + semantic_similarity_matrix_norm) / 2

# Step 4: K-Means Clustering
num_clusters = 4
kmeans_combined = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_combined.fit(combined_matrix)

# Step 5: Dimensionality Reduction for Visualization
reduced_combined_data = PCA(n_components=2, random_state=42).fit_transform(
    combined_matrix
)

# Step 6: Visualization
df_combined = pd.DataFrame(reduced_combined_data, columns=["PCA1", "PCA2"])
df_combined["Cluster"] = kmeans_combined.labels_
df_combined["Keyword"] = all_keywords

plt.figure(figsize=(12, 8))
for cluster in range(num_clusters):
    cluster_data = df_combined[df_combined["Cluster"] == cluster]
    plt.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}")

# Annotate points with keywords
for _, row in df_combined.iterrows():
    plt.text(row["PCA1"], row["PCA2"], row["Keyword"], fontsize=8, alpha=0.7)

plt.title("Hybrid Clustering of Keywords (Co-occurrence + Semantic Similarity)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.tight_layout()
plt.show()
