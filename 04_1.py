import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from adjustText import adjust_text

# Constants for styling
COLOR_CLUSTER1 = "#ecb39c"  # Light salmon color
COLOR_CLUSTER2 = "#44827f"  # Teal color
CLUSTER_COLORS = [
    COLOR_CLUSTER1,
    COLOR_CLUSTER2,
    "#99d3c4",
    "#ffe6b7",
]  # Additional colors for more clusters

# Example data: List of documents with keywords and their relevance scores
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

# Convert keywords to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_keywords)

# Perform K-means clustering
num_clusters = 4  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2, random_state=42)
reduced_data = pca.fit_transform(X.toarray())

# Create a DataFrame for visualization
df = pd.DataFrame(reduced_data, columns=["PCA1", "PCA2"])
df["Cluster"] = kmeans.labels_
df["Keyword"] = all_keywords

# Visualize the clusters
plt.figure(figsize=(12, 8))
for cluster in range(num_clusters):
    cluster_data = df[df["Cluster"] == cluster]
    color = CLUSTER_COLORS[
        cluster % len(CLUSTER_COLORS)
    ]  # Cycle through defined colors
    plt.scatter(
        cluster_data["PCA1"],
        cluster_data["PCA2"],
        color=color,
        label=f"Cluster {cluster}",
        alpha=0.7,
        edgecolors="black",
        s=100,
    )

# Annotate points with keywords using adjustText
texts = []
for _, row in df.iterrows():
    texts.append(plt.text(row["PCA1"], row["PCA2"], row["Keyword"], fontsize=8))

adjust_text(
    texts,
    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
    force_points=0.3,
    force_text=0.6,
    expand_text=(1.2, 1.4),
)

plt.title("K-Means Clustering of Keywords", fontsize=16)
plt.xlabel("PCA1", fontsize=12)
plt.ylabel("PCA2", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
