import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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

# Create a network graph
G = nx.Graph()

# Add edges based on co-occurrence in documents
for doc in original_keywords:
    keywords = [kw[0] for kw in doc]
    for i, kw1 in enumerate(keywords):
        for kw2 in keywords[i + 1 :]:
            if G.has_edge(kw1, kw2):
                G[kw1][kw2]["weight"] += 1
            else:
                G.add_edge(kw1, kw2, weight=1)

# Extract edge weights
weights = np.array([G[u][v]["weight"] for u, v in G.edges()])

# Normalize edge weights for visualization
normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())

# Use a colormap for edge colors
from matplotlib import cm

cmap = cm.viridis
edge_colors = [cmap(w) for w in normalized_weights]

# Visualize the network graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, seed=42)  # Layout for node positioning
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.8)
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=2)
nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

plt.title("Network Graph of Keywords with Co-occurrence")
plt.axis("off")
plt.show()
