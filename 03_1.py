import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgb  # Import to_rgb from matplotlib.colors

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

# Define the colors and opacities
color_group1_5 = "#ecb39c"  # RGB color for groups 1-5
color_group6_10 = "#44827f"  # RGB color for groups 6-10
opacities = [0.9, 0.7, 0.5, 0.3, 0.1]  # Opacities for each group

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

# Assign colors and opacities to nodes based on their group
node_colors = {}
for i, doc in enumerate(original_keywords):
    group_color = (
        color_group1_5 if i < 5 else color_group6_10
    )  # Choose color based on group
    opacity = opacities[i % 5]  # Assign opacity based on position in group
    rgba_color = (*to_rgb(group_color), opacity)  # Convert to RGBA with opacity
    for kw, _ in doc:
        node_colors[kw] = rgba_color  # Assign the same color to all nodes in the group

# Extract edge weights
weights = np.array([G[u][v]["weight"] for u, v in G.edges()])

# Normalize edge weights for visualization
if weights.max() != weights.min():  # Avoid division by zero
    normalized_weights = (weights - weights.min()) / (weights.max() - weights.min())
else:
    normalized_weights = np.zeros_like(weights)  # If all weights are the same

# Use a colormap for edge colors
from matplotlib import cm

cmap = cm.viridis
edge_colors = [cmap(w) for w in normalized_weights]  # Use colormap for edge colors

# Visualize the network graph
plt.figure(figsize=(15, 15))
pos = nx.spring_layout(G, k=0.5, seed=42)  # Layout for node positioning
# pos = nx.kamada_kawai_layout(G)  # Alternative layout

# Draw nodes with assigned colors and opacities
node_sizes = [G.degree(node) * 200 for node in G.nodes()]  # Scale node size by degree
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=[node_colors[node] for node in G.nodes()],  # Use colors for unique nodes
    edgecolors="black",
    linewidths=1,
)

# Draw edges
edge_widths = [
    G[u][v]["weight"] * 1 for u, v in G.edges()
]  # Scale edge width by weight
nx.draw_networkx_edges(
    G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_widths, alpha=0.6
)

# edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

# Add title and display
plt.title("Network Graph of Keywords with Co-occurrence")
plt.axis("off")
plt.show()
