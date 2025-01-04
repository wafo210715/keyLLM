import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import to_rgb, Normalize
from matplotlib import cm

# Constants for styling
COLOR_GROUP1_5 = "#ecb39c"
COLOR_GROUP6_10 = "#44827f"
OPACITIES = [0.9, 0.7, 0.5, 0.3, 0.1]
LAYOUT_SEED = 42
LAYOUT_K = 0.5
NODE_SIZE_SCALE = 300
EDGE_WIDTH_SCALE = 2

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

# Build the graph
G = nx.Graph()
for doc in original_keywords:
    keywords = [kw[0] for kw in doc]
    for i, kw1 in enumerate(keywords):
        for kw2 in keywords[i + 1 :]:
            if G.has_edge(kw1, kw2):
                G[kw1][kw2]["weight"] += 1
            else:
                G.add_edge(kw1, kw2, weight=1)

# Assign colors and opacities to nodes
node_colors = {}
for i, doc in enumerate(original_keywords):
    group_color = COLOR_GROUP1_5 if i < 5 else COLOR_GROUP6_10
    opacity = OPACITIES[i % 5]
    rgba_color = (*to_rgb(group_color), opacity)
    for kw, _ in doc:
        node_colors[kw] = rgba_color

# Extract and normalize edge weights
weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
norm = Normalize(vmin=weights.min(), vmax=weights.max())
edge_colors = [cm.viridis(norm(w)) for w in weights]
edge_widths = weights * EDGE_WIDTH_SCALE

# Position nodes
pos = nx.spring_layout(G, k=LAYOUT_K, seed=LAYOUT_SEED)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(15, 15))

# Draw nodes
node_sizes = [G.degree(node) * NODE_SIZE_SCALE for node in G.nodes()]
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color=[node_colors[node] for node in G.nodes()],
    edgecolors="black",
    linewidths=1,
    ax=ax,
)

# Draw edges
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=G.edges(),
    edge_color=edge_colors,
    width=edge_widths,
    alpha=0.6,
    ax=ax,
)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)

# Add colorbar for edge weights
sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label="Edge Weight (Co-occurrence)")

# Add title and show plot
ax.set_title("Network Graph of Keywords with Co-occurrence", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()
