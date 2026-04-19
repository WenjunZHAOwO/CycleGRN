
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import pandas as pd

def create_matrices_from_pandas(input_pd, truth_file, genes_file, case="upper"):
    """
    Build (score_submatrix, binary_truth, matrix, string_links_filtered) with
    case-insensitive gene matching.

    Parameters
    ----------
    input_pd : pd.DataFrame
        Predicted score matrix (rows/cols are gene identifiers or names).
    truth_file : str
        Path to STRING (or other) edge list with columns '#node1' and 'node2'.
    genes_file : str
        CSV with columns including 'id' and 'gene_short_name' for renaming.
    case : {'upper','lower'}
        Normalization to enforce for matching.

    Returns
    -------
    score_submatrix : pd.DataFrame
    binary_truth : pd.DataFrame
    matrix : pd.DataFrame
        Renamed + normalized full matrix.
    string_links_filtered : pd.DataFrame
        Truth links restricted to genes present in matrix.
    """

    def norm(s: str) -> str:
        if s is None:
            return s
        s = str(s)
        return s.upper() if case == "upper" else s.lower()

    # Load
    genes = pd.read_csv(genes_file, index_col=0)
    matrix = input_pd.copy()

    # Build mapping id -> gene_short_name, normalized
    map_dict = genes.set_index("id")["gene_short_name"].to_dict()
    map_dict = {k: norm(v) for k, v in map_dict.items()}

    # Rename rows/cols using mapping, then normalize anything unmapped too
    matrix.rename(index=map_dict, inplace=True)
    matrix.rename(columns=map_dict, inplace=True)

    # Normalize all remaining labels (unmapped IDs/names)
    matrix.index = matrix.index.map(norm)
    matrix.columns = matrix.columns.map(norm)

    # If mapping causes duplicate gene names, aggregate to avoid ambiguity
    # (common when multiple IDs map to same gene symbol)
    if matrix.index.duplicated().any():
        matrix = matrix.groupby(level=0).max()
    if matrix.columns.duplicated().any():
        matrix = matrix.T.groupby(level=0).max().T

    # Load truth links and normalize gene names
    string_links = pd.read_csv(truth_file, sep="\t")
    string_links["#node1"] = string_links["#node1"].map(norm)
    string_links["node2"] = string_links["node2"].map(norm)

    valid_genes = set(matrix.index)

    # Keep only links where both nodes are in the matrix
    string_links_filtered = string_links[
        string_links["#node1"].isin(valid_genes) & string_links["node2"].isin(valid_genes)
    ].copy()

    row_genes = string_links_filtered["#node1"].unique()
    col_genes = string_links_filtered["node2"].unique()

    # Extract submatrix of predicted scores
    score_submatrix = matrix.loc[row_genes, col_genes]

    # Normalize safely (avoid divide-by-zero)
    denom = score_submatrix.abs().to_numpy().max()
    if denom > 0:
        score_submatrix = score_submatrix / denom
    score_submatrix = score_submatrix.abs()

    # Ground-truth binary matrix
    binary_truth = pd.DataFrame(0, index=row_genes, columns=col_genes, dtype=int)
    for _, row in string_links_filtered.iterrows():
        binary_truth.loc[row["#node1"], row["node2"]] = 1

    return score_submatrix, binary_truth, matrix, string_links_filtered

##def create_matrices_from_pandas( input_pd, truth_file, genes_file ):
##
##
##    genes = pd.read_csv(genes_file,index_col = 0)
##    matrix = input_pd#pd.read_csv(input_file, index_col=0)
##    map_dict = genes.set_index("id")["gene_short_name"].to_dict()
##    matrix.rename(index=map_dict, inplace=True)
##    matrix.rename(columns=map_dict, inplace=True)
##    
##
##    string_links = pd.read_csv(truth_file,sep='\t')
##
##    valid_genes = set(matrix.index)
##
##    # Keep only links where both nodes are in the matrix
##    string_links_filtered = string_links[
##        string_links["#node1"].isin(valid_genes) & string_links["node2"].isin(valid_genes)
##    ].copy()
##
##
##    row_genes = string_links_filtered["#node1"].unique()
##    col_genes = string_links_filtered["node2"].unique()
##
##    # Step 2: Extract submatrix of predicted scores
##    score_submatrix = matrix.loc[row_genes, col_genes]  # shape: (#row_genes, #col_genes)
##    score_submatrix = score_submatrix / score_submatrix.abs().values.max()
##    score_submatrix = abs(score_submatrix)
##
##    # Step 3: Create ground-truth binary matrix of same shape
##    # Initialize to all zeros
##    binary_truth = pd.DataFrame(0, index=row_genes, columns=col_genes)
##
##    # Set entries to 1 for known links
##    for _, row in string_links_filtered.iterrows():
##        binary_truth.loc[row["#node1"], row["node2"]] = 1
##
##    return score_submatrix, binary_truth,matrix, string_links_filtered

def create_matrices( input_file, truth_file ):


    genes = pd.read_csv('../../mouseprogenitormtxs/10x_mouse_retina_development_feature.csv',index_col = 0)
    matrix = pd.read_csv(input_file, index_col=0)
    map_dict = genes.set_index("id")["gene_short_name"].to_dict()
    matrix.rename(index=map_dict, inplace=True)
    matrix.rename(columns=map_dict, inplace=True)
    

    string_links = pd.read_csv(truth_file,sep='\t')

    valid_genes = set(matrix.index)

    # Keep only links where both nodes are in the matrix
    string_links_filtered = string_links[
        string_links["#node1"].isin(valid_genes) & string_links["node2"].isin(valid_genes)
    ].copy()


    row_genes = string_links_filtered["#node1"].unique()
    col_genes = string_links_filtered["node2"].unique()

    # Step 2: Extract submatrix of predicted scores
    score_submatrix = matrix.loc[row_genes, col_genes]  # shape: (#row_genes, #col_genes)
    score_submatrix = score_submatrix / score_submatrix.abs().values.max()
    score_submatrix = abs(score_submatrix)

    # Step 3: Create ground-truth binary matrix of same shape
    # Initialize to all zeros
    binary_truth = pd.DataFrame(0, index=row_genes, columns=col_genes)

    # Set entries to 1 for known links
    for _, row in string_links_filtered.iterrows():
        binary_truth.loc[row["#node1"], row["node2"]] = 1

    return score_submatrix, binary_truth,matrix, string_links_filtered


def visualize_matrix_gene(score_submatrix, truth_list, center_gene = "Ccnb1", top_k = 10 ):
    """
    Visualize top-k highest scoring edges involving center_gene (in either direction).
    Validated edges from truth_list are highlighted in dark blue.
    """
    # Build ground truth lookup
    truth_edges = set(tuple(row) for row in truth_list[['#node1', 'node2']].values)

    # Collect all directed edges involving center_gene
    edges = []

    # Outgoing edges
    out_scores = score_submatrix.loc[center_gene].drop(center_gene, errors='ignore')
    for target, score in out_scores.items():
        edges.append((center_gene, target, score))

    # Incoming edges
    in_scores = score_submatrix.loc[:, center_gene].drop(center_gene, errors='ignore')
    for source, score in in_scores.items():
        edges.append((source, center_gene, score))

    # Select top-k edges by score
    top_edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_k]

    # Build directed graph
    G = nx.DiGraph()
    for u, v, w in top_edges:
        is_valid = (u, v) in truth_edges
        G.add_edge(u, v, weight=w, validated=is_valid)

    # Layout
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100)

    # Node colors
    node_colors = ['red' if n == center_gene else 'lightblue' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000)
    nx.draw_networkx_labels(G, pos, font_size=20)

    # Draw edges with weight-proportional thickness and validation color
    for u, v, d in G.edges(data=True):
        color = 'navy' if d['validated'] else 'lightblue'
        width = max(1.5, d['weight'] * 5)
        label = f"{d['weight']:.2f}"

        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            connectionstyle="arc3,rad=0.1",
            edge_color=color,
            width=width,
            arrows=True,
            arrowstyle='-|>',
            arrowsize=15,
            min_source_margin=25,   # shorten from source
            min_target_margin=25    # shorten from target
        )

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={(u, v): label},
            font_size=15
        )
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='navy', lw=3, label='Validated (in truth list)'),
        Line2D([0], [0], color='lightblue', lw=3, label='Predicted only')
    ]
    # plt.legend(handles=legend_elements, loc='upper center', fontsize=9, frameon=False)

    # plt.title(f"Top-{top_k} Interactions Involving {center_gene}")
    plt.axis('off')
    # plt.show()

        
  

    

def visualize_truth_gene(string_links_filtered, center_gene = "Ccnb1"):
    # Step 1: Build directed graph from edge list
    df_links = string_links_filtered.rename(columns={'#node1': 'source', 'node2': 'target'})
    G = nx.from_pandas_edgelist(df_links, source='source', target='target', create_using=nx.DiGraph())

    

    # Only keep edges where Ccnb1 is involved
    edges_ccnb1 = [
        (u, v) for (u, v) in G.edges()
        if u == center_gene or v == center_gene
    ]

    # Step 3: Build a new graph only from these edges
    subG = nx.DiGraph()
    subG.add_edges_from(edges_ccnb1)

    # Step 4: Plot
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(subG, seed=42, k=1.5, iterations=100)

    # Color nodes
    node_colors = [
        'red' if n == center_gene else 'lightblue'
        for n in subG.nodes()
    ]

    nx.draw(subG, pos,
            with_labels=True,
            width=2,
            node_color=node_colors,
            node_size=2000,
            edge_color='navy',
            arrows=True,
            font_size=15,
            connectionstyle="arc3,rad=0.1")

    plt.title(f"True Edges Involving {center_gene}")
    # plt.show()
