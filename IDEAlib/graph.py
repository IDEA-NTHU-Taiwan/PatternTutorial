import IDEAlib.utils as utils
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import os
import nltk
import time

"""
TODO:
1. minus graph function


"""

def ngramGraph(text, gram_n=2, space_token='_space_', 
               max_norm=True, take_ratio=1.0, freq_threshold=3):
    """
    # input: list of tokenized text
    # output: `networkx` undirected graph

    # example: 

      text = [['to', 'infinity', 'and', 'beyond'], 
                  ['sharon', 'is', 'not', 'white']]

      my_graph = ngramGraph(text)


    # ref: 
      https://networkx.github.io/documentation/stable/reference/classes/index.html
    ---
    # args:

    - text: input text, it should be `np.array` or `list` type.
    - gram_n: the distance between tokens you concerned.
    - space_token: the token to replace the space in text.
    - max_norm: using max-frequency to normalize (T/F).
    - take_ratio: the ratio of tokens to construct the graph. (order by freq)
    - freq_threshold: the freq threshold of tokens to construct the graph.

    """

    # split text to gram
    n_gram_list = []
    for test in text:
        n_gram_list += list(nltk.ngrams(test, n=gram_n))

    freq_dist = nltk.FreqDist(n_gram_list)

    # max_norm T/F
    if max_norm:
        max_value = freq_dist.most_common(1)[-1][-1]
    else:
        max_value = 1

    # construct graph
    ngramGraph = nx.Graph()
    weighted_edges_list = []
    for key, val in freq_dist.most_common(int(len(freq_dist)*take_ratio)):
        if val < freq_threshold:
            continue
        token_1, token_2 = key
        if len(token_1) == 0 or len(token_2) == 0:
            continue
        if token_1 == ' ':
            token_1 = space_token
        if token_2 == ' ':
            token_2 = space_token
        weighted_edges_list.append((token_1, token_2, val/max_value))
    ngramGraph.add_weighted_edges_from(weighted_edges_list)

    print('Graph constructed from ({}) texts, there are ({}) nodes in this graph.'.
          format(len(text), len(ngramGraph)))
    return ngramGraph


def measure_ec_cc(graph, show_time=False):
    """
    # input: `networkx` undirected graph

    # output: `2 pd.DataFrame of eigenvector_centrality & clustering coefficitnt`    
    """
    # dict_ec = nx.eigenvector_centrality(graph)
    st = time.time()
    dict_ec = nx.eigenvector_centrality_numpy(graph)
    if show_time:
        print('eigenvector centrality cost: ', time.time()-st)
    st = time.time()
    dict_cc = nx.clustering(graph)
    if show_time:
        print('clustering coefficitnt cost: ', time.time()-st)
    df_ec = utils.dict2df(dict_ec, key_col='token', val_col='value')
    df_cc = utils.dict2df(dict_cc, key_col='token', val_col='value')
    return df_ec, df_cc


def minus(G1, G2):
    print('not finish, return G1')
    return G1


def show(G, seed=9527):
    """
    ref: 
    https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_weighted_graph.html
    """
    plt.figure(figsize=(30,30))
    thr = 0.3
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > thr]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= thr]
    pos = nx.spring_layout(G, seed=seed)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=120, alpha=0.3)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1, edge_color='k')
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=0.6, alpha=0.5, 
                           edge_color='c', style='dashed')
    # labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    plt.axis('off')
    plt.show()


def show_cwsw(G, cw_list, sw_list, seed=9527):
    """
    ref: 
    https://networkx.github.io/documentation/stable/auto_examples/drawing/plot_weighted_graph.html
    """
    plt.figure(figsize=(30,30))
    thr = 0.3
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > thr]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= thr]
    # cw = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > thr]
    # sw = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= thr]
    pos = nx.spring_layout(G, seed=seed)
    # nodes
    norm = [node for node in G.nodes() if node not in sw_list and node not in cw_list]
    nx.draw_networkx_nodes(G, pos, node_size=80, alpha=0.3, nodelist=norm)
    nx.draw_networkx_nodes(G, pos, node_size=120, alpha=0.3, node_color='b', nodelist=sw_list)
    nx.draw_networkx_nodes(G, pos, node_size=120, alpha=0.3, node_color='r', nodelist=cw_list)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1, edge_color='k')
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=0.6, alpha=0.5, 
                           edge_color='c', style='dashed')
    # labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    plt.axis('off')
    plt.show()

