import IDEAlib.utils as utils
import matplotlib.pyplot as plt
import multiprocessing as mp
import pandas as pd
import networkx as nx
import os
import nltk
import time



"""
TODO:
1. minus graph function


"""

def ngramGraph(texts=None, text=None, gram_n=2, space_token='_space_', 
               max_norm=True, take_ratio=1.0, freq_threshold=3):
    """
    # input: list of tokenized text
    # output: `networkx` undirected graph

    # example: 

      texts = [['to', 'infinity', 'and', 'beyond'], 
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
    
    if text:
        print('\nVariable "text" will be remove in next version, please refer to Variable "texts"\n')
        texts = text

    # split text to gram
    n_gram_list = []

    for text in texts:
        n_gram_list += list(nltk.ngrams(text, n=gram_n))

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

    print('Graph constructed from ({}) sentences, there are ({}) nodes(tokens) in this graph.'.
          format(len(texts), len(ngramGraph)))
    return ngramGraph


def eigenvector_centrality(graph, show_time=False):
    """
    input: graph, output: dataframe
    """    
    st = time.time()
    dict_ec = nx.eigenvector_centrality_numpy(graph)
    if show_time:
        print('eigenvector centrality cost: {:.4f} sec'.format(time.time()-st))
    df_ec = utils.dict2df(dict_ec, key_col='token', val_col='value')
    return df_ec

def clustering_coefficitnt(graph, show_time=False, triangle_threshold=0):
    print('"clustering_coefficient()" is spelling error function name, please refer to "clustering_coefficient(graph, show_time=False, triangle_threshold=0)" in the future.\n')
    
    return clustering_coefficient(graph, show_time=show_time, triangle_threshold=triangle_threshold)

def clustering_coefficient(graph, show_time=False, triangle_threshold=0):
    """
    input: graph, output: dataframe
    """
    st = time.time()
    dict_cc = nx.clustering(graph)
    
    df_cc = utils.dict2df(dict_cc, key_col='token', val_col='value')
    
    if triangle_threshold > 0:
        df_tri = nx.triangles(graph)
        filtered_words = [k for k, v in dict_cc.items() if v >= triangle_threshold]
        df_cc = df_cc[~df_cc['token'].isin(filtered_words)]
        
    if show_time:
        print('clustering coefficient cost: {:.4f} sec'.format(time.time()-st))
        
    return df_cc

def measure_ec_cc(graph, show_time=False):
    """
    # input: `networkx` undirected graph
    # output: `2 pd.DataFrame of eigenvector_centrality & clustering coefficitnt`    
    """
    df_ec = eigenvector_centrality(graph, )
    df_cc = clustering_coefficitnt(graph, show_time)
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
    pos = nx.spring_layout(G)
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
    pos = nx.spring_layout(G)
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

