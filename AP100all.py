import  networkx as nx
import matplotlib.pyplot as pl
import numpy
import math
import pandas as pd
import community as community_louvain

pl.style.use('ggplot')

def flip_nodes_and_communities(dict_nodes_communities):
    # Step 1: initialize communities as keys
    new_dict = {}
    for k, v in dict_nodes_communities.items():
        new_dict[v] = []

    # Step 2: Fill in nodes
    for kk, vv in new_dict.items():
        for k, v in dict_nodes_communities.items():
            if dict_nodes_communities[
                k] == kk:  # If the community number (value) in `best` is the same as new_dict key (key), append the node (key) in `best`
                # print(k,v)
                new_dict[kk].append(k)

    return new_dict

def distanca(i, comm, communities):
    new_res = 0.0
    for k, v in communities.items():
        if v == comm:
             new_res = new_res + dist[i][k]

    return new_res

def convert(num):
    if num > 0.069:
        return 1
    else:
        return 0

def n(A, i):
    di = 0
    Ei =  0.0
    for j in range(100):
        di = di + A[i][j]
    #Ei += di*di + di

    for j in range(100):
        if j != i:
            for k in range(100):
                if k != i:
                    Ei += A[i][j]*A[j][k]

    return Ei/di


dist = numpy.zeros((100,100))


with open('coords100.txt', 'r') as f:
    coords = [[float(num) for num in line.split(' ')] for line in f]
    for i in range(100):
        for j in range(100):
            x1 = coords[i][0]
            y1 = coords[i][1]
            x2 = coords[j][0]
            y2 = coords[j][1]
            dist[i][j] =  math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

avg_d = 0.0
for i in range(100):
    for j in range(100):
        if i != j:
            avg_d += dist[i][j]

G1 = nx.from_numpy_matrix(dist)

def commmunityBasedCentrality(g, communities):
    # communities: dictionary of node: community number

    # Get the size of each community
    communities_flipped = flip_nodes_and_communities(communities)  # call outer function
    community_size_dict = {}
    for i in communities_flipped:
        community_size_dict[i] = len(communities_flipped[i])

    # Get total nodes
    total_nodes = len(g)

    # Keep track of the link type a.k.a to which community the node externally links to
    list = g.nodes()
    dict_graph = dict()  # nodes in the key and their neighbors
    for i in list:
        dict_graph[i] = []
    for i in list:
        iteri = g.neighbors(i)
        for j in iteri:
            dict_graph[i].append(j)

    dict_external_communities_and_sizes = {}  # node: tup(external community, size)

    for i in communities:  # for each node
        community = communities[i]  # get its community and put it in a variable
        dict_external_communities_and_sizes[i] = []
        for j in dict_graph[i]:  # get neighbors of node i
            if communities[j] != community:  # check if the communities of the neighbors are not the same as node i
                tup = ()
                tup = (communities[j], community_size_dict[communities[j]],
                       j)  # external community, its size, the neighbor of node i in that external community
                dict_external_communities_and_sizes[i].append(tup)

    # Get external CBC for each node
    dict_cbc_external = {}
    for index1 in dict_external_communities_and_sizes:  # for each node
        dict_cbc_external[index1] = 0
        for index2 in dict_external_communities_and_sizes[
            index1]:  # index2 contains the tuple of each node, we can now access it
            community_size = index2[1]
            temp = distanca(index1, index2[0], communities)  #(community_size * 1) / total_nodes
            dict_cbc_external[index1] = dict_cbc_external[index1] + temp
        dict_cbc_external[index1] = 1.0 / (dict_cbc_external[index1]+1)

    dict_internal_communities_and_sizes = {}  # node: tup(internal community, size)

    for i in communities:  # for each node
        community = communities[i]  # get its community and put it in a variable
        dict_internal_communities_and_sizes[i] = []
        for j in dict_graph[i]:  # get neighbors of node i
            if communities[j] == community:  # check if the communities of the neighbors are the SAME as node i
                tup = ()
                tup = (communities[j], community_size_dict[communities[j]],
                       j)  # internal community, its size, the neighbor of node i in that external community
                dict_internal_communities_and_sizes[i].append(tup)

    # Get internal CBC for each node
    dict_cbc_internal = {}
    for index1 in dict_internal_communities_and_sizes:  # for each node
        dict_cbc_internal[index1] = 0
        for index2 in dict_internal_communities_and_sizes[
            index1]:  # index2 contains the tuple of each node, we can now access it
            # print(index2)
            # print(index2[0]) # Community number
            # print(index2[1]) # Community size
            # print(index2[2]) # Node in that community
            community_size = index2[1]
            temp = distanca(index1, index2[0], communities)#(community_size * 1) / total_nodes
            dict_cbc_internal[index1] = dict_cbc_internal[index1] + temp
        dict_cbc_internal[index1] = 1.0 / dict_cbc_internal[index1]

    # Add up CBC internal and CBC external
    dict_cbc_final = {}
    for i in dict_cbc_internal:
        a = dict_cbc_internal[i]
        b = dict_cbc_external[i]
        dict_cbc_final[i] = math.sqrt(a*a + b*b)

    return dict_cbc_final

max = 0.0
node = -1
closeness = numpy.zeros(100)
avg_b = 0.0
for i in range(100):
    s = 0.0
    for j in range(100):
        s += dist[i][j]
    v = 99/s
    closeness[i] = v
    if v > max:
        max = v
        node = i

def allocate(hubs, inds):
    for i in range(100):
        j_a = inds[0]
        min_d = dist[i][inds[0]]
        for j in range(len(inds)):
            for k in range(len(inds)):
                if min_d > dist[i][inds[j]]:
                    j_a = inds[j]
                    min_d = dist[i][inds[j]]
        hubs[i] = j_a


def checkIfIn(inds, i):
    s = 0
    for j in range(len(inds)):
        if dist[i][j] < avg_b:
            s = s + 1
    if s == 0:
        print(i)
    return s > 0

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return numpy.allclose(a, a.T, rtol=rtol, atol=atol)

with open('APS100.txt', 'r') as f:
    l = [[convert(float(num)) for num in line.split(' ')] for line in f]
    r = numpy.matrix(l)


    best = 0.0
    imax = 0
    dict_CM =  {}
    for i in range(100):
        dict_CM[i] = closeness[i]*n(l, i)

    G = nx.from_numpy_matrix(r)
    partition = community_louvain.best_partition(G)
    ranks = commmunityBasedCentrality(G, partition)

    dict_katz = nx.katz_centrality_numpy(G, alpha=0.009)
    # print(dict_katz)

    dict_degree = nx.degree_centrality(G)
    dict_betweenes = nx.betweenness_centrality(G)
    dict_closenes = nx.closeness_centrality(G)
    dict_pagerank = nx.pagerank(G, alpha=0.45)
    dict_eigen = nx.eigenvector_centrality(G)
    dict_close = nx.current_flow_closeness_centrality(G)
    dict_info = nx.information_centrality(G)
    dict_sub = nx.subgraph_centrality(G)

    degree_cent = []
    for i in dict_degree.values():
        degree_cent.append(i)

    CM_cent = []
    for i in dict_CM.values():
        CM_cent.append(i)

    betweenes_cent = []
    for i in dict_betweenes.values():
        betweenes_cent.append(i)

    closenes_cent = []
    for i in dict_closenes.values():
        closenes_cent.append(i)

    pagerank_cent = []
    for i in dict_pagerank.values():
        pagerank_cent.append(i)

    katz_cent = []
    for i in dict_katz.values():
        katz_cent.append(i)

    eigen_cent = []
    for i in dict_eigen.values():
        eigen_cent.append(i)

    modular_cent = []
    for i in ranks.values():
        modular_cent.append(i)

    cfcc_cent = []
    for i in dict_close.values():
        cfcc_cent.append(i)

    sub_cent = []
    for i in dict_sub.values():
        sub_cent.append(i)

    info_cent = []
    for i in dict_info.values():
        info_cent.append(i)

    a = pd.Series(degree_cent)
    b = pd.Series(CM_cent)
    c = pd.Series(betweenes_cent)
    d = pd.Series(closenes_cent)
    e = pd.Series(pagerank_cent)
    f = pd.Series(eigen_cent)
    g = pd.Series(modular_cent)
    h = pd.Series(cfcc_cent)
    i = pd.Series(katz_cent)
    j = pd.Series(sub_cent)
    k = pd.Series(info_cent)

    abcde = pd.DataFrame({'DC': a, 'CM': b, 'BC': c,
                          'CC': d, 'PRC': e, 'EC': f, 'Mod': g, 'CFCC': h,
                          'Katz': i, 'SC': j, 'IC': k})

    corr_matrix = abcde.corr(method='pearson').to_numpy()

    fig, ax = pl.subplots()
    im = ax.imshow(corr_matrix)
    im.set_clim(-0.007, 1)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1, 2, 3, 4, 5, 6, 7,8,9,10), ticklabels=('DC', 'CM', 'BC',
                                                          'CC', 'PRC', 'EC', 'Mod', 'CFCC', 'Katz', 'SC', 'IC'))
    ax.yaxis.set(ticks=(0, 1, 2, 3, 4, 5, 6, 7,8, 9, 10),
                 ticklabels=('DC', 'CM', 'BC',
                          'CC', 'PRC', 'EC', 'Mod','CFCC', 'Katz', 'SC', 'IC'))
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix[1])):
            ax.text(j, i, corr_matrix[i, j].round(decimals=2), ha='center', va='center',
                    color='w')
    pl.title("Pearson's Correlation AP100")
    pl.show()

print(nx.density(G))
print(nx.transitivity(G))
print(nx.degree_assortativity_coefficient(G))
print(nx.info(G))


nx.draw(G)
pl.show()