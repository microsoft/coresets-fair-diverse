import itertools
import random
import copy
import numpy as np
import networkx as nx
import math
from sklearn.metrics.pairwise import euclidean_distances


def reverse_bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted in descending order.

    The return value i is such that all e in a[:i] have e >= x, and all e in
    a[i:] have e < x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Essentially, the function returns number of elements in a which are >= than x.
    >>> a = [8, 6, 5, 4, 2]
    >>> reverse_bisect_right(a, 5)
    3
    >>> a[:reverse_bisect_right(a, 5)]
    [8, 6, 5]
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def separate_color(data, color):
    """Return the index where to insert item x in list a, assuming a is sorted in descending order.
    Args:
        data: numpy with embedded messages
        color: numpy with color value of the message
    Returns:
        data_col (list): reduce numpy 2D matrix with only messages from the color
        index_col (list): indexes of the messages of the color
    """
    unique_colors = list(set(list(color)))
    data_col = []
    index_col = []
    for col in unique_colors:
        col_index = list(np.where(color == col)[0])
        data_col.append(data[col_index][:])
        index_col.append(col_index)
    return data_col, index_col


def div_score_min_pairwise(euclid_distances, sub_set):
    """Return the diversity score for a subset in the notion of Min-Pairwise distances.
    Args:
        euclid_distances: euclidean distance matrix of the data entries
        sub_set: set of elements the diversity score is computed
    Returns:
        div_score (float): the diversity score expressed in the notion of Min-Pairwise (sum of nearest neighbors)
    """
    dist_mod = euclid_distances[np.ix_(sub_set, sub_set)]
    dist_mod[dist_mod == 0] = np.Inf
    div_score = np.min(dist_mod)
    return div_score


def div_score_sum_pairwise(euclid_distances, sub_set):
    """Return the diversity score for a subset in the notion of Sum-Pairwise distances.
    Args:
        euclid_distances: euclidean distance matrix of the data entries
        sub_set: set of elements the diversity score is computed
    Returns:
        div_score (float): the diversity score expressed in the notion of Sum-Pairwise
    """
    dist_mod = euclid_distances[np.ix_(sub_set, sub_set)]
    div_score = np.sum(dist_mod) / 2.0
    return div_score


def div_score_sum_nn(euclid_distances, sub_set):
    """Return the diversity score for a subset in the notion of Sum-NN distances.
    Args:
        euclid_distances: euclidean distance matrix of the data entries
        sub_set: set of elements the diversity score is computed
    Returns:
        div_score (float): the diversity score expressed in the notion of Sum-NN (sum of nearest neighbors)
    """
    dist_mod = euclid_distances[np.ix_(sub_set, sub_set)]
    dist_mod[dist_mod == 0] = np.Inf
    div_score = sum(np.min(dist_mod, axis=0))
    return div_score


def dm_min_pairwise(euclid_distances, k):
    '''
    Diverse Maximization (DM) for Min-Pairwise

    Given a colored point set $P=P_1\cup\cdots\cup P_m$, and $k=k_1+\cdots+k_m$\\
    Return a subset $S=\{p_1,\cdots,p_k\}\subseteq P$\\

    Args:
        euclid_distances: euclidean distance matrix of the data entries
        k (int): the number of elements to be returned
    Returns:
        S (set): a subset of elements that is the most optimal
        dist_S (int): an optimal value of the metric
    '''
    n = euclid_distances.shape[0]
    q = np.unravel_index(np.argmax(euclid_distances, axis=None), euclid_distances.shape)
    S = [q[0]]
    dist_S = []
    P_minus_S = set(range(n)).difference(S)
    for i in range(k - 1):
        list_P_minus_S = list(P_minus_S)
        list_S = list(S)
        if len(list_S) == 0 or len(list_P_minus_S) == 0: break
        dist_mod = euclid_distances[np.ix_(list_P_minus_S, list_S)]

        ind_min_q_per_p = np.argmin(dist_mod, axis=1)
        min_q_per_p = np.min(dist_mod, axis=1)
        if len(min_q_per_p) == 0: continue
        max_p = np.argmax(min_q_per_p)
        dist_max_p = np.max(min_q_per_p)
        max_p = list_P_minus_S[max_p]

        S.append(max_p)
        P_minus_S.remove(max_p)
        dist_S.append(dist_max_p)

    return S, dist_S


def cs_construction_min_pairwise(data, colors, k):
    '''
    Core-set Construction (partitions based on colors) for Min-Pairwise distance

    Given a colored point set $P=P_1\cup\cdots\cup P_m$, and $k=k_1+\cdots+k_m$
    Output subsets $S_i\subseteq P_i$, s.t. $\opt(S)\approx \opt(P)$ where $S=\bigcup_i S_i$\\

    Args:
        data: numpy with embedded messages
        colors: numpy with color value of the message
        k (list): list of desired number of items k_i's per color
    Returns:
        coresets (set): a subset of elements that is the most optimal (coresets)
    '''
    data_col, index_col = separate_color(data, colors)

    coresets = []
    sum_k = sum(k)
    for ind_col in range(len(data_col)):
        A_i = data_col[ind_col]
        ind_A_i = index_col[ind_col]
        eucld_dist_i = euclidean_distances(A_i)
        S_i, _ = dm_min_pairwise(eucld_dist_i, sum_k)

        coreset_i = set([ind_A_i[elem] for elem in S_i])
        coresets.append(coreset_i)

    return coresets


def fdm_min_pairwise(data, colors, k, gamma):
    '''
    Fair Diversity Maximization for Min-Pairwise distance: Flow-Based for minimum pairwise distance

    Given a colored point set $P=P_1\cup\cdots\cup P_m$, and $k=k_1+\cdots+k_m$, threshold $\gamma$\\
    Output subsets $S_i\subseteq P_i$, s.t. $|S_i|=k_i$ and s.t. $\div(S=\bigcup_i S_i)$ is approximately maximized

    Args:
        data: numpy with embedded messages
        colors: numpy with color value of the message
        k (list): list of desired number of items k_i's per color
    Returns:
        S (set): a list of sets of elements per color that is the most optimal
    '''
    sum_k = sum(k)
    eucld_dist = euclidean_distances(data)  # separate data per color
    data_col, index_col = separate_color(data, colors)
    unique_colors = list(set(list(colors)))
    m = len(data_col)
    d_1 = m * gamma / (3.0 * m - 1)
    d_2 = gamma / (3.0 * m - 1)

    Z = []
    for ind_col in range(m):
        A_i = data_col[ind_col]
        ind_A_i = index_col[ind_col]
        eucld_dist_i = eucld_dist[np.ix_(ind_A_i, ind_A_i)]
        S_i, dist_S_i = dm_min_pairwise(eucld_dist_i, sum_k)
        Y_i = [ind_A_i[elem] for elem in S_i]
        Z_i = Y_i[:reverse_bisect_right(dist_S_i,
                                        d_1)]  # maximal prefix of $Y_i$ s.t. all points in $Z_i$ are $>= d_1$ apart
        Z.append(Z_i)

    # Construct the graph $G_Z$ where the nodes are $Z=\bigcup_i Z_i$ and
    # edges $(z_1,z_2)$ if $\dist(z_1,z_2) < d_2 $
    flat_Z = list(itertools.chain(*Z))
    G_Z = nx.Graph()

    for nd in flat_Z:
        G_Z.add_node(nd)

    for potential_edge in itertools.combinations(flat_Z, 2):
        if eucld_dist[potential_edge[0]][potential_edge[1]] < d_2:
            G_Z.add_edge(potential_edge[0], potential_edge[1])

    # Coalculate the connected components
    C_components = [set(G_Z.subgraph(c).nodes) for c in nx.connected_components(G_Z)]
    t = len(C_components)

    # Construct the auxiliary graph $G=(V,E)$ where $V=\{a,u_1,\cdots,u_m,v_1,\cdots,v_t,b\}$ and
    # $E=\{(a,u_i)$ with capacity $k_i\}\cup \{(v_j,b)$ with capacity $1\} \cup\{(u_i,v_j)$
    # with capacity $1:|Z_i\cap C_j|\geq 1\}$
    G = nx.DiGraph()
    for id_node in range(m):
        G.add_node("u_{}".format(id_node + 1))
    for jd_node in range(t):
        G.add_node("v_{}".format(jd_node + 1))

    for id_node in range(m):
        G.add_edge("a", "u_{}".format(id_node + 1), capacity=k[id_node])

    for jd_node in range(t):
        G.add_edge("v_{}".format(jd_node + 1), "b", capacity=1)

    for id_node in range(m):
        for jd_node in range(t):
            if len(set(Z[id_node]).intersection(C_components[jd_node])) >= 1:
                G.add_edge("u_{}".format(id_node + 1), "v_{}".format(jd_node + 1), capacity=1)

    # Run max-flow algorithm a-b
    flow_size, flow_dict = nx.maximum_flow(G, "a", "b")

    if flow_size < sum_k:
        print("Optimal core-set doesn't exist")
        return None
    else:
        # $\forall (u_i,v_j)$ with a flow, add a node in $C_j$ with color $i$ to $S_i$
        S = [set() for col in range(m)]
        for id_node in range(m):
            S_i = set()
            for jd_node in range(t):
                if G.has_edge("u_{}".format(id_node + 1), "v_{}".format(jd_node + 1)):
                    if flow_dict["u_{}".format(id_node + 1)]["v_{}".format(jd_node + 1)] > 0:
                        elem = C_components[jd_node].pop()
                        S[colors[elem]].add(elem)

    return S


def dm_sum_pairwise(euclid_distances, k, eps=1e-5):
    '''
    Diverse Maximization (DM) for Sum-Pairwise

    Given a colored point set $P=P_1\cup\cdots\cup P_m$, and $k=k_1+\cdots+k_m$\\
    Return a subset $S=\{p_1,\cdots,p_k\}\subseteq P$\\

    Args:
        euclid_distances: euclidean distance matrix of the data entries
        k (int): the number of elements to be returned
        eps (float): approximation small value
    Returns:
        S (set): a subset of elements that is the most optimal
    '''
    n = euclid_distances.shape[0]
    q = np.unravel_index(np.argmax(euclid_distances, axis=None), euclid_distances.shape)
    S = {q[0], q[1]}  # initialize S with two ellements of maximum size
    dist_S = [euclid_distances[q[0]][q[1]]]
    P_minus_S = set(range(n)).difference(S)

    for i in range(k - 2):  # initialize coreset such that color condition is satisfied
        d = -1  # initial value of the metric
        ind = None
        for j in P_minus_S:  # checking all candidate points
            list_S = list(S)  # try if p <-> q swap in S_i will lead to an improvement
            list_S.append(j)
            dist_mod = euclid_distances[np.ix_(list_S, list_S)]
            metric = np.sum(dist_mod) / 2.0
            if metric > d:  # if j is the best candidate so far
                d = metric
                ind = j  # ind contains the optimal index

        if ind is not None:
            S.add(ind)
            P_minus_S.remove(ind)

    improved = 1  # shows if we have made any progress in each iteration
    while improved:
        improved = 0
        list_S = list(S)
        dist_mod = euclid_distances[np.ix_(list_S, list_S)]
        metric_S = np.sum(dist_mod) / 2.0
        prod_change = itertools.product(S, P_minus_S)
        for item in prod_change:
            item_q, item_p = item[0], item[1]
            list_S = list(S)
            list_S.append(item_p)
            list_S.remove(item_q)
            dist_mod = euclid_distances[np.ix_(list_S, list_S)]
            metric = np.sum(dist_mod) / 2.0
            if metric > (1 + eps) * metric_S:  # improvment achived, exchange p <-> q elements in S_i
                S.add(item_p)
                S.remove(item_q)
                P_minus_S.add(item_q)
                P_minus_S.remove(item_p)
                improved = 1
                break

    return S


def cs_construction_sum_pairwise(data, colors, k):
    '''
    Core-set Construction (partitions based on colors) for Sum-Pairwise distance

    Given a colored point set $P=P_1\cup\cdots\cup P_m$, and $k=k_1+\cdots+k_m$
    Output subsets $S_i\subseteq P_i$, s.t. $\opt(S)\approx \opt(P)$ where $S=\bigcup_i S_i$\\

    Args:
        data: numpy with embedded messages
        colors: numpy with color value of the message
        k (list): list of desired number of items k_i's per color
    Returns:
        S (list): a list of sets of elements that is the most optimal (corsets)
    '''
    S = []
    data_col, index_col = separate_color(data, colors)
    unique_colors = list(set(list(colors)))
    sum_k = sum(k)

    coresets = []
    for ind_col in range(len(data_col)):
        A_i = data_col[ind_col]
        ind_A_i = index_col[ind_col]
        eucld_dist = euclidean_distances(A_i)
        n = eucld_dist.shape[0]
        list_P = set(list(range(n)))

        q = np.unravel_index(np.argmax(eucld_dist, axis=None), eucld_dist.shape)
        S = [q[0], q[1]]
        P_minus_S = set(range(n)).difference(S)
        for i in range(sum_k):  # (k[ind_col]-2):
            list_P_minus_S = list(P_minus_S)
            list_S = list(S)
            dist_mod = eucld_dist[np.ix_(list_P_minus_S, list_S)]

            ind_min_q_per_p = np.argmin(dist_mod, axis=1)
            min_q_per_p = np.min(dist_mod, axis=1)
            max_p = np.argmax(min_q_per_p)
            dist_max_p = np.max(min_q_per_p)
            max_p = list_P_minus_S[max_p]

            S.append(max_p)
            P_minus_S.remove(max_p)

        list_S = list(S)
        r = np.min(eucld_dist[np.ix_(list_S, list_S)])
        T = set()
        for p in S:
            for j in range(sum_k):  # (k[ind_col]):
                P_minus_T = set(list_P).difference(T)
                for el_pi in list(P_minus_T):
                    dist_mod = eucld_dist[np.ix_([el_pi], list_S)]
                    # dist_mod[dist_mod==0]=np.Inf
                    ind_min_qi_per_pi = np.argmin(dist_mod, axis=1)
                    for pot_pi in ind_min_qi_per_pi:
                        if pot_pi == p:
                            T.add(el_pi)

        S = set(S).union(T)
        S = {ind_A_i[elem] for elem in S}
        coresets.append(S)

    return coresets


def fdm_sum_pairwise(data, colors, k, eps=1e-5):
    '''
    Fair Diversity Maximization for Sum-Pairwise distance

    Given a colored point set $P=P_1\cup\cdots\cup P_m$, and $k=k_1+\cdots+k_m$, threshold $\gamma$\\
    Output subsets $S_i\subseteq P_i$, s.t. $|S_i|=k_i$ and s.t. $\div(S=\bigcup_i S_i)$ is approximately maximized

    Args:
        data: numpy with embedded messages
        colors: numpy with color value of the message
        k (list): list of desired number of items k_i's per color
        eps (float): approximation small value
    Returns:
        S (set): a list of sets of elements per color that is the most optimal (corsets)
    '''
    eucld_dist = euclidean_distances(data)  # calculate euclidean distance at the beginning
    data_col, index_col = separate_color(data, colors)  # separate data per color
    unique_colors = list(set(list(colors)))

    S = []  # initialize coreset such that color condition is satisfied
    for ind_col in range(len(data_col)):
        A_i = data_col[ind_col]
        ind_A_i = index_col[ind_col]
        n_i = A_i.shape[0]
        eucld_dist_i = euclidean_distances(A_i)
        S_i = dm_sum_pairwise(eucld_dist_i, k[ind_col])  # run optimal opt_sumpairwise for a single color
        S_i = set([ind_A_i[elem] for elem in S_i])
        S.append(S_i)

    # complementary set for replacement
    P_minus_S = [set(index_col[col]).difference(S[col]) for col in unique_colors]
    copy_S_flat = [list(elem) for elem in S]
    copy_S_flat = list(itertools.chain.from_iterable(copy_S_flat))
    dist_mod = eucld_dist[np.ix_(copy_S_flat, copy_S_flat)]
    metric_S = np.sum(dist_mod) / 2.0

    improved = 1  # shows if we have made any progress in each iteration
    while improved:
        improved = 0
        prod_change = [itertools.product(S[col], P_minus_S[col]) for col in unique_colors]

        for ind_col, prod_change_per_color in enumerate(prod_change):
            col = unique_colors[ind_col]
            for swap_cand in prod_change_per_color:
                copy_S = S.copy()  # try if p <-> q swap in S_i will lead to an improvement
                S_i = copy_S[col].copy()
                S_i.remove(swap_cand[0])
                S_i.add(swap_cand[1])
                copy_S[col] = S_i

                copy_S_flat = [list(elem) for elem in copy_S]
                copy_S_flat = list(itertools.chain.from_iterable(copy_S_flat))

                dist_mod = eucld_dist[np.ix_(copy_S_flat, copy_S_flat)]
                metric = np.sum(dist_mod) / 2.0

                if metric > (1 + eps) * metric_S:  # improvment achived, exchange p <-> q elements in S_i
                    S = copy_S
                    P_minus_S[col].remove(swap_cand[1])
                    P_minus_S[col].add(swap_cand[0])
                    improved = 1
                    metric_S = metric
                    break

    return S


def dm_sum_nn(euclid_distances, k, seed=47):
    '''
    Diverse Maximization (DM) for Sum-NN notion of diversity

    Given a colored point set $P=P_1\cup\cdots\cup P_m$, and $k=k_1+\cdots+k_m$\\
    Return a subset $S=\{p_1,\cdots,p_k\}\subseteq P$\\

    Args:
        euclid_distances: euclidean distance matrix of the data entries
        k (int): the number of elements to be returned
        seed (int): random seed
    Returns:
        S (set): a subset of elements that is the most optimal

    Optimization for sum of minimum distances (Algorithm 10)

    Input: a point set $P$, and $k$
    Output: a subset $S=\{p_1,\cdots,p_k\}\subseteq P$\\ such that the metric is approximately maximized
    '''
    n = euclid_distances.shape[0]
    p = np.unravel_index(np.argmax(euclid_distances, axis=None), euclid_distances.shape)
    S = [p[0], p[1]]
    R = [euclid_distances[p[0]][p[1]]]
    R_multiple_i = [euclid_distances[p[0]][p[1]]]
    P = set(range(n))
    P_minus_S = P.difference(S)
    for i in range(k - 2):
        list_P_minus_S = list(P_minus_S)
        list_S = list(S)
        dist_mod = euclid_distances[np.ix_(list_P_minus_S, list_S)]

        ind_min_q_per_p = np.argmin(dist_mod, axis=1)
        min_q_per_p = np.min(dist_mod, axis=1)
        max_p = np.argmax(min_q_per_p)
        dist_max_p = np.max(min_q_per_p)
        p_i = list_P_minus_S[max_p]

        dist_mod_r = euclid_distances[np.ix_([p_i], list_S)]
        r_i = np.min(dist_mod_r)

        S.append(p_i)
        R.append(r_i)
        R_multiple_i.append((i + 2) * r_i)

    q = np.argmax(R_multiple_i[:(k + 1)])
    B = [set() for ind in range(q)]
    card_B = list(np.zeros(q))

    for p in P:
        for i in range(q):
            if euclid_distances[S[i]][p] < R[q] / 2:
                B[i].add(p)
                card_B[i] = card_B[i] + 1

    z = math.floor((q + 1) / 2)
    if len(card_B) == 1 and z == 1:
        idx_sparesest = np.array([0])
    else:
        idx_sparesest = np.argpartition(card_B, z)[:z]

    coreset = set([S[elem] for elem in idx_sparesest])

    Union_Bz = set()
    for elem in idx_sparesest:
        Union_Bz = Union_Bz.union(B[elem])

    P_diff_unZ = P.difference(Union_Bz)

    random.seed(seed)
    coreset = coreset.union(set(random.sample(P_diff_unZ, k - z)))

    return coreset


def cs_construction_sum_nn(data, colors, k):
    '''
    Core-set Construction (partitions based on colors) for Sum-NN distance

    Given a colored point set $P=P_1\cup\cdots\cup P_m$, and $k=k_1+\cdots+k_m$
    Output subsets $S_i\subseteq P_i$, s.t. $\opt(S)\approx \opt(P)$ where $S=\bigcup_i S_i$\\

    Args:
        data: numpy with embedded messages
        colors: numpy with color value of the message
        k (list): list of desired number of items k_i's per color
    Returns:
        S (list): a list of sets of elements that is the most optimal (corsets)
    '''
    S = []
    data_col, index_col = separate_color(data, colors)
    unique_colors = list(set(list(colors)))
    sum_k = np.sum(k)

    coresets = []
    for ind_col in range(len(data_col)):
        A_i = data_col[ind_col]
        ind_A_i = index_col[ind_col]
        eucld_dist = euclidean_distances(A_i)
        n = eucld_dist.shape[0]
        k_i = k[ind_col]
        S = set()
        P_minus_G = set(range(n))

        for j in range(k_i):
            G = [0]
            P_minus_G = P_minus_G.difference(set(G))
            for l in range(sum_k):
                list_P_minus_G = list(P_minus_G)
                list_G = list(G)
                if len(list_P_minus_G) == 0: break
                dist_mod = eucld_dist[np.ix_(list_P_minus_G, list_G)]

                ind_min_q_per_p = np.argmin(dist_mod, axis=1)
                min_q_per_p = np.min(dist_mod, axis=1)
                max_p = np.argmax(min_q_per_p)
                p_l = list_P_minus_G[max_p]
                G.append(p_l)
                P_minus_G.remove(p_l)

                S = S.union(G)
                P_minus_G = P_minus_G.difference(G)


        coreset_i = set([ind_A_i[elem] for elem in S])
        coresets.append(coreset_i)

    return coresets


def best_subset_bals(B, data_col, index_col, i, k, color=None, centersB=None):
    '''

    Given a set of at most $k$ disjoint balls $B$ with centers
    $c_1,\cdots,c_t$ and radius $r$, $P_1,\cdots,P_m$, and $k=k_1+\cdots+k_m$, and a special index $i$\\
    Output an (approximately) largest subset of balls
    $B'\subseteq B$ such that $P\setminus B'$ contains at least $k_i-\normX{B'}$ points from color $i$,
    and at least $k_{\ell}$ points from color $\ell$ for each color $\ell\neq i$.\\

    Args:
        B: disjoint set of points
        data_col: a list of data numpy 2D arrays per color
        i (int): a concerned color
        k (list): list of desired number of items k_i's per color
        centersB: a list of centers of B
    Returns:
        S (list): a list of sets of elements that is the most optimal (corsets)

    '''
    d = []
    B_flat = list(itertools.chain.from_iterable(B))
    Pprime = []
    for l in range(len(data_col)):
        d.append(k[l])
        Pprime.append(index_col[l])

    for l in range(len(data_col)):
        a = min(d[l], len(set(Pprime[l]).difference(set(B_flat))))
        d[l] = d[l] - a
        Pprime.append(index_col[l])
        for _ in range(a):
            Pprime[l].pop()

    SOL1 = []
    fin_ind_b = -1
    for ind_b, b in enumerate(B):
        B_diff_b = set(B_flat).difference(set(b))
        if len(B_diff_b.difference(set(index_col[i]))) < d[i] - 1: continue

        take_B = True
        for l in range(len(data_col)):
            if l == i: continue

            if len(B_diff_b.difference(set(index_col[l]))) < d[l] - 1:
                take_B = False
                break

        if take_B:
            SOL1 = [b]
            fin_ind_b = ind_b
            break

    SOL2 = []
    fin_ind_b1, fin_ind_b2 = -1, -1
    for ind_b1, b1 in enumerate(B):
        for ind_b2, b2 in enumerate(B):
            if ind_b1 == ind_b2: continue
            B_diff_b = set(B_flat).difference(set(b1))
            B_diff_b = set(B_diff_b).difference(set(b2))
            if len(B_diff_b.difference(set(index_col[i]))) < d[i] - 1: continue

            take_B = True
            for l in range(len(data_col)):
                if l == i: continue

                if len(B_diff_b.difference(set(index_col[l]))) < d[l] - 1:
                    take_B = False
                    break

            if take_B:
                SOL2 = [b1, b2]
                fin_ind_b1, fin_ind_b2 = ind_b1, ind_b2
                break

    indx_l = []
    SOLM = copy.deepcopy(B)
    centersSOLM = copy.deepcopy(centersB)
    for l in range(len(data_col)):
        if l == i: continue
        if d[l] > 0:
            lrg_b, id_lrg_b = -1, -1
            for ind_b, b in enumerate(SOLM):
                nr_l = len(set(b).intersection(set(index_col[l])))
                if nr_l > lrg_b:
                    lrg_b = nr_l
                    id_lrg_b = ind_b
            if id_lrg_b != -1 and B[id_lrg_b] in SOLM:
                indx_l.append(id_lrg_b)
                SOLM.pop(id_lrg_b)
                centersSOLM.pop(id_lrg_b)

            for pp in B[id_lrg_b]:
                if d[color[pp]] > 0:
                    d[color[pp]] = d[color[pp]] - 1

    SOLM_flat = list(itertools.chain.from_iterable(SOLM))
    len_SOLM = len(SOLM_flat)
    all_contibutions_zero = True
    for l in range(len(data_col)):
        if l == i: continue
        if d[l] != 0:
            all_contibutions_zero = False
            break

    if len_SOLM > 2 and d[i] < len_SOLM and all_contibutions_zero:
        return SOLM, centersSOLM
    elif SOL2 != []:
        return SOL2, [fin_ind_b1, fin_ind_b2]
    elif SOL1 != []:
        return SOL1, [fin_ind_b]
    else:
        return [], []


def fdm_sum_nn(data, colors, k):
    '''
    Fair Diversity Maximization for Sum-NN distance

    Given a colored point set $P=P_1\cup\cdots\cup P_m$, and $k=k_1+\cdots+k_m$, threshold $\gamma$\\
    Output subsets $S_i\subseteq P_i$, s.t. $|S_i|=k_i$ and s.t. $\div(S=\bigcup_i S_i)$ is approximately maximized

    Args:
        data: numpy with embedded messages
        colors: numpy with color value of the message
        k (list): list of desired number of items k_i's per color
        eps (float): approximation small value
    Returns:
        S (set): a list of sets of elements per color that is the most optimal (corsets)
    '''
    eucld_dist = euclidean_distances(data)  # calculate euclidean distance at the beginning
    data_col, index_col = separate_color(data, colors)  # separate data per color
    unique_colors = list(set(list(colors)))
    sum_k = sum(k)
    coresets = cs_construction_sum_nn(data, colors, k)

    S = []  # initialize coreset such that color condition is satisfied
    for ind_col in range(len(data_col)):
        A_i = data_col[ind_col]
        ind_A_i = index_col[ind_col]
        n_i = A_i.shape[0]
        eucld_dist_i = euclidean_distances(A_i)

        S_i = dm_sum_nn(eucld_dist_i, k[ind_col])
        S_i = set([ind_A_i[elem] for elem in S_i])
        S.append(S_i)

    copy_S_flat = [list(elem) for elem in S]
    copy_S_flat = list(itertools.chain.from_iterable(copy_S_flat))
    score_S = div_score_sum_nn(eucld_dist, copy_S_flat)

    for ind_col in range(len(data_col)):
        # A_i = data_col[ind_col]
        ind_A_i = index_col[ind_col]
        # eucld_dist_i = euclidean_distances(A_i)
        # n_i = eucld_dist_i.shape[0]
        k_i = k[ind_col]
        G = [0]
        r = []
        P_minus_G = set(ind_A_i).difference(G)
        list_P_minus_G = list(P_minus_G)

        while len(r) < sum_k:
            list_P_minus_G = list(P_minus_G)
            list_G = list(G)
            if len(list_P_minus_G) == 0: break
            dist_mod = eucld_dist[np.ix_(list_P_minus_G, list_G)]

            ind_min_q_per_p = np.argmin(dist_mod, axis=1)
            min_q_per_p = np.min(dist_mod, axis=1)
            max_p = np.argmax(min_q_per_p)
            p_l = list_P_minus_G[max_p]
            G.append(p_l)
            P_minus_G.remove(p_l)
            r.append(np.max(min_q_per_p))

        if (len(r)) < sum_k: continue
        for j in range(1, sum_k):
            t_j = j
            while t_j < sum_k - 1 and r[t_j + 1] >= r[j] / 2.0:  # t_j + 1 < len(r) and
                t_j = t_j + 1

            B = []
            centersB = []
            for v in range(t_j):
                potential_ball = copy.deepcopy(eucld_dist[G[v]])
                B_v = [idx for idx, val in enumerate(list(potential_ball)) if val <= r[t_j] / 2.0 and t_j < len(r)]
                centersB.append(G[v])
                B.append(B_v)

            SOL = [[] for _ in range(len(data_col))]
            Bprime, fin_ind_b = best_subset_bals(B, data_col, index_col, ind_col, k, colors, centersB)
            Bprime_flat = list(itertools.chain.from_iterable(Bprime))

            for ind_Bpr in fin_ind_b:
                if len(SOL[colors[ind_Bpr]]) == k_i: break
                SOL[colors[ind_Bpr]].append(ind_Bpr)

            for indj_col in range(len(data_col)):
                Pj_minus_Bprime = set(index_col[indj_col]).difference(set(Bprime_flat))
                ct = len(SOL[indj_col])
                while ct < k[indj_col]:
                    if len(Pj_minus_Bprime) == 0: break
                    x = Pj_minus_Bprime.pop()
                    if x not in SOL[indj_col]:
                        SOL[indj_col].append(x)
                        ct = ct + 1

            SOL_flat = list(itertools.chain.from_iterable(SOL))
            score_SOL = div_score_sum_nn(eucld_dist, SOL_flat)

            if score_SOL < np.Inf and score_SOL > score_S:
                score_S = score_SOL
                S = SOL

    return S
