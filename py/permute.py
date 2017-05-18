import pandas as pd
import collections
import random
import logging
import os


def get_pair_lists(edges):
    """
    Takes edge list dataframe formatted for neo4j and converts to a dictionary of pair lists

    :param edges: pandas.DataFrame edge list formatted for neo4j import
    :return: Dictionary, key is edge type, value, list of tuples, ids of start and end nodes for given edge
    """
    # Get edge types
    edge_types = set(edges[':TYPE'])

    # Iterate over types and get a pair list for each
    pair_list_dict = dict()
    for kind in edge_types:
        edges_of_type = edges[edges[':TYPE'] == kind]

        start = list(edges_of_type[':START_ID'])
        end = list(edges_of_type[':END_ID'])
        pair_list = [(s, e) for s, e in zip(start, end)]

        pair_list_dict[kind] = pair_list

    return pair_list_dict


def permute_pair_list(pair_list, directed=False, multiplier=10, excluded_pair_set=set(), seed=0, log=False):
    """
    From a list of node-pairs that make up an edge, permutes the edges such that degree is preserved for a given edge

    If n_perm is not specific, perform 10 times the number of edges of permutations
    May not work for directed edges
    """
    random.seed(seed)

    pair_set = set(pair_list)
    assert len(pair_set) == len(pair_list)

    edge_number = len(pair_list)
    n_perm = int(edge_number * multiplier)

    count_same_edge = 0
    count_self_loop = 0
    count_duplicate = 0
    count_undir_dup = 0
    count_excluded = 0

    if log:
        logging.info('{} edges, {} permutations (seed = {}, directed = {}, {} excluded_edges)'.format(
            edge_number, n_perm, seed, directed, len(excluded_pair_set)))

    orig_pair_set = pair_set.copy()
    step = max(1, n_perm // 10)
    print_at = list(range(step, n_perm, step)) + [n_perm - 1]

    stats = list()
    for i in range(n_perm):

        # Same two random edges
        i_0 = random.randrange(edge_number)
        i_1 = random.randrange(edge_number)

        # Same edge selected twice
        if i_0 == i_1:
            count_same_edge += 1
            continue
        pair_0 = pair_list.pop(i_0)
        pair_1 = pair_list.pop(i_1 - 1 if i_0 < i_1 else i_1)

        new_pair_0 = pair_0[0], pair_1[1]
        new_pair_1 = pair_1[0], pair_0[1]

        valid = False
        for pair in new_pair_0, new_pair_1:
            if pair[0] == pair[1]:
                count_self_loop += 1
                break  # edge is a self-loop
            if pair in pair_set:
                count_duplicate += 1
                break  # edge is a duplicate
            if not directed and (pair[1], pair[0]) in pair_set:
                count_undir_dup += 1
                break  # edge is a duplicate
            if pair in excluded_pair_set:
                count_excluded += 1
                break  # edge is excluded
        else:
            # edge passed all validity conditions
            valid = True

        # If new edges are invalid
        if not valid:
            for pair in pair_0, pair_1:
                pair_list.append(pair)

        # If new edges are valid
        else:
            for pair in pair_0, pair_1:
                pair_set.remove(pair)
            for pair in new_pair_0, new_pair_1:
                pair_set.add(pair)
                pair_list.append(pair)

        if i in print_at:
            stat = collections.OrderedDict()
            stat['cumulative_attempts'] = i
            index = print_at.index(i)
            stat['attempts'] = print_at[index] + 1 if index == 0 else print_at[index] - print_at[index - 1]
            stat['complete'] = (i + 1) / n_perm
            stat['unchanged'] = len(orig_pair_set & pair_set) / len(pair_set)
            stat['same_edge'] = count_same_edge / stat['attempts']
            stat['self_loop'] = count_self_loop / stat['attempts']
            stat['duplicate'] = count_duplicate / stat['attempts']
            stat['undirected_duplicate'] = count_undir_dup / stat['attempts']
            stat['excluded'] = count_excluded / stat['attempts']
            stats.append(stat)

            count_same_edge = 0
            count_self_loop = 0
            count_duplicate = 0
            count_undir_dup = 0
            count_excluded = 0

    assert len(pair_set) == edge_number
    return pair_list, stats


def pair_list_to_df(pair_list_dict):
    """
    Takes the dictonary of pair lists and converts back to a DataFrame

    :param pair_list_dict: Dictionary, keys being edge types, and values being list of node pairs making up an edge
    :return: pandas.DataFrame, edge list formatted for neo4j import
    """

    td = dict()
    td[':START_ID'] = []
    td[':END_ID'] = []
    td[':TYPE'] = []
    for key, values in pair_list_dict.items():
        td[':START_ID'] += [v[0] for v in values]
        td[':END_ID'] += [v[1] for v in values]
        td[':TYPE'] += [key]*len(values)

    return pd.DataFrame(td)


def permute_edges(edges, multiplier=10, seed=0):
    """
    Permutes the edges from a given edge dataframe.

    :param edges: pandas.DataFrame, edge list formatted for neo4j import
    :param multiplier: int, multiplier for number of permutations that should be done on edges
    :param seed: int, the seed for random numbers used in edge permutation
    :return: pandas.DataFrame, edge list formatted for neo4j import with edges permuted
    :return: pandas.DataFrame, statistics on the permutations
    """

    pair_list_dict = get_pair_lists(edges)

    # Container for Stats
    all_stats = list()

    # Permute from the pair list
    permuted_pair_lists = dict()
    for kind, pair_list in pair_list_dict.items():
        permuted_pair_list, stats = permute_pair_list(pair_list, directed=True, multiplier=multiplier, seed=seed)
        permuted_pair_lists[kind] = permuted_pair_list
        for stat in stats:
            stat['metaedge'] = kind
            stat['abbrev'] = kind.split('_')[-1]
        all_stats.extend(stats)

    # Return to a dataFrame
    return pair_list_to_df(permuted_pair_lists), all_stats


def run_permute_pipeline(edge_file, n_perms=5, multiplier=5, save_dir='neo/import'):
    """
    Runs the entire permute pipeline and saves the permuted egde files and statistics to disk

    :param edge_file: String, the location of the neo4j import formatted edge file to be permuted
    :param n_perms: int, number of permutations to produce
    :param multiplier: int, multiplier for the permutations
    :param save_dir: String, location of the save folder
    :return: None
    """
    import time

    # Get start time
    start = time.time()

    # Read in edge file
    print('Reading Edges...')
    edges = pd.read_csv(edge_file)

    # Initialize other variables
    path = os.path.abspath(save_dir)
    permuted_edges = edges
    stat_dfs = []

    for i in range(1, n_perms+1):
        # Permute edges
        print('Starting permutation', i)
        permuted_edges, stats = permute_edges(permuted_edges, multiplier=multiplier, seed=i)

        # Add permutation statistics
        stat_df = pd.DataFrame(stats)
        stat_df['permutation'] = i
        stat_dfs.append(stat_df)

        # Save permuted edges
        filename = os.path.join(path, 'hetnet_edges_perm-{}.csv'.format(i))
        permuted_edges.to_csv(filename, index=False)
        print('Saved file:\n{}'.format(filename))
        print('')

    # Save statistics
    stat_df = pd.concat(stat_dfs)
    filename = os.path.join(path, 'stats.tsv')
    stat_df.to_csv(filename, sep='\t', index=False, float_format='%.5g')
    print('Saved statistics on permutations to:\n{}'.format(filename))

    # Get and print timing info
    m, s = divmod(time.time() - start, 60)
    h, m = divmod(m, 60)
    print('Took {} hours, {} minutes, {} seconds'.format(int(h), int(m), int(s)))
