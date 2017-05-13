import sparql_tools as qt
import pandas as pd
import os


def get_node_type_dict(df, label = 'label'):
    """
    Returns a mapping dictionary from node URI to the node's type

    :param df: Pandas.DataFrame, with column 's' containing node URIs and column 'label' class names
    :param label: String, alternate names column containing class names
    :return: dict, with keys URIs and values classes for the node
    """
    return df.set_index('s', drop=True).to_dict()[label]


def format_nodes_neo(edge_df, type_dict):
    """
    Takes a list of edges, and returns a csv of just the nodes, formatted for Neo4j import.

    :param edge_df: Pandas.DataFrame with of the edges to be added to the graph
    :param col: String, the column uri and label to produce the nodes from
    :return: Pandas.DataFrame, formatted for neo4j import, ready to be exported to CSV
    """

    node_out = pd.DataFrame()
    node_out[':ID'] = pd.concat([edge_df['s'], edge_df['o']])
    node_out[':LABEL'] = node_out[':ID'].apply(lambda u: type_dict[u])
    node_out[':ID'] = node_out[':ID'].apply(qt.id_from_uri)
    node_out['identifier:String'] = node_out[':ID']
    node_out['name:String'] = pd.concat([edge_df['sLabel'], edge_df['oLabel']])

    node_out = node_out[[':ID', 'identifier:String', 'name:String', ':LABEL']].drop_duplicates()
    node_out.reset_index(drop=True)
    return node_out


def format_edges_neo(edge_df):
    """
    Takes a list of edges, and converts the format to a neo4j importable csv

    :param edge_df: Pandas.DataFrame, with edges ready to be added to the graph
    :return: Pandas.DataFrame, formatted for neo4j
    """
    edge_out = pd.DataFrame()

    edge_out[':START_ID'] = edge_df['s'].apply(qt.id_from_uri)
    edge_out[':END_ID'] = edge_df['o'].apply(qt.id_from_uri)
    edge_out[':TYPE'] = edge_df['e_type']

    return edge_out


def filter_untyped_nodes(edge_df):
    """
    Filter out edges that point to objects not in the list of subjects

    :param edge_df:  Pandas.DataFrame, with edges to be added to the graph
    :return: Pandas.DataFrame, with bad edges removed
    """
    subject_uri = list(set(edge_df['s']))
    filt_edges = edge_df.query('o in {!r}'.format(subject_uri)).reset_index(drop=True)

    return filt_edges


def get_edge_types(edge_df, node_type_dict):
    """
    Given an edge dataframe and a dictionary of node types, returns the edge types as a Series.

    :param edge_df: pandas.DataFrame, edges, with untyped nodes filtered out.
    :param node_type_dict: dictionary, keys are node uri, values, the type
    :return: pandas.Series, the edge types
    """
    abbrev_dict = get_abbrev_dict(edge_df, node_type_dict)

    def get_edge_type(row):
        # Get types and abbreviations for edge start type and end type
        start_type = node_type_dict[row['s']]
        end_type = node_type_dict[row['o']]
        edge_type = row['pLabel'].replace(' ', '-')

        start_abbrev = abbrev_dict[start_type]
        end_abbrev = abbrev_dict[end_type]
        edge_abbrev = abbrev_dict[edge_type]

        # Example 'Compound' -physically interacts with-> 'Protein' becomes:
        # 'physically-interacts-with_CpiwP'
        return edge_type + '_' + start_abbrev + edge_abbrev + end_abbrev

    edge_types = edge_df.apply(get_edge_type, axis=1)
    return edge_types


def remove_low_count_edges(edge_df, node_types, cutoff=.01):
    """
    Removes edges of low degree in comparison to the amount of nodes of a given type, as compared to the
    node of less abundant type.

    :param edge_df: pandas.DataFrame, with untyped object nodes filtered out.
    :param node_types: dictionary, with keys node uri, and values node type
    :param cutoff: float, a minimum fraction of edges per node for a given edge type
    :return: pandas.DataFrame, with all edges that do not meet the cutoff removed.
    """
    from collections import Counter
    # Copy dataframe to avoid mutation
    edge_df_c = edge_df.copy()
    
    # Make sure edges have been typed, if not, type them
    if 'e_type' not in edge_df_c:
        edge_df_c['e_type'] = get_edge_types(edge_df_c)

    # Get counts for edge and node types
    edge_type_counts = edge_df_c['e_type'].value_counts()

    # Count the number of unique nodes of each type
    nodes = set(edge_df_c['s']).union(set(edge_df_c['o']))
    node_type_counts = Counter()
    for node in nodes:
        node_type_counts[node_types[node]] += 1

    # Make a mapping of start and end node types
    edge_to_start_mapping = edge_df_c.set_index('e_type')['s'].apply(lambda x: node_types[x]).to_dict()
    edge_to_end_mapping = edge_df_c.set_index('e_type')['o'].apply(lambda x: node_types[x]).to_dict()

    # Find the valid edges
    valid_edges = []
    for edge_type, edge_count in edge_type_counts.to_dict().items():
        # Take the start and end node types
        start_type = edge_to_start_mapping[edge_type]
        end_type = edge_to_end_mapping[edge_type]
        # Find which one is smaller
        node_count = min(node_type_counts[start_type], node_type_counts[end_type])

        # keep if the there are more edges than the cutoff percentage ot the node count
        if edge_count >= cutoff * node_count:
            valid_edges.append(edge_type)

    return edge_df_c.query('e_type in {}'.format(valid_edges)).reset_index(drop=True)


def get_pair_lists(edge_df):
    """
    Takes edge dataframe and converts to a dictionary of pair lists, with keys edge type, and values a list
    of tuples, first value edge start URI and second value edge end URI.

    :param edge_df: pandas.DataFrame edge list with node and edge types included
    :return: Dictionary, key is edge type, value, list of tuples, URIs of start and end nodes for given edge
    """
    # Get edge types
    edge_types = set(edge_df['e_type'])

    # Iterate over types and get a pair list for each
    pair_list_dict = dict()
    for kind in edge_types:
        edges_of_type = edge_df[edge_df['e_type'] == kind]

        start = list(edges_of_type['s'])
        end = list(edges_of_type['o'])
        pair_list = [(s, e) for s, e in zip(start, end)]

        pair_list_dict[kind] = pair_list

    return pair_list_dict


def invert_pairs(pair_list):
    """
    Takes a pair list and swaps start and end URIs for all pairs.

    :param pair_list: list of tuples, Start and End URIs for each edge of a given type
    :return: list of tuples, with start and end URIs inverted.
    """
    return [(b, a) for (a, b) in pair_list]


def calc_overlap(pair_list, invert_list, cutoff=0.5):
    """
    Given two edge pairs, calculates the overlap percentage of the two lists. This is
    used to see if the two pair lists represent a reciprocal relationship.

    :param pair_list: List of tuples, the start and end URIs for a give edge type
    :param invert_list: List of tuples, the result of invert_pairs of a pair list
    :param cutoff: float, fraction of edges that must overlap between the two lists
    :return: the minimum overlap.
    """

    # if there is a large difference in size, don't bother checking
    if (len(pair_list) / len(invert_list) < cutoff or
            len(invert_list) / len(pair_list) < cutoff):
        return 0

    # Check both directions
    count_a = len(set(pair_list).intersection(set(invert_list)))
    count_b = len(set(invert_list).intersection(set(pair_list)))

    # Return the minimum overlap
    return min(count_a / len(pair_list), count_b / len(invert_list))


def find_reciprocal_relations(pair_list, cutoff=0.5):
    """
    Finds out which edges types are opposites of each other, using a certain cutoff for overlap percetage.
    0.5 is fairly permissive, however most edges are either exactly, or very near 0 or very near 1.

    :param pair_list: dictionary, key = edge type, and value is a list of tuples, with start and end URIs for the edge
    :param cutoff: float, the minimum fraction of overlap between edges to be considered recipcrocal
    :return: list of lists, with the two edge types found to be reciprocal to each other
    """
    invert_list = {key: invert_pairs(value) for key, value in pair_list.items()}
    kinds = list(pair_list.keys())

    reciprocal_types = []

    # Only compare 2 different relationship types once
    for i, kind in enumerate(kinds):
        for idx in range(i, len(kinds)):
            overlap = calc_overlap(pair_list[kind], invert_list[kinds[idx]])
            if overlap >= cutoff:
                reciprocal_types.append([kind, kinds[idx]])

    reciprocal_types = [sorted(t, key=lambda x: len(x)) for t in reciprocal_types]
    return reciprocal_types


def remove_reciprocals(edge_df, reciprocal_types):
    """
    Removes the reciprocal edges from the edge dataframe.

    :param edge_df: pandas.DataFrame, edge format, with edge types
    :param reciprocal_types: list of lists, each containing the two edge types reciprocal to each other
    :return: pandas.DataFrame, with only one of the two reciprocal types remaining
    """
    edge_copy = edge_df.copy()

    def swap_reciprocals(types):
        """
        Takes two edge types, swaps start and end posistions of 1 type, and renames that type to the other type.

        :param types: list, the two types to be swapped
        :return: The df slice with the swapped edges.
        """

        # Make a copy of the subsection
        swapped = edge_copy.loc[edge_copy['e_type'] == types[1]].copy()

        # Swap URIs and Labels
        tmp_uri = swapped['s'].copy()
        tmp_label = swapped['sLabel'].copy()
        swapped['s'] = swapped['o']
        swapped['sLabel'] = swapped['oLabel']
        swapped['o'] = tmp_uri
        swapped['oLabel'] = tmp_label

        # Change the edge type
        swapped['e_type'] = types[0]
        return swapped

    def remove_duplicates(sub_group):
        """
        When the same edge goes in both directions e.g. 'Compound <-signficatn-drug-intraction-> Compound'
        inverted edges must be removed.

        :param sub_group: slice of the DataFrame with only the duplicated edge type.
        :return: slice of the Dataframe, with duplicate edges removed.
        """
        # Bind the start and end ids together
        sub_group['tuples'] = list(zip(sub_group['s'], sub_group['o']))
        edge_tups = list(sub_group['tuples'])

        to_keep = []
        while len(edge_tups) > 0:
            # Store each edge
            tup = edge_tups.pop()
            to_keep.append(tup)

            # If the inverse still in the list, remove it
            inverse = (tup[1], tup[0])
            if inverse in edge_tups:
                edge_tups.remove(inverse)

        # only keep new unique edges
        idx = sub_group['tuples'].isin(to_keep)
        new_sub = sub_group.loc[idx]

        return new_sub.drop('tuples', axis=1)

    for types in reciprocal_types:
        # Relationship is bidirectional; can't just query for 1 type or the other
        if types[0] == types[1]:
            sub_group = edge_copy[edge_copy['e_type'] == types[0]].copy()
            new_sub = remove_duplicates(sub_group)
            edge_copy.drop(sub_group.index, axis=0, inplace=True)
            edge_copy = pd.concat([edge_copy, new_sub]).reset_index(drop=True)
        else:
            edge_copy.loc[edge_copy['e_type'] == types[1]] = swap_reciprocals(types)

    return edge_copy.drop_duplicates(subset=['s','o', 'e_type'])


def prep_for_export(edge_df, overlap_cutoff=0.5, edge_cutoff=0.01):
    """
    Takes a list of edges, and returns a csv of the edges, formatted for Neo4j import.

    :param edge_df: Pandas.DataFrame, with the edges to be added to the graph
    :return: Pandas.DataFrame, formatted for neo4j, import ready to be exported to csv
    """
    import re

    # Get node types first
    node_type_dict = get_node_type_dict(edge_df, 'type')

    # Remove edges where objects (o) are not in the set of subjects (s)
    filt_edges = filter_untyped_nodes(edge_df)

    # Remove some special characters and extra spaces from edge names
    func = lambda s: ' '.join(re.sub('[^a-zA-Z0-9-_*.]', ' ', s).split())
    filt_edges['pLabel'] = filt_edges['pLabel'].apply(func)

    # Get the edge types
    filt_edges['e_type'] = get_edge_types(filt_edges, node_type_dict)

    # Remove the edges with low counts
    filt_edges = remove_low_count_edges(filt_edges, node_type_dict, edge_cutoff)

    # Remove reciprocal relationships
    pair_list = get_pair_lists(filt_edges)
    reciprocal_types = find_reciprocal_relations(pair_list, overlap_cutoff)
    filt_edges = remove_reciprocals(filt_edges, reciprocal_types)

    return filt_edges, node_type_dict, reciprocal_types


def get_abbrev_dict(edge_df, node_types):
    """
    Gets a dictionary of abbreviations for node and edge types

    :param edge_df: pandas.DataFrame of edges with e_type column included.
    :param node_types: dictionary, key node URIs, and value types
    :return: dictionary, with keys node or edge types, and values the respective abbreviation
    """

    import re

    # Create an abbreviation dict for fast mapping of nodes and edges
    node_types = list(set(node_types.values()))
    edge_types = list(set(edge_df['pLabel'].str.replace(' ', '-')))

    node_abbrevs = [''.join([w[0].upper() for w in re.split('[ -]', t)]) for t in node_types]
    edge_abbrevs = [''.join([w[0].lower() for w in t.split('-')]) for t in edge_types]

    node_abbrev_dict = {t:a for t,a in zip(node_types, node_abbrevs)}
    edge_abbrev_dict = {t:a for t,a in zip(edge_types, edge_abbrevs)}

    return {**node_abbrev_dict, **edge_abbrev_dict}


def get_metaedge_tuples(edge_df, node_type_dict, reciprocal_relations=None, forward_edges=None):
    """
    Gets the metaedge tuples needed for the a metagraph.

    :param edge_df:
    :param node_type_dict:
    :param reciprocal_relations:
    :param forward_edges:
    :return:
    """
    from itertools import chain

    def get_tuple(row):
        start_kind = node_type_dict[row['s']]
        end_kind = node_type_dict[row['o']]
        edge = row['e_type'].split('_')[0]

        # If forward edges supplied, use to define directions
        if forward_edges:
            if row['e_type'] in forward_edges:
                direction = 'forward'
            else:
                direction = 'both'

        # Use reciprocal relations to define directions
        elif reciprocal_relations:
            if row['e_type'] in chain(*reciprocal_relations):
                direction = 'both'
            else:
                direction = 'forward'

        # if nothing supplied, every edge is bidirectional
        else:
            direction = 'both'

        return start_kind, end_kind, edge, direction

    return list(edge_df.apply(get_tuple, axis=1).unique())


def prep_hetio(edge_df, node_types, reciprocal_relations, save_dir='data', forward_edges=None,
               neo_nodes=None, neo_edges=None):
    """
    Preps a filtered edge_df unconverted, as well as the reciprocal relations and
    metaedge tuples, then produces all the files needed for hetio import and learn pipeline.

    :param edge_df: Pandas.DataFrame, filtered edge dataframedata frame containing edges, output of prep_for_export
    :param node_types: dict, key=uri, value=node_type, output of get_node_types or prep_for_export
    :param reciprocal_relations: nested list, output of find_reciprocal_relations, or prep_for_export
    :param save_dir: string, directory to save the output files
    :param forward_edges: list, the edge types to be forced to forward relationships in graph
    :param neo_nodes: pandas.DataFrame, nodes formatted for neo4j import if available
    :param neo_edges: pandas.DataFrame, edges formatted for neo4j import if available
    :return: None

    """
    from hetio.hetnet import MetaGraph, Graph
    from hetio.readwrite import write_metagraph
    from hetio.stats import degrees_to_excel, get_metaedge_style_df

    def add_node_from_row(row):
        graph.add_node(kind=row[':LABEL'], identifier=row[':ID'], name=row['name:String'])

    def add_edge_from_row(row):
        start_id = (node_types_id[row[':START_ID']], row[':START_ID'])
        end_id = (node_types_id[row[':END_ID']], row[':END_ID'])
        kind = row[':TYPE'].split('_')[0]
        graph.add_edge(start_id, end_id, kind, dir_dict[row[':TYPE']])

    # Get metaedge tuples and abbreviation dicts to build the metagraph
    metaedge_tuples = get_metaedge_tuples(edge_df, node_types, reciprocal_relations, forward_edges)
    abbrev_dict = get_abbrev_dict(edge_df, node_types)

    metagraph = MetaGraph.from_edge_tuples(metaedge_tuples, abbrev_dict)
    filename = os.path.join(save_dir, 'metagraph.json')
    write_metagraph(metagraph, filename)

    # Format new dicts needed for mappings
    node_types_id = {qt.id_from_uri(k): v for k, v in node_types.items()}
    # Make a mapper from edge to direction
    dir_dict = {me[2] + '_' + abbrev_dict[me[0]] + abbrev_dict[me[2]] +
                abbrev_dict[me[1]]: me[3] for me in metaedge_tuples}

    # Get the neo formatted version of nodes and edges (ensure exact match)
    if not neo_nodes:
        neo_nodes = format_nodes_neo(edge_df, node_types)
    if not neo_edges:
        neo_edges = format_edges_neo(edge_df)

    # Initialize graph and add nodes and edges
    graph = Graph(metagraph)
    neo_nodes.apply(add_node_from_row, axis=1)
    neo_edges.apply(add_edge_from_row, axis=1)

    print('Added {} nodes to graph'.format(len(list(graph.get_nodes()))))
    print('Added {} edges to graph'.format(len(list(graph.get_edges()))))

    # Save Files
    filename = os.path.join(save_dir, 'degrees.xlsx')
    degrees_to_excel(graph, filename)

    filename = os.path.join(save_dir, 'metaedge-styles.tsv')
    metaedge_style_df = get_metaedge_style_df(metagraph)
    metaedge_style_df.to_csv(filename, sep='\t', index=False)


def process_edges(edge_df, save_dir='data', forward_edges=None):
    """


    :param edge_df:
    :param save_dir:
    :param forward_edges:
    :return:
    """

    # Prep the edges for export
    prepped, node_types, reciprocal_relations = prep_for_export(edge_df)

    # Format to neo
    neo_nodes = format_nodes_neo(prepped, node_types)
    neo_edges = format_edges_neo(prepped)

    # Save to disk
    nodes_out = os.path.join(save_dir, 'hetnet_nodes.csv')
    edges_out = os.path.join(save_dir, 'hetnet_edges.csv')
    neo_nodes.to_csv(nodes_out, index=False)
    neo_edges.to_csv(edges_out, index=False)

    # Get the files ready for learn, via hetio
    prep_hetio(edge_df, node_types, reciprocal_relations, save_dir, forward_edges=forward_edges,
               neo_nodes=neo_nodes, neo_edges=neo_edges)