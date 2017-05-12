import sparql_tools as qt
import pandas as pd


def fix_categoires(df):
    """
    Takes a query result as a Pandas DataFrame and converts WikiData Category items into
    the category itself by following the 'category's main topic' edge

    :param df: Pandas.DataFrame, The query result dataframe to be altered
    :return: Pandas.DataFrame, the dataframe with corrected categories
    """
    # copy the dataframe to avoid mutation
    df_out = df.copy()

    # text to query for the new categories
    query_text = """
    SELECT distinct ?s ?sLabel ?p ?o ?oLabel
    WHERE
    {{
        values ?s {{wd:{}}}
        # Category's Main Topic
        values ?p {{wdt:P301}}
        # Get edges and object nodes
        ?s ?p ?o .
        # Make sure using direct properties
        FILTER REGEX(STR(?p), "prop/direct")
        FILTER REGEX(STR(?o), "entity/Q")
        # Make sure object nodes (o) have there own edges and nodes
        ?o ?p2 ?o2 .
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}"""

    # Get indicies of the catgories
    idx = df_out['oLabel'].str.contains('Category:')

    # Take the catgories and get their uris and labels
    cat_uri = list(set(df_out[idx]['o']))
    categories = [qt.id_from_uri(cat) for cat in cat_uri]

    # Query along the "category's main topic" edge
    result = qt.query_from_list(categories, query_text=query_text)

    # Create a dictionary to map to new category names
    map_dict = result.set_index('s', drop=True).to_dict()

    # Map the new uris and labels
    to_change = df_out[idx]
    uri = to_change['o'].apply(lambda v: map_dict['o'][v] if v in map_dict['o'].keys() else v)
    label = to_change['o'].apply(lambda v: map_dict['oLabel'][v] if v in map_dict['oLabel'].keys() else v)

    df_out.loc[idx, 'o'] = uri
    df_out.loc[idx, 'oLabel'] = label

    return df_out


def filter_bad(df, bad_ps=None, bad_os=None):
    """
    Filters out predicates and objects determiend to be unimportant to the graph

    :param df: Pandas.DataFrame, the query result dataframe from which the bad lines are to be filtered
    :param bad_ps: List, the list of bad predicates to be removed
    :param bad_os: List, the list of bad objects to be removed
    :return: Pandas.DataFrame, the datafame with the unwanted lines removed
    """

    # initialzie the items to be filtered if no arguments are passed
    if not bad_ps:
        bad_ps = ["topic's main category"]
    if not bad_os:
        bad_os = ["Wikimedia category", "Wikimedia list article", "Wikimedia template"]

    out_df = (df.query('pLabel not in {}'.format(bad_ps)).
                 query('oLabel not in {}'.format(bad_os)).
                 reset_index(drop=True))
    return out_df


def combine_multiclass(df, edge_type="instance of", sep='; '):
    """
    Function to combine the objects of edges of a certain type into 1 string.
    Example: gene 'IZUMO4' has 'insance of' edges to 'gene' and 'protein-coding gene'
    this function run with default settings will return 'gene; protein-coding gene'
    for this gene

    :param df: Pandas.DataFrame, the result dataframe where multi-class items will be found
    :param edge_type: String or list of Strings, the edge type that will be joined for multi-class items
    :param sep: String, the separator for the classes joined
    :return: Pandas.DataFrame, with columns 's' with the uri for the node, and 'label' with the
             new combined label for the node
    """
    # Query and reset the index to create copy
    if type(edge_type) == str:
        df_out = df.query('pLabel == {!r}'.format(edge_type)).reset_index(drop=True)
    elif type(edge_type) == list:
        df_out = df.query('pLabel in {!r}'.format(edge_type)).reset_index(drop=True)
    else:
        raise TypeError

    # Define the function to combine the types
    comb_func = lambda dat: sep.join(list(dat['oLabel']))

    # Combine the types
    df_out = (df_out.
                sort_values(['s', 'oLabel'])[['s', 'oLabel']].
                groupby('s').
                apply(comb_func).
                to_frame().
                reset_index().
                rename(columns={0:'label'}))

    return df_out


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


def nodes_neo_export(edge_df):
    """
    Takes a list of edges, filters for only those connected to subject nodes, and returns dataframe
    with of all the nodes, formatted for neo4j import

    :param edge_df: Pandas.DataFrame, with the edges to be added to the graph
    :return: Pandas.DataFrame, formatted for neo4j, import ready to be exported to csv
    """
    # Get a mapping from uri to type
    type_dict = get_node_type_dict(edge_df, 'type')

    # Filter for subject nodes connected to other subject nodes, and only keep unique
    filt_edges = filter_untyped_nodes(edge_df)
    subj_nodes = filt_edges.drop_duplicates(subset='s')
    subj_nodes = subj_nodes.reset_index(drop=True)

    # Some edges may only be one way, so some objects may not also be in the subject column
    # in this filtered dataframe
    obj_nodes = filt_edges.drop_duplicates(subset='o')
    obj_nodes = obj_nodes.query('o not in {!r}'.format(list(set(subj_nodes['s']))))
    obj_nodes = obj_nodes.reset_index(drop=True)
    obj_nodes.loc[:,'type'] = obj_nodes['o'].apply(lambda uri: type_dict[uri])

    # Convert to Neo4j import format and combine
    subj_out = format_nodes_neo(subj_nodes, col='s')
    obj_out = format_nodes_neo(obj_nodes, col='o')
    node_out = pd.concat([subj_out, obj_out]).drop_duplicates()
    node_out = node_out.reset_index(drop=True)

    return node_out

def filter_untyped_nodes(edge_df):
    """
    Filter out edges that point to objects not in the list of subjects

    :param edge_df:  Pandas.DataFrame, with edges to be added to the graph
    :return: Pandas.DataFrame, with bad edges removed
    """
    subject_uri = list(set(edge_df['s']))
    filt_edges = edge_df.query('o in {!r}'.format(subject_uri))

    return filt_edges


def get_edge_types(edge_df, node_type_dict):
    abbrev_dict = get_abbrev_dict(edge_df)

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


def remove_low_count_edges(edge_df, node_types=None, cutoff=.01):
    """
    Removes the low count edges from the edge dataframe

    :param edge_df:
    :param cutoff:
    :return:
    """
    from collections import Counter
    # Make sure edges have been typed, if not, type them
    if 'e_type' not in edge_df:
        edge_df['e_type'] = get_edge_types(edge_df)

    # Get counts for edge and node types
    edge_type_counts = edge_df['e_type'].value_counts()

    # Provides a more accurate count
    if node_types:
        nodes = set(edge_df['s']).union(set(edge_df['o']))
        node_type_counts = Counter()
        for node in nodes:
            node_type_counts[node_types[node]] += 1
    else:
        # A close approximation if types for the objects are not provided
        node_type_counts = edge_df.drop_duplicates('s')['type'].value_counts()

    # Generate an edge to node mapping
    edge_to_node_mapping = edge_df.set_index('e_type')['type'].to_dict()

    # Find the valid edges
    valid_edges = []
    for edge_type, count in edge_type_counts.to_dict().items():
        node_type = edge_to_node_mapping[edge_type]
        # Keep if there are more than the cuttoff percentage of nodes
        if count >= cutoff * node_type_counts[node_type]:
            valid_edges.append(edge_type)

    return edge_df.query('e_type in {}'.format(valid_edges)).reset_index(drop=True)


def get_pair_lists(edge_df):
    """
    Takes edge list dataframe and converts to a dictionary of pair lists

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
    return [(b, a) for (a, b) in pair_list]


def calc_overlap(pair_list, invert_list, cutoff=0.5):
    count_a, count_b = 0, 0

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
    invert_list = {key: invert_pairs(value) for key, value in pair_list.items()}
    kinds = list(pair_list.keys())

    reciprocal_types = []

    # Only compare 2 different relationship types once
    for i, kind in enumerate(kinds):
        for idx in range(i, len(kinds)):
            overlap = calc_overlap(pair_list[kind], invert_list[kinds[idx]])
            if overlap > cutoff:
                reciprocal_types.append([kind, kinds[idx]])

    reciprocal_types = [sorted(t, key=lambda x: len(x)) for t in reciprocal_types]
    return reciprocal_types


def remove_reciprocals(edge_df, reciprocal_types):
    edge_copy = edge_df.copy()


    def swap_reciprocals(types):
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
        sub_group['tuples'] = list(zip(sub_group['s'], sub_group['o']))

        edge_tups = list(sub_group['tuples'])
        to_keep = []
        while len(edge_tups) > 0:
            tup = edge_tups.pop()
            inverse = (tup[1], tup[0])
            if inverse in edge_tups:
                edge_tups.remove(inverse)
            to_keep.append(tup)

        idx = sub_group['tuples'].isin(to_keep)
        new_sub = sub_group.loc[idx]

        return new_sub.drop('tuples', axis=1)

    for types in reciprocal_types:
        # Relationship is bidirectional
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

    # Remove some special Characters from edge labels
    edge_df['pLabel'] = edge_df['pLabel'].apply(lambda s: re.sub('[^a-zA-Z0-9-_*.]', '', s))

    # Filter the edges
    filt_edges = filter_untyped_nodes(edge_df).copy()
    # Get their types
    filt_edges['e_type'] = get_edge_types(filt_edges, node_type_dict)
    # Remove the edges with low counts
    filt_edges = remove_low_count_edges(filt_edges, node_type_dict, edge_cutoff)

    # Remove reciprocal relationships
    pair_list = get_pair_lists(filt_edges)
    reciprocal_types = find_reciprocal_relations(pair_list, overlap_cutoff)
    filt_edges = remove_reciprocals(filt_edges, reciprocal_types)

    return filt_edges, node_type_dict, reciprocal_types


def get_abbrev_dict(edge_df):
    # Create an abbreviation dict for fast mapping of nodes and edges
    node_types = list(set(edge_df['type']))
    edge_types = list(set(edge_df['pLabel'].str.replace(' ', '-')))

    node_abbrevs = [''.join([w[0].upper() for w in t.split(' ')]) for t in node_types]
    edge_abbrevs = [''.join([w[0].lower() for w in t.split('-')]) for t in edge_types]

    node_abbrev_dict = {t:a for t,a in zip(node_types, node_abbrevs)}
    edge_abbrev_dict = {t:a for t,a in zip(edge_types, edge_abbrevs)}

    return {**node_abbrev_dict, **edge_abbrev_dict}


def get_metaedge_tuples(edge_df, node_type_dict, reciprocal_relations=None, forward_edges=None):
    from itertools import chain
    def get_tuple(row):

        start_kind = node_type_dict[row['s']]
        end_kind = node_type_dict[row['o']]
        edge = row['e_type'].split('_')[0]

        if reciprocal_relations:
            if row['e_type'] in chain(*reciprocal_relations):
                direction = 'both'
            else:
                direction = 'forward'
        else:
            direction = 'both'

        return start_kind, end_kind, edge, direction

    return list(edge_df.apply(get_tuple, axis=1).unique())

def prep_hetio(edge_df, node_types, reciprocal_relations, save_dir='data'):
    """
    Preps a filtered edge_df unconverted, as well as the reciprocal relations and
    metaedge tuples, then produces all the files needed for hetio import and learn pipeline.

    :param edge_df: Pandas.DataFrame, filtered edge dataframedata frame containing edges, output of prep_for_export
    :param node_types: dict, key=uri, value=node_type, output of get_node_types or prep_for_export
    :param_reciprocal_relations: nested list, output of find_reciprocal_relations, or prep_for_export
    :return: None

    """
    import os
    from hetio.hetnet import MetaGraph, Graph
    from hetio.readwrite import write_metagraph
    from hetio.stats import degrees_to_excel, get_metaedge_style_df

    def add_node_from_row(row):
        graph.add_node(kind = row[':LABEL'], identifier=row[':ID'], name=row['name:String'])

    def add_edge_from_row(row):
        start_id = (node_types_id[row[':START_ID']], row[':START_ID'])
        end_id = (node_types_id[row[':END_ID']], row[':END_ID'])
        kind = row[':TYPE'].split('_')[0]
        graph.add_edge(start_id, end_id, kind, dir_dict[kind])

    # Get metaedge tuples and abbreviation dicts to build the metagraph
    metaedge_tuples = get_metaedge_tuples(edge_df, node_types, reciprocal_relations)
    abbrev_dict = get_abbrev_dict(edge_df)

    metagraph = MetaGraph.from_edge_tuples(metaedge_tuples, abbrev_dict)

    # Format new dicts needed for mappings
    node_types_id = {qt.id_from_uri(k):v for k,v in node_types.items()}
    dir_dict = {x[2]: x[3] for x in metaedge_tuples}

    # Get the neo formatted version of nodes and edges (ensure exact match)
    neo_nodes = format_nodes_neo(edge_df, node_types)
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
