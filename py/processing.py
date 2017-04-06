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


def class_mapp_dict(df, label = 'label'):
    """
    Returns a mapping dictionary from node URI to the node's class

    :param df: Pandas.DataFrame, with column 's' containing node URIs and column 'label' class names
    :param label: String, alternate names column containing class names
    :return: dict, with keys URIs and values classes for the node
    """
    return df.set_index('s', drop=True).to_dict()[label]


def format_nodes_neo(edge_df, col='s'):
    """
    Takes a list of edges, and returns a csv of just the nodes, formatted for Neo4j import.

    :param edge_df: Pandas.DataFrame with of the edges to be added to the graph
    :param col: String, the column uri and label to produce the nodes from
    :return: Pandas.DataFrame, formatted for neo4j import, ready to be exported to CSV
    """

    node_out = pd.DataFrame()
    node_out[':ID'] = edge_df[col].apply(qt.id_from_uri)
    node_out['identifier:String'] = node_out[':ID']
    node_out['name:String'] = edge_df[col+'Label']
    node_out[':LABEL'] = edge_df['type']

    node_out.reset_index(drop=True)
    return node_out


def format_edges_neo(edge_df):
    """

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
    type_dict = edge_df.set_index('s')['type'].to_dict()

    # Filter for subject nodes connected to other subject nodes, and only keep unique
    subject_uri = list(set(edge_df['s']))
    filt_edges = edge_df.query('o in {!r}'.format(subject_uri))
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



def edges_neo_export(edge_df):
    """
    Takes a list of edges, and returns a csv of the edges, formatted for Neo4j import.

    :param edge_df: Pandas.DataFrame, with the edges to be added to the graph
    :return: Pandas.DataFrame, formatted for neo4j, import ready to be exported to csv
    """
    type_dict = edge_df.set_index('s')['type'].to_dict()
    subject_uri = list(set(edge_df['s']))
    filt_edges = edge_df.query('o in {!r}'.format(subject_uri))

    # Create an abbreviation dict for fast mapping
    types = list(set(edge_df['type']))
    abbrevs = [''.join([w[0].upper() for w in t.split(' ')]) for t in types]
    abbrev_dict = {t:a for t,a in zip(types, abbrevs)}

    def get_edge_type(row):
        # Get abbreviaton for edge start type and end type
        start_type = type_dict[row['s']]
        start_abbrev = abbrev_dict[start_type]
        end_type = type_dict[row['o']]
        end_abbrev = abbrev_dict[end_type]

        # Example 'Compound' -physically interacts with-> 'Protein' becomes:
        # 'physically-interacts-with_CpP'
        return row['pLabel'].replace(' ', '-') + '_'+ start_abbrev + row['pLabel'][0] + end_abbrev

    edge_out = pd.DataFrame()
    edge_out[':START_ID'] = filt_edges['s'].apply(qt.id_from_uri)
    edge_out[':END_ID'] = filt_edges['o'].apply(qt.id_from_uri)
    edge_out[':TYPE'] = filt_edges.apply(get_edge_type, axis = 1)

    return edge_out

