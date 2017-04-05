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
    :param edge_type: String, the edge type that will be joined for multi-class items
    :param sep: String, the separator for the classes joined
    :return: Pandas.DataFrame, with columns 's' with the uri for the node, and 'label' with the
             new combined label for the node
    """
    # Query and reset the index to create copy
    df_out = df.query('pLabel == {!r}'.format(edge_type)).reset_index(drop=True)

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
