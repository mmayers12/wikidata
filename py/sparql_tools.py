from pymantic import sparql
import pandas as pd
import xmltodict
import requests
import json
import os

def id_from_uri(uri):
    """
    Splits the URI for a property to give the ID

    :param uri: String, the URI for the property
    :return: String, the entity or property's ID (e.g. Q20747295 or P31)
    """
    return uri.split('/')[-1]



class Mapping(object):
    """
    A Base Class to hold id to Label mappings for items from wikidata
    """
    def __init__(self, map_file):
        """
        :param map_file: String, location of .json file containig previously acquired mappings
        """
        self.file_loc = map_file
        if os.path.exists(map_file):
            with open(map_file, 'r') as fin:
                self.mapp = json.load(fin)
        else:
            self.mapp = dict()


    def save_map(self, loc = None):
        """
        Saves all mappings to a json file.
        If no location added, and no changes, prints a no changes message.

        :param loc: file location to save the mappings, if used, will overwrite any existing file
        """

        if not loc:
            with open(self.file_loc, 'r') as fin:
                mapp_dict = json.load(fin)
            # Compare what new properties we've looked up to those already in the file
            if set(self.mapp.keys()) - set(mapp_dict.keys()):
                # This block will run if there's something in this obj, not already in file.
                self.mapp.update(mapp_dict)
                with open(self.file_loc, 'w') as fout:
                    json.dump(self.mapp, fout, indent=2)
            else:
                name = self.__class__.__name__
                print('No new {} to save'.format(name))

        else:
            with open(loc, 'w') as fout:
                json.dump(self.mapp, fout, indent=2)

class Properties(Mapping):
    """
    Contains mappings from property ids to the label for the proerty.
    e.g. P31 -> 'instance of'
    """
    def __init__(self, map_file = 'wd_properties.json'):
        """
        :param map_file: string, the location of the .json file that contains previously determined mappings
        """
        Mapping.__init__(self, map_file)


    def get_prop_label(self, uri):
        """
        Function to get the property name from a uri

        :param uri: String, the wikidata property uri
        :return: String, the label for the given property (e.g. P31 reutrns 'instance of')
        """

        # Get the id for the property
        pid = id_from_uri(uri)

        # Check to see if the property is alreay in obj
        if pid in self.mapp:
            return self.mapp[pid]

        else:
            # request the info for the property
            r = requests.get(uri)
            # Parse the xml
            rdict = xmltodict.parse(r.text)
            # Get the page title which is the property name
            title = rdict['html']['head']['title']

            # titles appear like this: 'Title name - Wikidata'
            # Just want the title name
            title = title[:title.find('-')-1]

            # Add to dictionary for future use
            self.mapp[pid] = title

        return title


class Entities(Mapping):
    def __init__(self, map_file = 'wd_entities.json'):
        """
        Class to hold mappings from wikidata entitiy Qids to entity labels
        e.g. Q1472 -> 'Crohn's Disease'
        """
        Mapping.__init__(self, map_file)

    def get_entity_label(self, uri):
        """
        Function to get the name from the entities uri

        :param prop: String, the wikidata property code (e.g. P31)
        :return: String, the label for the given property (e.g. P31 reutrns 'instance of')
        """

        # Initialzie needed values
        eid = id_from_uri(uri)
        new_eid = None

        # Check to see if the entity is alreay in the file
        if eid in self.mapp:
            return self.mapp[eid]

        else:
            # Unlike properties, entities get changed a lot more readily
            # and funny things can happen like merging and duplicaiton
            try:
                res = requests.get(uri)
                r_dict = json.loads(res.text)
                if eid in r_dict['entities']:
                    labels = r_dict['entities'][eid]['labels']
                else:
                    # Some entities have been merged or renamed, and automatically redirect
                    # to a new value. If so, keep information for both.
                    new_eid = list(r_dict['entities'].keys())[0]
                    labels = r_dict['entities'][new_eid]['labels']

                # Make sure there is an english label (soure of lots of errors)
                if 'en' in labels:
                    label = labels['en']['value']
                else:
                    label = 'No label defined'


                # Add new mapping to the map dictionary
                self.mapp[eid] = label
                if new_eid:
                    self.mapp[new_id] = label
            # If there's an error in this at all, just give back the ID
            except:
                label = eid

        return label


def open_datafile(filename):
    """
    Opens a file and returns the data as a list

    :param filename: String, the name of the file
    :return: List, the list of the items in the datafile
    """
    out_list = []
    with open('data/'+filename, 'r') as fin:
        for line in fin.readlines():
            out_list.append(line.strip())
    return out_list


def query_to_df(result):
    """
    Takes the json result from a sparql query and converts to a Pandas DataFrame

    :param result: json, result from sparql query
    :return: DataFrame, results in tabulated dataframe format
    """
    dat = result['results']['bindings']
    dat1 = []
    for d in dat:
        d = {k:v['value'] for k, v in d.items()}
        dat1.append(d)
    return pd.DataFrame(dat1)


def query_from_list(query_list, query_text=None, url='http://127.0.0.1:9999/bigdata/sparql'):
    """
    Takes a list and queries the sparql server for each item in the list.
    Returns all edges and nodes 1 degree out from source.

    :param query_list: list, the items to be queried
    :reutrn: DataFrame, results from the qurey in tabulated dataframe format
    """
    from tqdm import tqdm
    import time

    # Initialze server
    server = sparql.SPARQLServer(url)
    # Initialize Query Text
    if not query_text:
        query_text = """
        SELECT distinct ?s ?sLabel ?p ?o ?oLabel
        WHERE
        {{
            values ?s {{wd:{}}}
            # Get edges and object nodes
            ?s ?p ?o .
            # Make sure using direct properties
            FILTER REGEX(STR(?p), "prop/direct")
            FILTER REGEX(STR(?o), "entity/Q")
            # Make sure object nodes (o) have there own edges and nodes
            ?o ?p2 ?o2 .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}"""

    # Initalize results
    results = []
    # Append results for each item
    print('Running queries...')
    time.sleep(1)
    for item in tqdm(query_list):
        results.append(server.query(query_text.format(item)))

    # Make properties
    props = Properties()

    # Concatenate results into 1 Dataframe
    results_dfs = []
    for result in results:
        # Make into df
        res_df = query_to_df(result)
        if not res_df.empty:
            results_dfs.append(res_df)
    df = pd.concat(results_dfs)

    # Reset the index, remove duplicates
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset = ['s', 'p', 'o'])

    # Remove any non-entitiy objects
    df['oid'] = df['o'].apply(id_from_uri)
    idx = df['oid'].str.contains('Q')
    df = df[idx]
    df = df.reset_index(drop=True)

    # Generate labels for the proerties
    df['pLabel'] = df['p'].apply(props.get_prop_label)

    # Add in property
    props.save_map()

    # Return df
    return df[['sLabel', 'pLabel', 'oLabel', 's', 'p', 'o']]
