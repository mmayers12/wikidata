import os
import sys

import hashlib
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from glmnet import LogitNet
from itertools import product
from sklearn.model_selection import StratifiedKFold

sys.path.append('../../hetnet-ml/src')
import graph_tools as gt
from extractor import MatrixFormattedGraph
from processing import DegreeTransform, DWPCTransform

## Set arguments to run the script
parser = argparse.ArgumentParser(description='Run Machine Learning on Time-Based Wikidata Network')
parser.add_argument('data_dir', help="The directory of the source files for machine learning", type=str)
parser.add_argument('-g', '--gs_treat', help='Use the original TREATS edge rather than that from the gold standard', action='store_true')
parser.add_argument('-a', '--alpha', help="Set the alpha value for the ElasticNet Regression", type=float, default=0.1)
parser.add_argument('-w', '--weight', help="Set the weight factor for DWPC calculations", type=float, default=0.4)
parser.add_argument('-m', '--multiplier', help="Multiplier of number positives for number of negative examples, to be selected for training", type=int, default=10)
parser.add_argument('-s', '--scoring', help='Scoring metric to use for ElasticNet regression', type=str, default='recall')
parser.add_argument('-n', '--num_folds', help='The number of folds for the Cross Validation', type=int, default=5)
parser.add_argument('-e', '--seed', help='Seed for random split between folds', type=int, default=0)
parser.add_argument('-c', '--comp_xval', help='Use a Cross-Validation method where holdouts are based on drugs rather than indication', action='store_true')
args = parser.parse_args()

## Define variables that will govern the network analysis
data_dir = args.data_dir
alpha = args.alpha
scoring = args.scoring
negative_multiplier = args.multiplier
gs_treat = args.gs_treat
w = args.weight
comp_xval = args.comp_xval
n_folds = args.num_folds
seed = args.seed
if scoring.lower() == 'none':
    scoring = None

# Convert load dir into an integer for a consistent random seed
ini_seed = int(hashlib.sha1(data_dir.encode()).hexdigest(), 16) % 2**16

# Out dir is based on this filename
out_dir = os.path.join('../2_pipeline/', sys.argv[0].split('.')[0], 'out')

# Test params will be prepended to any output files if they differ from defaults
test_params = ''

for k in sorted(list(vars(args).keys())):
    if k == 'data_dir':
        continue
    v = vars(args)[k]
    if v != parser.get_default(k):
        test_params += '{}-{}_'.format(k, v)
print('Non-default testing params: {}'.format(test_params))

n_jobs = 32

# Make sure the save directory exists, if not, make it
try:
    os.stat(out_dir)
except:
    os.makedirs(out_dir)

# Read input files
nodes = gt.remove_colons(pd.read_csv(os.path.join(data_dir, 'nodes.csv')))
edges = gt.remove_colons(pd.read_csv(os.path.join(data_dir, 'edges.csv')))

comp_ids = set(nodes.query('label == "Compound"')['id'])
dis_ids = set(nodes.query('label == "Disease"')['id'])

# Get Indications ready
#indications = pd.read_csv(os.path.join('../2_pipeline/01_get_gold_standard/out/', 'gold_standard.csv'))
#gs_edges = (indications.rename(columns={'comp_wd':'start_id', 'disease_wd': 'end_id'})
#              [['start_id', 'end_id']])
## Make sure indications are actually in the Graph
#gs_edges = gs_edges.query('start_id in @comp_ids and end_id in @dis_ids').reset_index(drop=True)
#print('Usable gold standard edges: {}'.format(len(gs_edges)))

# For now we will just use the TREATS edges in the network as GS edges
gs_edges = edges.query('type == "TREATS_CtD"').reset_index(drop=True)

# Just look at compounds and diseases in the gold standard
compounds = gs_edges['start_id'].unique().tolist()
diseases = gs_edges['end_id'].unique().tolist()

print('Based soley on gold standard...')
print('{:,} Compounds * {:,} Diseases = {:,} CD Pairs'.format(len(compounds),
                                                              len(diseases), len(compounds)*len(diseases)))
# Add in some other edges... anything with a degree > 1... or a CtD edge
compounds = set(compounds)
diseases = set(diseases)

print('Adding some more compounds and diseases....')
# Do some magic to find nodes with degree > 1
mg = MatrixFormattedGraph(nodes, edges)
first_comp = nodes.query('label == "Compound"')['id'].iloc[0]
first_disease = nodes.query('label == "Disease"')['id'].iloc[0]
comp_degrees = mg.extract_degrees(end_nodes=[first_disease])
comp_degrees = comp_degrees.loc[:, ['compound_id']+[c for c in comp_degrees.columns if c.startswith('C')]]
comp_degrees['total'] = comp_degrees[[c for c in comp_degrees.columns if c.startswith('C')]].sum(axis=1)
dis_degrees = mg.extract_degrees(start_nodes=[first_comp])
dis_degrees = dis_degrees.loc[:, ['disease_id']+[c for c in dis_degrees.columns if c.startswith('D')]]
dis_degrees['total'] = dis_degrees[[c for c in dis_degrees.columns if c.startswith('D')]].sum(axis=1)

compounds.update(set(comp_degrees.query('total > 1')['compound_id']))
diseases.update(set(dis_degrees.query('total > 1')['disease_id']))

compounds = list(compounds)
diseases = list(diseases)

print('Now comps and diseases')
print('{:,} Compounds * {:,} Diseases = {:,} CD Pairs'.format(len(compounds),
                                                              len(diseases), len(compounds)*len(diseases)))

# look at all compounds and diseases for now
#compounds = comp_ids
#diseases = dis_ids


# Ensure all the compounds and diseases actually are of the correct node type and in the network
node_kind = nodes.set_index('id')['label'].to_dict()
compounds = [c for c in compounds if c in comp_ids]
diseases = [d for d in diseases if d in dis_ids]

if not gs_treat:
    print('Using the original TREATS edge from Wikidata')
else:
    print('Removing Wikidata TREATS edges and repalcing with those from Gold Standard')

    def drop_edges_from_list(df, drop_list):
        idx = df.query('type in @drop_list').index
        df.drop(idx, inplace=True)

    # Filter out any compounds and diseases wrongly classified
    gs_edges = gs_edges.query('start_id in @compounds and end_id in @diseases')
    # Remove the TREATs edge form edges
    drop_edges_from_list(edges, ['TREATS_CtD'])
    gs_edges['type'] = 'TREATS_CtD'

    column_order = edges.columns
    edges = pd.concat([edges, gs_edges], sort=False)[column_order].reset_index(drop=True)


print('{:,} Nodes'.format(len(nodes)))
print('{:,} Edges'.format(len(edges)))

print('{:,} Compounds * {:,} Diseases = {:,} CD Pairs'.format(len(compounds),
                                                              len(diseases), len(compounds)*len(diseases)))


def remove_edges_from_gold_standard(to_remove, gs_edges):
    """
    Remove edges from the gold standard
    """
    remove_pairs = set([(tup.c_id, tup.d_id) for tup in to_remove.itertuples()])
    gs_tups = set([(tup.start_id, tup.end_id) for tup in gs_edges.itertuples()])

    remaining_edges = gs_tups - remove_pairs

    return pd.DataFrame({'start_id': [tup[0] for tup in remaining_edges],
                         'end_id': [tup[1] for tup in remaining_edges],
                         'type': 'TREATS_CtD'})

def add_percentile_column(in_df, group_col, new_col, cdst_col='prediction'):

    grpd = in_df.groupby(group_col)
    predict_dfs = []

    for grp, df1 in grpd:
        df = df1.copy()

        total = df.shape[0]

        df.sort_values(cdst_col, inplace=True)
        order = np.array(df.reset_index(drop=True).index)

        percentile = (order+1) / total
        df[new_col] = percentile

        predict_dfs.append(df)

    return pd.concat(predict_dfs, sort=False)

def glmnet_coefs(glmnet_obj, X, f_names):
    """Helper Function to quickly return the model coefs and correspoding fetaure names"""
    l = glmnet_obj.lambda_best_[0]

    coef = glmnet_obj.coef_[0]
    coef = np.insert(coef, 0, glmnet_obj.intercept_)

    names = np.insert(f_names, 0, 'intercept')

    z_intercept = coef[0] + sum(coef[1:] * X.mean(axis=0))
    z_coef = coef[1:] * X.values.std(axis=0)
    z_coef = np.insert(z_coef, 0, z_intercept)

    return pd.DataFrame([names, coef, z_coef]).T.rename(columns={0:'feature', 1:'coef', 2:'zcoef'})

num_pos = len(gs_edges)
num_neg = negative_multiplier*num_pos

# Make a DataFrame for all compounds and diseases
# Include relevent compund info and treatment status (ML Label)
cd_df = pd.DataFrame(list(product(compounds, diseases)), columns=['c_id', 'd_id'])
id_to_name = nodes.set_index('id')['name'].to_dict()
cd_df['c_name'] = cd_df['c_id'].apply(lambda i: id_to_name[i])
cd_df['d_name'] = cd_df['d_id'].apply(lambda i: id_to_name[i])

merged = pd.merge(cd_df, gs_edges, how='left', left_on=['c_id', 'd_id'], right_on=['start_id', 'end_id'])
merged['status'] = (~merged['start_id'].isnull()).astype(int)
cd_df = merged.loc[:, ['c_id', 'c_name', 'd_id', 'd_name', 'status']]

cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

# Set up training and testing for fold CV
pos_idx = cd_df.query('status == 1').index
neg_idx = cd_df.query('status == 0').sample(n=num_neg, random_state=ini_seed*(seed+1)).index
# Trining set is all indictations and randomly selected negatives
train_idx = pos_idx.union(neg_idx)

# Different train-test split paradigm, based on compounds/n_folds for each test set
# Blinds the model to a particular compound for each fold of the cv
if comp_xval:
    # Needs to be array for indexing operations
    compounds = np.array(compounds)
    # set up dummy y-vals
    y = np.repeat(1, len(compounds))

    for i, (train, test) in enumerate(cv.split(compounds, y)):
        # Subset the compounds for this fold
        test_comps = compounds[test]

        # Testing and trainnig are broken up by compound:
        # Both positives and negatives in test set will be pulled from same compounds
        pos_holdout_idx = cd_df.query('status == 1 and c_id in @test_comps').index
        neg_holdout_idx = cd_df.loc[neg_idx].query('c_id in @test_comps').index

        print('Fold {}: {} Test Postives, {} Test negatives: {} ratio'.format(
                i, len(pos_holdout_idx), len(neg_holdout_idx), len(neg_holdout_idx)/len(pos_holdout_idx)))

        holdout_idx = pos_holdout_idx.union(neg_holdout_idx)
        cd_df.loc[holdout_idx, 'holdout_fold'] = i

else:
    train_df = cd_df.loc[train_idx]

    # Dummy xvalues and true results for y values used
    X = np.zeros(len(train_idx))
    y = train_df['status'].values

    # Just split indications randomly so equal number of pos and neg examples in each fold
    for i, (train, test) in enumerate(cv.split(X, y)):
        holdout_idx = train_df.iloc[test].index
        cd_df.loc[holdout_idx, 'holdout_fold'] = i

target_edges = edges.query('type == "TREATS_CtD"').copy()
other_edges = edges.query('type != "TREATS_CtD"').copy()

coefs = []
probas = []

for i in range(n_folds):

    print('Beginning fold {}'.format(i))

    fold_train_idx = cd_df.loc[train_idx].query('holdout_fold != @i').index
    to_remove = cd_df.query('holdout_fold == @i')
    to_keep = remove_edges_from_gold_standard(to_remove, target_edges)

    edges = pd.concat([other_edges, to_keep], sort=False)

    print('Training Positives: {}'.format(cd_df.query('holdout_fold != @i')['status'].sum()))
    print('Testing Positives: {}'.format(cd_df.query('holdout_fold == @i')['status'].sum()))

    # Convert graph to Matrices for ML feature extraction
    mg = MatrixFormattedGraph(nodes, edges, 'Compound', 'Disease', w=w)
    # Extract prior
    prior = mg.extract_prior_estimate('CtD', start_nodes=compounds, end_nodes=diseases)
    prior = prior.rename(columns={'compound_id': 'c_id', 'disease_id':'d_id'})
    # Extract degree Features
    degrees = mg.extract_degrees(start_nodes=compounds, end_nodes=diseases)
    degrees = degrees.rename(columns={'compound_id': 'c_id', 'disease_id':'d_id'})
    degrees.columns = ['degree_'+c if '_id' not in c else c for c in degrees.columns]
    # Generate blacklisted features and drop
    blacklist = mg.generate_blacklist('CtD')
    degrees.drop([b for b in blacklist if b.startswith('degree_')], axis=1, inplace=True)
    # Extract Metapath Features (DWPC)
    mp_blacklist = [b.split('_')[-1] for b in blacklist]
    mps = [mp for mp in mg.metapaths.keys() if mp not in mp_blacklist]
    dwpc = mg.extract_dwpc(metapaths=mps, start_nodes=compounds, end_nodes=diseases, n_jobs=n_jobs)
    dwpc = dwpc.rename(columns={'compound_id': 'c_id', 'disease_id':'d_id'})
    dwpc.columns = ['dwpc_'+c if '_id' not in c else c for c in dwpc.columns]

    # Merge extracted features into 1 DataFrame
    print('Merging Features...')
    feature_df = pd.merge(cd_df, prior, on=['c_id', 'd_id'], how='left')
    feature_df = pd.merge(feature_df, degrees, on=['c_id', 'd_id'], how='left')
    feature_df = pd.merge(feature_df, dwpc, on=['c_id', 'd_id'], how='left')

    features = [f for f in feature_df.columns if f.startswith('degree_') or f.startswith('dwpc_')]
    degree_features = [f for f in features if f.startswith('degree_')]
    dwpc_features = [f for f in features if f.startswith('dwpc_')]

    # Transform Features
    dt = DegreeTransform()
    dwpct = DWPCTransform()

    X_train = feature_df.loc[fold_train_idx, features].copy()
    y_train = feature_df.loc[fold_train_idx, 'status'].copy()

    print('Transforming Degree Features')
    X_train.loc[:, degree_features] = dt.fit_transform(X_train.loc[:, degree_features])
    print('Tranforming DWPC Features')
    X_train.loc[:, dwpc_features] = dwpct.fit_transform(X_train.loc[:, dwpc_features])

    # Train our ML Classifer (ElasticNet Logistic Regressor)
    print('Training Classifier...')
    classifier = LogitNet(alpha=alpha, n_jobs=n_jobs, min_lambda_ratio=1e-8, n_lambda=150, standardize=True,
                  random_state=(ini_seed+1)*(seed+1), scoring=scoring)

    classifier.fit(X_train, y_train)

    coefs.append(glmnet_coefs(classifier, X_train, features))

    print('Positivie Coefficients: {}\nNegative Coefficitents: {}'.format(len(coefs[i].query('coef > 0')), len(coefs[i].query('coef < 0'))))

    # Get probs for all pairs
    print('Beginning extraction of all probabilities')
    print('Transforming all features...')
    feature_df.loc[:, degree_features] = dt.transform(feature_df.loc[:, degree_features])
    feature_df.loc[:, dwpc_features] = dwpct.transform(feature_df.loc[:, dwpc_features])

    print('Calculating Probabilities')
    all_probas = classifier.predict_proba(feature_df.loc[:, features])[:, 1]
    probas.append(all_probas)

# Finish the probability data and save to disk
for i in range(n_folds):
    cd_df['probas_{}'.format(i)] = probas[i]

for i in range(n_folds):
    cd_df = add_percentile_column(cd_df, group_col='c_id', new_col='c_percentile_{}'.format(i), cdst_col='probas_{}'.format(i))
    cd_df = add_percentile_column(cd_df, group_col='d_id', new_col='d_percentile_{}'.format(i), cdst_col='probas_{}'.format(i))

cd_df.to_csv(os.path.join(out_dir, test_params+'predictions.csv'), index=False)

# Merge the Coefficient data and save to disk
for i in range(n_folds):
    coefs[i] = coefs[i].set_index('feature')
    coefs[i].columns = [l+'_{}'.format(i) for l in coefs[i].columns]
coefs = pd.concat(coefs, axis=1).reset_index()

coefs.to_csv(os.path.join(out_dir, test_params+'model_coefficients.csv'), index=False)

