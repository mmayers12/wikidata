#!/bin/bash

# Start Blazegraph
cd db
nohup ./runBlazegraph.sh > db.log &

# Get the data directory
pushd ../data/split > /dev/null
DATA_DIR=`pwd`
popd > /dev/null

# Begin the dataload process
nohup ./loadRestAPI.sh -n wdq -d $DATA_DIR &> dataload.log &
