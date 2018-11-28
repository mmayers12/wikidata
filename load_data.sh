#!/bin/bash

# Start Blazegraph
cd db
nohup ./runBlazegraph.sh > ../db.log &

# Get the data directory
pushd ../0_data/manual > /dev/null
DATA_DIR=`pwd`
popd > /dev/null

# Begin the dataload process
sleep 25
nohup ./loadRestAPI.sh -n wdq -d $DATA_DIR &> ../dataload.log &
