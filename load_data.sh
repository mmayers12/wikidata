#!/bin/bash

if [ $1 ]; then
    RELEASE=$1
else
    RELEASE='2_1_4'
fi

cd 'BLAZEGRAPH_RELEASE_'$RELEASE

# Begin process of loading data: run in backgroun, output to dataload.log
nohup ./scripts/dataLoader.sh -format Turtle build.properties ../data/wikidata-20170116-all-BETA.ttl.gz > ../dataload.log &
