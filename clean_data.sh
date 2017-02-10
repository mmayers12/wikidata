#!/bin/bash

DATAFILE=`ls data/*.ttl.gz`

mkdir data/split
cd db

# Begin Process of cleaning the data
# Script will remvoe non-english entries, split into managable sized files
# And remove sitelinks
nohup ./munge.sh -f ../$DATAFILE -d ../data/split -l en -s &> ../dataclean.log &

