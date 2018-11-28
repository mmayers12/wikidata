#!/bin/bash

DATAFILE=`ls 0_data/external/*.ttl.gz`

mkdir 0_data/manual
cd db

# Begin Process of cleaning the data
# Script will remvoe non-english entries, split into managable sized files
# And remove sitelinks
nohup ./munge.sh -f ../$DATAFILE -d ../0_data/manual -l en -s &> ../dataclean.log &

