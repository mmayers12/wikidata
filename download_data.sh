#!/bin/bash

if [ $1 ]; then
    DATE=$1
else
    DATE='20170130'
fi

mkdir data
cd data
nohup wget 'https://dumps.wikimedia.org/wikidatawiki/entities/'$DATE'/wikidata-'$DATE'-all-BETA.ttl.gz' --progress=bar:force:noscroll &> ../download.log &
