#!/bin/bash

if [ $1 ]; then
    DATE=$1
else
    DATE='20181112'
fi

mkdir -p 0_data/external
cd 0_data/external
nohup wget 'https://dumps.wikimedia.org/wikidatawiki/entities/'$DATE'/wikidata-'$DATE'-all-BETA.ttl.gz' --progress=bar:force:noscroll &> ../../download.log &
