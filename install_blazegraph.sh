#!/bin/bash

RELEASE='2_1_4'

# Download Blazegraph
git clone -b BLAZEGRAPH_RELEASE_$RELEASE --single-branch https://github.com/blazegraph/database.git BLAZEGRAPH_RELEASE_$RELEASE

# Build Blazegraph
cd BLAZEGRAPH_RELEASE_$RELEASE
./scripts/mavenInstall.sh

# Configure the properties to dump to the wikidata.jnl file
echo -e '\n# Persistence Store File\ncom.bigdata.journal.AbstractJournal.file=wikidata.jnl' >> build.properties

# Run DataLoader script to rebuild the database
./scripts/dataLoader.sh -format Turtle build.properties ../data/wikidata-20170116-all-BETA.ttl.gz
