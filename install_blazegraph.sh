#!/bin/bash

if [ $1 ]; then
    RELEASE=$1
else
    RELEASE='2_1_4'
fi

# Download Blazegraph
git clone -b BLAZEGRAPH_RELEASE_$RELEASE --single-branch https://github.com/blazegraph/database.git BLAZEGRAPH_RELEASE_$RELEASE

# Build Blazegraph
cd BLAZEGRAPH_RELEASE_$RELEASE
./scripts/mavenInstall.sh

# Configure the properties to dump to the wikidata.jnl file
echo -e '\n# Persistence Store File\ncom.bigdata.journal.AbstractJournal.file=wikidata.jnl' >> build.properties
