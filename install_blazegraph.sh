#!/bin/bash

RELEASE='2_1_4'

# Download Blazegraph
git clone -b BLAZEGRAPH_RELEASE_$RELEASE --single-branch https://github.com/blazegraph/database.git BLAZEGRAPH_RELEASE_$RELEASE

# Build Blazegraph
cd BLAZEGRAPH_RELEASE_$RELEASE
./scripts/mavenInstall.sh
