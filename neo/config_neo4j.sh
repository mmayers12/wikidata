#!/bin/bash

# User Parameters
VERSION='community-3.1.3'
NETWORK='wikidata-v0.1'
NUM_PERM=5

# Download Parameters
BASE_NAME='neo4j-'$VERSION
DOWNLOAD=$BASE_NAME'-unix.tar.gz'
DL_URL='http://neo4j.com/artifact.php?name='$DOWNLOAD

# Output names
NEO_NAME=$BASE_NAME'_'$NETWORK
PERM_NAME=$NEO_NAME'_perm-'

# Download zip
wget $DL_URL -O $DOWNLOAD

# Unzip and rename
tar -zxvf $DOWNLOAD
mv $BASE_NAME $NEO_NAME

# loop through each permutation
for i in $(seq 1 $NUM_PERM); do
    # Copy to permuted Directory
    cp -r $NEO_NAME $PERM_NAME$i

    # Edit config file
    CONFIG_FILE=$PERM_NAME$i'/conf/neo4j.conf'

    sed -i -e 's/#dbms.security.auth_enabled=false/dbms.security.auth_enabled=false/g' $CONFIG_FILE
    sed -i -e 's/#dbms.connector.bolt.listen_address=:7687/dbms.connector.bolt.listen_address=0.0.0.0:769'$i'/g' $CONFIG_FILE
    sed -i -e 's/#dbms.connector.http.listen_address=:7474/dbms.connector.http.listen_address=0.0.0.0:750'$i'/g' $CONFIG_FILE
    sed -i -e 's/dbms.connector.https.enabled=true/dbms.connector.https.enabled=false/g' $CONFIG_FILE
done

# Change config on original network
CONFIG_FILE=$NEO_NAME'/conf/neo4j.conf'

sed -i -e 's/#dbms.security.auth_enabled=false/dbms.security.auth_enabled=false/g' $CONFIG_FILE
sed -i -e 's/#dbms.connector.bolt.listen_address=:7687/dbms.connector.bolt.listen_address=0.0.0.0:7690/g' $CONFIG_FILE
sed -i -e 's/#dbms.connector.http.listen_address=:7474/dbms.connector.http.listen_address=0.0.0.0:7500/g' $CONFIG_FILE
sed -i -e 's/dbms.connector.https.enabled=true/dbms.connector.https.enabled=false/g' $CONFIG_FILE

