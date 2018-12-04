#!/bin/bash

if [ $1 ]; then
    TARGET_DIR=$1
else
    TARGET_DIR='../'
fi

# Get the latest version of wikidata-query-rdf
pushd $TARGET_DIR > /dev/null
git clone  --recurse-submodules https://gerrit.wikimedia.org/r/wikidata/query/rdf wikidata-query-rdf
cd wikidata-query-rdf
mvn package

# Get the filename
FILE_NAME=$(ls dist/target/ | grep SNAPSHOT-dist.zip)

# Move into this directory
popd > /dev/null
cp $(TARGET_DIR)wikidata-query-rdf/dist/target/$(FILE_NAME) .


# Unzip the archive containing databse and helper scripts
unzip $FILE_NAME

DIR_NAME=$(echo $FILE_NAME | rev | cut -c 5- | rev)
mv DIR_NAME db
cd db

# Update the web.xml file so that there are not query timeouts
# TODO Update this so that it grabs same FILENAME from above
jar xvf blazegraph-service-0.3.1-SNAPSHOT.war WEB-INF/web.xml
sed -i -e 's/600000/0/g' WEB-INF/web.xml
jar uvf blazegraph-service-0.3.1-SNAPSHOT.war WEB-INF

# Run Blazegraph for the first time to generate config files
./runBlazegraph.sh &> ../firstrun.log & export PID=$!

# Number of seconds to wait
WAIT_SECONDS=25

# Counter to keep track of how many seconds have passed
count=0

while [ $count -lt $WAIT_SECONDS ]; do
    sleep 1
    ((count++))
done

kill $PID
echo done
