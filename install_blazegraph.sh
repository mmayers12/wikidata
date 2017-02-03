#!/bin/bash

# Unzip the archive containing databse and helper scripts
unzip service-0.2.4-SNAPSHOT-dist.zip
mv service-0.2.4-SNAPSHOT db

# Run Blazegraph for the first time to generate config files
cd db
./runBlazegraph.sh &> ../firstrun.log &

# Number of seconds to wait
WAIT_SECONDS=25

# Counter to keep track of how many seconds have passed
count=0

while [ $count -lt $WAIT_SECONDS ]; do
    sleep 1
    ((count++))
done

USER=`whoami`
PID=`pgrep java -u $USER`
kill $PID
echo done
