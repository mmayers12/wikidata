#!/bin/bash

cd db

nohup ./runBlazegraph.sh &> ../db.log &
sleep 10
nohup ./runUpdate.sh -n wdq -l en -s &> ../update.log &
