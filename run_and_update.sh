#!/bin/bash

cd db

nohup ./runBlazegraph.sh -h 0.0.0.0 &> ../db.log &
sleep 10
nohup ./runUpdate.sh -n wdq -l en -s &> ../update.log &
