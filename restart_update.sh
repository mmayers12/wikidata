#!/bin/bash

# Get the date and time of the last write to the update file
DATE=`find update.log -maxdepth 0 -printf "%TY-%Tm-%Td--%TH:%TM\n"`

# Kill the updater and backup the log
pkill -f runUpdate
cp -p update.log update-$DATE.log

# Restart the updater
cd db
nohup ./runUpdate.sh -n wdq -l en -s &> ../update.log &
