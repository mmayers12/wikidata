# Compound Repositioning Uitilizing WikiData as a Backend

This repo will be focused on WikiData and the potential for using this graph
database to create the backend for a learning algorithm to dtermine possible
repositioning candidataes.

## Requirments

- Java 7 or greater

## Data Download

WikiData dump file was downloaded using `download_data.sh` to the directory `data/`
The data file was approx 11 GB and took around 90 min to download.

## Database Installation

Blazegraph can be installed using the `install_blazegraph.sh` script. The latest
version `2_1_4` will be downloaded and built to the directory `BLAZEGRAPH_RELEASE_2_1_4`
Edit the script to install other versions of blazegraph.
