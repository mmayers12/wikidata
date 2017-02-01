# Compound Repositioning Uitilizing WikiData as a Backend

This repo will be focused on WikiData and the potential for using this graph
database to create the backend for a learning algorithm to dtermine possible
repositioning candidataes.

## Requirments

### To compile blazegraph

- `javac` - java compiler, included in default-jdk in ubuntu
- `mvn` - apache maven

### To run Blazegraph

- Java 7 or greater

## Installation Instructions

### Data Download

WikiData dump file was downloaded using `download_data.sh` to the directory `data/`
The data file was approx 11 GB and took around 90 min to download.

### Database Installation

Blazegraph can be installed using the `install_blazegraph.sh` script. The latest
version `2_1_4` will be downloaded and built to the directory `BLAZEGRAPH_RELEASE_2_1_4`
Different release versions can be by running `install_blazegraph.sh $VERSION`

### Data Load

The `load_data.sh` script will load the wikidata dump file into blazegraph as a .jnl
file.  This journal file is > 200 GB (not sure how big yet, as has yet to complete
successfully).  This will also require a similar amount of space in the /tmp directory
