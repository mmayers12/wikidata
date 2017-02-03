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

Blazegraph can be installed using the `install_blazegraph.sh` script. This will
unzip the zipped archive. This archive was compiled from the [widkidata-qurey-rdf
repository](https://github.com/wikimedia/wikidata-query-rdf), liscenced under the
apache 2.0 liscence. This archive also contains many useful scripts for reducing
the size of the data in the dump and loading into blazegraph.

### Clean Data

The `clean_data.sh` script will split the data into smaller pieces for better
data loading. In the process of splitting, it will remove all non-english
language data and all sitelinks, to reduce data size.

### Data Load

The `load_data.sh` script will split load the wikidata dump file into blazegraph as a .jnl
file.  This journal file is > 200 GB (not sure how big yet, as has yet to complete
successfully).  This will also require a similar amount of space in the /tmp directory


