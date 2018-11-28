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

### To run the pipline

- Anaconda with python version 3.7

## Database Installation Instructions

### Data Download

WikiData dump file was downloaded using `download_data.sh` to the directory `0_data/`
The 20181112 data file was approx 46.13G and finshed in 8h 59m.

### Database Installation

Blazegraph can be installed using the `install_blazegraph.sh` script. This will
use git to clone the [widkidata-qurey-rdf
repository](https://github.com/wikimedia/wikidata-query-rdf), to the parent directory, 
build using maven and then copy the resulting scripts to a folder titled `db`.
wikidata-query-rdf is liscenced under the apache 2.0 liscence. This archive also
contains many useful scripts for reducing the size of the data in the dump and
loading into blazegraph.

### Clean Data

The `clean_data.sh` script will split the data into smaller pieces for better
data loading. In the process of splitting, it will remove all non-english
language data and all sitelinks, to reduce data size. (approx 18 hr)

### Data Load

The `load_data.sh` script will load the split wikidata dump file into blazegraph.
The data is stored in wikidata.jnl, which comes to about 413 GB in the current
(20181112) dump of wikidata. (approx 51h 33m 42s)

## Python installation instrucitons

This repo is designed to run with a python3.7 based anaconda enviornment

### Install and activate the anaconda enviornment

    $ conda env create envionment.yml
    $ source activate ml

