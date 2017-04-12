VERSION="neo4j-community-3.1.3"

DB_NAME="wikidata-v0.1"

declare -a suffixes=("" "_perm-1" "_perm-2" "_perm-3" "_perm-4" "_perm-5")

for suffix in "${suffixes[@]}"; do

    BIN_LOC="neo/${VERSION}_$DB_NAME$suffix/bin/neo4j-admin"

    CSV_NAME="nbs/data/permuted/hetnet_basedges$suffix"

    REPORT_NAME="network${suffix}_import.report"

    ./$BIN_LOC import --nodes "nbs/data/hetnet_basenodes.csv" --relationships "${CSV_NAME}.csv" --report-file="$REPORT_NAME"

done
