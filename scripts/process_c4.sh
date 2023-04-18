#!/bin/bash
TARGET_DIR=/home/wk247/data/c4/en-processed

for JSON_FILE in $(ls /home/wk247/data/c4/en-decompressed/*.json)
do
    FILE_NAME=$(basename $JSON_FILE)

    if [[ $FILE_NAME =~ (.*).json$ ]]; then
        NEW_FILE_NAME=${BASH_REMATCH[1]}
	fi

    if [ -f "$TARGET_DIR/$NEW_FILE_NAME" ]; 
    then
        echo "$NEW_FILE_NAME already exists, skipping"
    else
        echo "Processing $JSON_FILE"
        # parse json
        jq -M -r '.["text"] | gsub("[\\n\\t]"; " ")' $JSON_FILE > $TARGET_DIR/$NEW_FILE_NAME
    fi 
done
