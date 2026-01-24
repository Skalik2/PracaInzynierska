#!/bin/bash

FILE="params.txt"

if [ ! -f "$FILE" ]; then
    exit 1
fi

while IFS= read -r cmd || [ -n "$cmd" ]; do
    [[ -z "$cmd" || "$cmd" =~ ^# ]] && continue

    eval "$cmd" < /dev/null

    if [ $? -ne 0 ]; then
        echo "Error: $cmd "
    fi
    
    sleep 10

done < "$FILE"