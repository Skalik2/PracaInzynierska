#!/bin/zsh

FILE="params.txt"

if [[ ! -f "$FILE" ]]; then
    echo "Plik $FILE nie istnieje."
    exit 1
fi

while IFS= read -r cmd || [[ -n "$cmd" ]]; do
    [[ -z "$cmd" || "$cmd" =~ ^# ]] && continue

    echo "Uruchamiam: $cmd"
    eval "$cmd" < /dev/null

    if [[ $? -ne 0 ]]; then
        echo "Błąd podczas wykonywania: $cmd"
    fi
    
    sleep 10
done < "$FILE"