#!/bin/bash

set -e

if [ "$#" -eq 1 ]; then
  output_file="$1.sql"
else
  exit 1
fi

docker compose -f docker-compose.yml up -d logdatabase

COUNTER=0
until docker compose -f docker-compose.yml exec -T logdatabase pg_isready -U postgres -d gielda 2>/dev/null || [ $COUNTER -eq 30 ]; do
  sleep 1
  ((COUNTER++))
done

if [ $COUNTER -eq 30 ]; then
    docker compose -f docker-compose.yml down
    exit 1
fi

container_id=$(docker compose -f docker-compose.yml ps -q logdatabase)

if [ -z "$container_id" ]; then
  docker compose -f docker-compose.yml down
  exit 1
fi

docker exec -t "$container_id" pg_dump -U postgres -d gielda > "$output_file"

if [ $? -eq 0 ]; then
    echo "Database dump successful! The SQL file is saved as $output_file."
else
    docker compose -f docker-compose.yml down
    exit 1
fi

docker compose -f docker-compose.yml stop logdatabase
