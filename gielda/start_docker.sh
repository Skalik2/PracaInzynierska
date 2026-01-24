#!/bin/bash

# Check if the correct number of parameters have been passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <number_of_containers> <number_of_companies_per_container> <number_of_cache_elem_per_company> <cashe_time_in_sec>"
    exit 1
fi

NUM_CONTAINERS=$1
NUM_COMPANIES_PER_CONTAINER=$2
NUM_OF_CASHE=$3
CASHE_TIME=$4

# Generate the header of docker-compose.yml
cat <<EOL > docker-compose.generated.yml
version: "3.8"

services:
  database:
    image: postgres:13
    environment:
      POSTGRES_DB: \${DB_NAME}
      POSTGRES_USER: \${DB_USERNAME}
      POSTGRES_PASSWORD: \${DB_PASSWORD}
    ports:
      - "\${DB_PORT}:5432"
    volumes:
      - db_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 4g
        reservations:
          cpus: "0.5"
          memory: 2g

  gielda_service:
    build:
      context: .
      dockerfile: Dockerfile.gielda
    depends_on:
      - database
    ports:
      - "3000:3000"
    environment:
      DB_TYPE: \${DB_TYPE}
      DB_HOST: database
      DB_PORT: \${DB_PORT}
      DB_USER: \${DB_USERNAME}
      DB_PASSWORD: \${DB_PASSWORD}
      DB_NAME: \${DB_NAME}
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 4g
        reservations:
          cpus: "0.5"
          memory: 2g
EOL

# Generate trade_service containers
for (( i=1; i<=NUM_CONTAINERS; i++ ))
do
  OFFSET=$(( (i-1) * NUM_COMPANIES_PER_CONTAINER + 1 ))
  COMPANIES_IDS=$(seq -s, $OFFSET $((OFFSET + NUM_COMPANIES_PER_CONTAINER - 1)))

  cat <<EOL >> docker-compose.generated.yml

  trade_service_$i:
    build:
      context: .
      dockerfile: Dockerfile.trade
    depends_on:
      - database
    environment:
      TRADE_ID: $i
      DB_TYPE: \${DB_TYPE}
      DB_HOST: database
      DB_PORT: \${DB_PORT}
      DB_USER: \${DB_USERNAME}
      DB_PASSWORD: \${DB_PASSWORD}
      DB_NAME: \${DB_NAME}
      COMPANIES_IDS: "$COMPANIES_IDS"
      NUM_OF_CASHE:  $NUM_OF_CASHE
      CASHE_TIME: $CASHE_TIME
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 4g
        reservations:
          cpus: "0.5"
          memory: 2g
EOL
done

# Add volume definitions
cat <<EOL >> docker-compose.generated.yml

volumes:
  db_data:
EOL

echo "The file docker-compose.generated.yml has been generated."

# Start the containers using docker-compose
docker-compose -f docker-compose.generated.yml up --build

if [ $? -eq 0 ]; then
    echo "Containers have been successfully started."
else
    echo "An error occurred while starting the containers."
fi
