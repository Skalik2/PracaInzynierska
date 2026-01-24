#!/bin/bash

if [ "$#" -lt 2 ]; then
    exit 1
fi

TEST_DURATION=$1
TEST_NAME=$2

export SIMULATION_DURATION=$TEST_DURATION
export NUM_USERS=${3:-200}
CAUTIOUS=${4:-50}
ACTIVE=${5:-45}
BOT=${6:-5}

export PCT_CAUTIOUS=$CAUTIOUS
export PCT_ACTIVE=$ACTIVE
export PCT_BOT=$BOT

echo "Nazwa: $TEST_NAME"
echo "Czas: $TEST_DURATION s"
echo "Liczba użytkowników: $NUM_USERS"
echo "Rozkład: CAUTIOUS_USER=$PCT_CAUTIOUS%, ACTIVE_TRADER=$PCT_ACTIVE%, SCRAPER_BOT=$PCT_BOT%"

docker compose -f docker-compose-redis.yml down -v
docker compose -f docker-compose-redis.yml up  -d --build

if [ $? -ne 0 ]; then
    docker compose -f docker-compose-redis.yml down 
    exit 1
fi

echo "Oczekiwanie $TEST_DURATION sekund"
sleep $TEST_DURATION

echo "Zatrzymywanie kontenerów"
docker compose -f docker-compose-redis.yml down 

./pull-logs.sh "$TEST_NAME"
