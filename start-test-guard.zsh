#!/bin/zsh

if [ "$#" -lt 2 ]; then
    echo "Użycie: $0 <czas> <nazwa_testu> [liczba_uzytkownikow] [cautious] [active] [bot]"
    exit 1
fi

export SIMULATION_DURATION=$1
export TEST_NAME=$2
export NUM_USERS=${3:-200}
CAUTIOUS=${4:-50}
ACTIVE=${5:-45}
BOT=${6:-5}

export PCT_CAUTIOUS=$CAUTIOUS
export PCT_ACTIVE=$ACTIVE
export PCT_BOT=$BOT

echo "Nazwa: $TEST_NAME"
echo "Czas: $SIMULATION_DURATION s"
echo "Liczba użytkowników: $NUM_USERS"
echo "Rozkład: CAUTIOUS=$PCT_CAUTIOUS%, ACTIVE=$PCT_ACTIVE%, BOT=$PCT_BOT%"

docker compose -f docker-compose-redis.yml down -v
docker compose -f docker-compose-redis.yml up -d --build

if [ $? -ne 0 ]; then
    docker compose -f docker-compose-redis.yml down 
    exit 1
fi

echo "Oczekiwanie $SIMULATION_DURATION sekund..."
sleep $SIMULATION_DURATION

echo "Zatrzymywanie kontenerów..."
docker compose -f docker-compose-redis.yml down 

./pull-logs.zsh "$TEST_NAME"