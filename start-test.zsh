#!/bin/zsh

# Sprawdzenie wymaganych argumentów
if [ "$#" -lt 2 ]; then
    echo "Użycie: $0 <czas_trwania> <nazwa_testu> [liczba_użytkowników] [cautious] [active] [bot]"
    exit 1
fi

# Przypisanie argumentów do zmiennych
TEST_DURATION=$1
TEST_NAME=$2

# Eksportowanie zmiennych środowiskowych dla Docker Compose
export SIMULATION_DURATION=$TEST_DURATION
export NUM_USERS=${3:-200}
CAUTIOUS=${4:-50}
ACTIVE=${5:-45}
BOT=${6:-5}

export PCT_CAUTIOUS=$CAUTIOUS
export PCT_ACTIVE=$ACTIVE
export PCT_BOT=$BOT

# Wyświetlanie informacji o teście
echo "Nazwa: $TEST_NAME"
echo "Czas: $TEST_DURATION s"
echo "Liczba użytkowników: $NUM_USERS"
echo "Rozkład: CAUTIOUS_USER=$PCT_CAUTIOUS%, ACTIVE_TRADER=$PCT_ACTIVE%, SCRAPER_BOT=$PCT_BOT%"

# Czyszczenie poprzednich wolumenów i kontenerów
docker compose -f docker-compose.yml down -v

# Uruchamianie kontenerów w tle z przebudowaniem obrazów
docker compose -f docker-compose.yml up -d --build

# Obsługa błędu uruchomienia
if [[ $? -ne 0 ]]; then
    echo "Błąd podczas uruchamiania docker-compose. Zamykanie..."
    docker compose -f docker-compose.yml down
    exit 1
fi

echo "Oczekiwanie $TEST_DURATION sekund na zakończenie symulacji..."
sleep $TEST_DURATION

echo "Zatrzymywanie kontenerów..."
docker compose -f docker-compose.yml down

# Wywołanie skryptu pobierającego logi (pamiętaj o zmianie rozszerzenia na .zsh jeśli je zmieniłeś)
if [[ -f "./pull-logs.zsh" ]]; then
    ./pull-logs.zsh "$TEST_NAME"
else
    ./pull-logs.sh "$TEST_NAME"
fi