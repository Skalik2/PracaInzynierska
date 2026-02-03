#!/bin/zsh

# Ustawienie przerywania skryptu w przypadku błędu
set -e

# Kolory dla lepszej czytelności (macOS zsh)
cyan='\033[0;36m'
yellow='\033[1;33m'
green='\033[0;32m'
red='\033[0;31m'
nc='\033[0m' # No Color

echo "${cyan}Tworzenie wirtualnego środowiska venv...${nc}"
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
else
    echo "Środowisko 'venv' już istnieje."
fi

echo "${cyan}Aktywacja wirtualnego środowiska...${nc}"
source ./venv/bin/activate

echo "${cyan}Instalacja zależności z requirements.txt...${nc}"
pip install -r requirements.txt > /dev/null

# Znajdowanie pliku SQL (szuka w ./main i w folderze bieżącym)
sqlFilePath=$(find . -path "./main/*.sql" -o -name "*.sql" -maxdepth 2 | head -n 1)

if [[ -z "$sqlFilePath" ]]; then
    echo "${red}Błąd: Nie znaleziono pliku SQL!${nc}"
    exit 1
fi

# Wyciągnięcie nazwy pliku bez rozszerzenia dla folderu wyjściowego
sqlFileName=$(basename "$sqlFilePath")
outputDir="${sqlFileName%.*}"

echo "${cyan}Znaleziono plik SQL: $sqlFilePath${nc}"
echo "${cyan}Wyniki zostaną zapisane w folderze: $outputDir${nc}"

echo "\n${yellow}Konwersja SQL do CSV${nc}"
python3 ./sqlToCsv.py --file "$sqlFilePath" --out "$outputDir"

echo "\n${yellow}Tworzenie listy użytkowników${nc}"
python3 ./generate_user_classes.py --dir "./$outputDir"

echo "\n${yellow}Przygotowanie danych${nc}"
python3 ./dataPrep.py --dir "./$outputDir"

echo "\n${yellow}Główna analiza${nc}"
python3 ./mainAnalysis.py --dir "./$outputDir"

echo "\n${yellow}Analiza pojedynczego scenariusza${nc}"
python3 ./single_scenario_analysis.py --dir "./$outputDir"

echo "\n${yellow}Analiza zablokowanych użytkowników${nc}"
python3 ./analyze_blocked_users.py --dir "./$outputDir"

echo "\n${yellow}Analiza Macierzy Pomyłek${nc}"
python3 ./analyze_confusion_matrix.py --dir "./$outputDir"

echo "\n${green}ZAKOŃCZONO${nc}"