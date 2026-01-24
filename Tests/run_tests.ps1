# Ustawienie przerywania skryptu w przypadku błędu
$ErrorActionPreference = "Stop"

# Tworzenie wirtualnego środowiska
if (-not (Test-Path "venv")) {
    Write-Host "Tworzenie wirtualnego srodowiska venv"
    python -m venv venv
} else {
    Write-Host "Środowisko 'venv' już istnieje."
}

# Aktywacja środowiska
Write-Host "Aktywacja wirtualnego srodowiska..."
. .\venv\Scripts\Activate.ps1

# Instalacja zależności
Write-Host "Instalacja zaleznosci z requirements.txt"
pip install -r requirements.txt | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "Błąd instalacji zależności!"; exit 1 }

# Znajdowanie pliku SQL
$sqlFile = Get-ChildItem -Path ".\main", "." -Filter "*.sql" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1

if (-not $sqlFile) {
    Write-Error "Nie znaleziono pliku SQL!"
    exit 1
}

$sqlFilePath = $sqlFile.FullName
$outputDir = $sqlFile.BaseName 

Write-Host "Znaleziono plik SQL: $sqlFilePath" -ForegroundColor Cyan
Write-Host "Wyniki zostaną zapisane w folderze: $outputDir" -ForegroundColor Cyan

Write-Host "`nKonwersja SQL do CSV" -ForegroundColor Yellow
python .\sqlToCsv.py --file "$sqlFilePath" --out "$outputDir"

Write-Host "`nTworzenie listy użytkowników" -ForegroundColor Yellow
python .\generate_user_classes.py --dir ".\$outputDir"

Write-Host "`nPrzygotowanie danych" -ForegroundColor Yellow
python .\dataPrep.py --dir ".\$outputDir"

Write-Host "`nGlowna analiza" -ForegroundColor Yellow
python .\mainAnalysis.py --dir ".\$outputDir"

Write-Host "`nAnaliza pojedynczego scenariusza" -ForegroundColor Yellow
python .\single_scenario_analysis.py --dir ".\$outputDir"

Write-Host "`nAnaliza zablokowanych uzytkownikow" -ForegroundColor Yellow
python .\analyze_blocked_users.py --dir ".\$outputDir"

Write-Host "`nAnaliza Macierzy Pomyłek" -ForegroundColor Yellow
python .\analyze_confusion_matrix.py --dir ".\$outputDir"

Write-Host "`nZAKONCZONO" -ForegroundColor Green