## Instrukcja uruchamiania systemu i testów

System giełdowy może działać w dwóch trybach: **podstawowym** oraz **predykcyjnym** z modułem Guard i Redis. 

### 1. Konfiguracja i uruchamianie - wersja podstawowa

W tym trybie system działa bez mechanizmów wykrywania botów.

Instalacja zależności node:

```
npm i
```

**Konfiguracja kodu:**
Upewnij się, że poniższe linie są **zakomentowane**:

* W pliku `gielda/src/index.ts`:
    ```typescript
    // app.use(securityMiddleware);
    ```

* W pliku `gielda/src/utils/activityMonitor.ts`:
    ```typescript
    // verifyUserActivity(info).catch(err => console.error("Error in bot protection:", err));
    ```

**Uruchomienie testu:**

W pliku **params.txt** wpisz testy do przeprowadzenia:

```bash
./start-test.[sh/zsh] [czas trwania sekundy] [nazwa testu] [liczba użytkowników] [% ACTIVE_TRADER] [% CAUTIOUS_USER] [% SCRAPER_BOT]

./start-test.sh 3600 H1U500-49-50-1 500 49 50 1
#./start-test.sh 3600 H1U500-49-50-1 500 49 50 1 - ten test nie zostanie aktywowany
```

uruchomienie wszystkich testów:

```bash
./run-all.[sh/zsh]
```

### Wersja z modułem predykcyjnym Guard

Dla wersji z modułem Guard należy odkomentować kod z **Konfiguracja kodu**. 

Moduł Guard potrzebuje wag które należy mu **przypisać w pliku bot_guard.py** oraz **umieścić w tym samym katalogu**.
* W pliku `traffic/bot_guard.py`:
```python
    model.load_model("bot_xgboost_model_H1U200-45-45-10.json")
    encoders_data = joblib.load("bot_xgboost_encoders_H1U200-45-45-10.pkl")
```

Przy wpisywaniu testów do params.txt używać ./start-test-**guard**.[sh/zsh]. 
### 2. Przeprowadzanie testów ETL, asocjacja, korelacja

Przenieś wygenerowany plik sql do `/main`, a następnie uruchom 

```bash
run_tests.ps1 #dla Windows
run_tests.zsh #dla Linux/MacOS
```
po przeprowadzonym teście powstanie folder z tą samą nazwą co plik sql z wszystkimi grafami oraz informacjami na temat danego przebiegu testu.

### 3. Szkolenie modeli

W ścieżce `Tests/models` uruchom jeden z poniższych kodów:

```bash
python model_IsolationForest.py --dir [nazwa katalogu danych po ETL] 
python model_OC-SVM.py --dir H1U200-45-45-10
python model_RCE.py --dir H1U200-45-45-10
python model_XGBoost.py --dir H1U200-45-45-10
```

### 4. Wizualizacja PCE

Wizualizacje PCE należy przeprowadzać podmieniając dane w odpowiednich plikach:
`visualize_IsolationForest.py`
`visualize_OCSVM.py`
`visualize_RCE.py`
`visualize_XGBoost.py`

ustawiając lokację np.:

```bash
INPUT_FILE = '../H1U10-50-40-10/merged_data.csv'
MODEL_FILE = './weights/bot_request_modelH1U10-50-40-10.json'
ENCODERS_FILE = './weights/bot_request_encodersH1U10-50-40-10.pkl'
OUTPUT_DIR = '../H1U10-50-40-10/visualizations'
```