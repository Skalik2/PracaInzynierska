import os
import re
import argparse
import pandas as pd

def sort_csv_files(target_dir):
    print(f"Sortowanie plików CSV w {target_dir}")
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
                        
                        df = df.sort_values(by='timestamp')
                        
                        df.to_csv(file_path, index=False)
                    else:
                        pass
                except Exception as e:
                    print(e)

def merge_logs(target_dir):
    print(f"\nŁączenie logów (market + traffic)")
    for root, dirs, files in os.walk(target_dir):
        if 'traffic_log.csv' in files and 'market_log.csv' in files:
            traffic_path = os.path.join(root, 'traffic_log.csv')
            market_path = os.path.join(root, 'market_log.csv')
            output_path = os.path.join(root, 'complete_market_log_csv.csv')
            
            try:
                traffic_df = pd.read_csv(traffic_path)
                market_df = pd.read_csv(market_path)
                
                complete_log_df = pd.merge(
                    market_df, 
                    traffic_df[['requestId', 'apiTime']], 
                    on='requestId', 
                    how='inner'
                )
                
                cols_to_keep = [
                    'id', 
                    'timestamp', 
                    'apiMethod', 
                    'applicationTime', 
                    'databaseTime', 
                    'endpointUrl', 
                    'apiTime', 
                    'userId',
                    'userPersona'
                ]
                
                cols_to_keep = [c for c in cols_to_keep if c in complete_log_df.columns]
                
                complete_log_df = complete_log_df[cols_to_keep]
                
                complete_log_df.to_csv(output_path, index=False)
                print(f"Utworzono połączony log: {output_path}")
                
            except Exception as e:
                print(f"Błąd przy łączeniu logów w {root}: {e}")

def filter_complete_logs(target_dir):
    print(f"\nFiltrowanie logów")
    
    allowed_patterns = [
        r"^/buyoffer/create$",
        r"^/selloffer/create$",
        r"^/stockrate/company/\d+$"
    ]
    combined_pattern = re.compile('|'.join(allowed_patterns))

    for root, dirs, files in os.walk(target_dir):
        if 'complete_market_log_csv.csv' in files:
            file_path = os.path.join(root, 'complete_market_log_csv.csv')
            try:
                df = pd.read_csv(file_path)
                
                if 'endpointUrl' in df.columns:
                    filtered_df = df[df['endpointUrl'].apply(lambda x: bool(combined_pattern.match(str(x))))]
                    
                    filtered_df.to_csv(file_path, index=False)
                    print(f"Przefiltrowano plik: {file_path}")
            except Exception as e:
                print(f"Błąd przy filtrowaniu {file_path}: {e}")

def process_trade_logs(target_dir):
    print(f"\nPrzetwarzanie trade logs (CumSum)")
    for root, dirs, files in os.walk(target_dir):
        if 'trade_log.csv' in files:
            file_path = os.path.join(root, 'trade_log.csv')
            output_path = os.path.join(root, 'sum_trade_log.csv')
            
            try:
                df = pd.read_csv(file_path)
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
                    df = df.sort_values(by='timestamp')
                    
                    if 'numberOfSellOffers' in df.columns:
                        df['numberOfSellOffers'] = df['numberOfSellOffers'].cumsum()
                    if 'numberOfBuyOffers' in df.columns:
                        df['numberOfBuyOffers'] = df['numberOfBuyOffers'].cumsum()
                    
                    df.to_csv(output_path, index=False)
                    print(f"Utworzono sumaryczny log handlowy: {output_path}")
                else:
                    print(f"Brak kolumny 'timestamp' w {file_path}")
                    
            except Exception as e:
                print(e)

def main():
    parser = argparse.ArgumentParser(description="Skrypt przygotowania danych z logów giełdy.")
    parser.add_argument('--dir', type=str, default='.', help='Ścieżka do katalogu z danymi (domyślnie obecny katalog)')
    args = parser.parse_args()

    target_directory = args.dir
    
    if not os.path.exists(target_directory):
        return

    sort_csv_files(target_directory)
    merge_logs(target_directory)
    filter_complete_logs(target_directory)
    process_trade_logs(target_directory)
    
    print("\nZakończono przetwarzanie danych.")

if __name__ == "__main__":
    main()