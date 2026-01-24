import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import argparse

def train_gaussian(input_file, model_output):    
    if not os.path.exists(input_file):
        print(f"Błąd: Nie znaleziono pliku {input_file}")
        return

    df = pd.read_csv(input_file, on_bad_lines='skip', low_memory=False)
    
    df['is_bot'] = df['userPersona'].astype(str).apply(lambda x: 1 if 'SCRAPER_BOT' in x else 0)
    
    feature_cols = [
        'apiTime', 
        'applicationTime', 
        'databaseTime', 
        'cpuUsage_market',    
        'cpuUsage_trade',     
        'memoryUsage_trade',  
        'memoryUsage_market', 
        'endpointUrl', 
        'apiMethod'
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            print(f"Ostrzeżenie: Brak kolumny '{col}' w danych. Wypełnianie zerami.")
            df[col] = 0
    
    df_model = df[feature_cols].copy().fillna(0)
    
    url_counts = df_model['endpointUrl'].value_counts()
    df_model['endpointUrl'] = df_model['endpointUrl'].map(url_counts).fillna(0)

    df_model = pd.get_dummies(df_model, columns=['apiMethod'], prefix='method')
    
    df_model = df_model.astype(float)

    X_train = df_model[df['is_bot'] == 0]
    
    X_test = df_model
    y_test_true = df['is_bot']

    train_columns = X_train.columns.tolist()

    print(f"Trening na {len(X_train)} próbkach (sami ludzie).")
    print(f"Testowanie na {len(X_test)} próbkach.")
    print(f"Liczba cech po transformacji: {len(train_columns)}")

    clf = EllipticEnvelope(
        contamination=0.01, 
        random_state=42,
        support_fraction=0.9
    )
    
    clf.fit(X_train)

    y_pred_raw = clf.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

    print("\nWyniki")
    cm = confusion_matrix(y_test_true, y_pred)
    
    try:
        tn, fp, fn, tp = cm.ravel()
        print(f"TP (Bot wykryty): {tp}")
        print(f"FP (User zablokowany): {fp}")
        print(f"TN (User wpuszczony): {tn}")
        print(f"FN (Bot wpuszczony): {fn}")
    except ValueError:
        print(cm)
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test_true, y_pred, target_names=['Human', 'Bot']))

    save_path = model_output
    
    if os.path.exists('weights'):
        save_path = os.path.join('weights', model_output)
    
    save_data = {
        'model': clf,
        'url_counts': url_counts,
        'train_columns': train_columns
    }

    joblib.dump(save_data, save_path)
    print(f"Model i metadane zapisane jako: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Nazwa katalogu (np. H1U10-50-40-10)')
    args = parser.parse_args()

    input_path = f'../{args.dir}/merged_data.csv'
    output_path = f'bot_gaussian_model{args.dir}.pkl'

    train_gaussian(input_path, output_path)