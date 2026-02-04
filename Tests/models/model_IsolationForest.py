import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder  # <-- Dodano import
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import argparse

def train_occ(input_file, model_output):
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
            df[col] = 0
            
    df_model = df[feature_cols].copy().fillna(0)

    url_counts = df_model['endpointUrl'].value_counts()
    df_model['endpointUrl'] = df_model['endpointUrl'].map(url_counts).fillna(0)
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    encoded_matrix = ohe.fit_transform(df_model[['apiMethod']])
    encoded_cols = ohe.get_feature_names_out(['apiMethod'])
    
    encoded_df = pd.DataFrame(encoded_matrix, columns=encoded_cols, index=df_model.index)
    
    df_model = pd.concat([df_model.drop(columns=['apiMethod']), encoded_df], axis=1)
    
    df_model = df_model.astype(float)

    X_train = df_model[df['is_bot'] == 0]
    
    X_test = df_model
    y_test_true = df['is_bot']

    train_columns = X_train.columns.tolist()

    print(f"Trening na {len(X_train)} próbkach ACTIVE_USER i CAUTIOUS_USER.")
    print(f"Liczba cech po kodowaniu: {len(train_columns)}")

    clf = IsolationForest(
        n_estimators=100, 
        max_samples='auto', 
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train)

    y_pred_raw = clf.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

    print("\nWyniki")
    cm = confusion_matrix(y_test_true, y_pred)
    try:
        print(f"TP (Bot wykryty): {cm[1][1]}")
        print(f"FP (User zablokowany): {cm[0][1]}")
        print(f"TN (User wpuszczony): {cm[0][0]}")
        print(f"FN (Bot wpuszczony): {cm[1][0]}")
    except IndexError:
        print(cm)

    print("\nRaport klasyfikacji:")
    print(classification_report(y_test_true, y_pred, target_names=['Human', 'Bot']))

    save_data = {
        'model': clf,
        'url_counts': url_counts,
        'ohe_encoder': ohe,
        'train_columns': train_columns
    }
    
    joblib.dump(save_data, model_output)
    print(f"Model Isolation Forest oraz metadane zapisane w: {model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Nazwa podkatalogu z danymi')
    args = parser.parse_args()

    input_file_path = f'../{args.dir}/merged_data.csv'
    
    model_output_path = f'bot_occ_model{args.dir}.pkl'
    
    print(f"Uruchamianie dla katalogu: {args.dir}")
    print(f"Plik wejściowy: {input_file_path}")
    
    train_occ(input_file_path, model_output_path)