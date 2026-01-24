import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import argparse

def train_ocsvm(input_file, model_output):
    if not os.path.exists(input_file):
        print(f"Błąd: Nie znaleziono pliku {input_file}")
        return

    df = pd.read_csv(input_file, on_bad_lines='skip', low_memory=False)
    
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    
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
    
    df_model = df[feature_cols].copy()
    
    num_cols = df_model.select_dtypes(include=[np.number]).columns
    df_model[num_cols] = df_model[num_cols].fillna(0)

    url_counts = df_model['endpointUrl'].value_counts(normalize=True)
    df_model['url_rarity'] = df_model['endpointUrl'].map(url_counts).fillna(0)
    
    df_model = df_model.drop(columns=['endpointUrl'])

    df_model = pd.get_dummies(df_model, columns=['apiMethod'], prefix='method')
    
    df_model = df_model.astype(float)

    train_mask = (df['is_bot'] == 0)
    
    X_train = df_model[train_mask]
    X_test = df_model
    y_test_true = df['is_bot']

    train_columns = X_train.columns.tolist()
    
    X_test = X_test.reindex(columns=train_columns, fill_value=0)

    print(f"Trening na {len(X_train)} próbkach ACTIVE_USER i CAUTIOUS_USER.")
    print(f"Liczba cech: {len(train_columns)}")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', OneClassSVM(
            kernel='rbf',
            gamma='scale', 
            nu=0.05
        ))
    ])
    
    pipeline.fit(X_train)

    y_pred_raw = pipeline.predict(X_test)
    
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

    print("\nWyniki")
    cm = confusion_matrix(y_test_true, y_pred)
    try:
        print(f"TP (Bot poprawnie wykryty):     {cm[1][1]}")
        print(f"FP (Człowiek uznany za bota):   {cm[0][1]}")
        print(f"TN (Człowiek wpuszczony):       {cm[0][0]}")
        print(f"FN (Bot niewykryty):            {cm[1][0]}")
    except IndexError:
        print(cm)
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test_true, y_pred, target_names=['Human', 'Bot']))

    save_data = {
        'pipeline': pipeline,
        'url_counts': url_counts,
        'train_columns': train_columns
    }
    
    joblib.dump(save_data, model_output)
    print(f"Model OC-SVM i metadane zapisane w: {model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trenowanie modelu OC-SVM')
    parser.add_argument('--dir', type=str, required=True, help='Nazwa katalogu (np. H1U200-45-45-10)')
    args = parser.parse_args()

    input_file_path = f'../{args.dir}/merged_data.csv'
    model_output_path = f'bot_ocsvm_model{args.dir}.pkl'
    
    print(f"Uruchamianie dla katalogu: {args.dir}")
    
    train_ocsvm(input_file_path, model_output_path)