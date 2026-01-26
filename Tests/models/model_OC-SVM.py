import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import argparse

def train_ocsvm(input_file, model_output):
    if not os.path.exists(input_file):
        print(f"Błąd: Nie znaleziono pliku {input_file}")
        return

    print("Wczytywanie danych...")
    df = pd.read_csv(input_file, on_bad_lines='skip', low_memory=False)
    
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    
    df['is_bot'] = df['userPersona'].astype(str).apply(lambda x: 1 if 'SCRAPER_BOT' in x else 0)
    
    feature_cols = [
        'apiTime', 'applicationTime', 'databaseTime', 
        'cpuUsage_market', 'cpuUsage_trade', 
        'memoryUsage_trade', 'memoryUsage_market', 
        'endpointUrl', 'apiMethod'
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    df_data = df[feature_cols + ['is_bot']].copy()
    
    num_cols = df_data.select_dtypes(include=[np.number]).columns
    df_data[num_cols] = df_data[num_cols].fillna(0)
    
    df_humans = df_data[df_data['is_bot'] == 0].copy()
    df_bots = df_data[df_data['is_bot'] == 1].copy()

    train_df, X_test_human = train_test_split(df_humans, test_size=0.2, random_state=42)

    test_df = pd.concat([X_test_human, df_bots], axis=0).sample(frac=1, random_state=42)
    
    y_test_true = test_df['is_bot']

    print(f"Trening na {len(train_df)} próbkach (tylko Human).")
    print(f"Test na {len(test_df)} próbkach (Human + Bot).")
    
    url_counts = train_df['endpointUrl'].value_counts(normalize=True)
    
    def map_url_rarity(data_series, counts):
        return data_series.map(counts).fillna(0)
    
    train_df['url_rarity'] = map_url_rarity(train_df['endpointUrl'], url_counts)
    test_df['url_rarity'] = map_url_rarity(test_df['endpointUrl'], url_counts)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(train_df[['apiMethod']])
    
    train_encoded = ohe.transform(train_df[['apiMethod']])
    test_encoded = ohe.transform(test_df[['apiMethod']])
    
    encoded_cols = ohe.get_feature_names_out(['apiMethod'])
    
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=train_df.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=test_df.index)
    
    drop_cols = ['apiMethod', 'endpointUrl', 'is_bot']
    
    X_train = pd.concat([train_df.drop(columns=drop_cols), train_encoded_df], axis=1)
    X_test = pd.concat([test_df.drop(columns=drop_cols), test_encoded_df], axis=1)
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    train_columns = X_train.columns.tolist()
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

    print("\nWyniki na zbiorze TESTOWYM:")
    cm = confusion_matrix(y_test_true, y_pred)
    try:
        tn, fp, fn, tp = cm.ravel()
        print(f"TP (Bot poprawnie wykryty):     {tp}")
        print(f"FP (Człowiek uznany za bota):   {fp}")
        print(f"TN (Człowiek wpuszczony):       {tn}")
        print(f"FN (Bot niewykryty):            {fn}")
    except Exception:
        print(cm)
    
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test_true, y_pred, target_names=['Human', 'Bot']))

    save_data = {
        'pipeline': pipeline,
        'url_counts': url_counts,
        'ohe_encoder': ohe,
        'train_columns': train_columns
    }
    
    joblib.dump(save_data, model_output)
    print(f"Model OC-SVM i metadane zapisane w: {model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trenowanie modelu OC-SVM')
    parser.add_argument('--dir', type=str, required=True, help='Nazwa katalogu (np. H1U200-45-45-10)')
    args = parser.parse_args()

    input_file_path = f'../{args.dir}/merged_data.csv'
    model_output_path = f'bot_ocsvm_model_{args.dir}.pkl'
    
    print(f"Uruchamianie dla katalogu: {args.dir}")
    
    train_ocsvm(input_file_path, model_output_path)