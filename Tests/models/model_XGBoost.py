import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import argparse

def train_model(input_file, model_output, encoder_output, plot_output):
    if not os.path.exists(input_file):
        print(f"Błąd: Nie znaleziono pliku {input_file}")
        return

    df = pd.read_csv(input_file, on_bad_lines='skip', low_memory=False)
    
    df['is_bot'] = df['userPersona'].astype(str).apply(lambda x: 1 if 'SCRAPER_BOT' in x else 0)
    
    print("Rozkład klas (0=Człowiek, 1=Bot):")
    print(df['is_bot'].value_counts())

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
    
    df_model = df[feature_cols].copy()
    df_model = df_model.fillna(0)
    
    label_encoders = {}
    for col in ['endpointUrl', 'apiMethod']:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le
        print(f"Zakodowano kolumnę {col}: {len(le.classes_)} unikalnych wartości")

    X = df_model
    y = df['is_bot']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)

    print("\nWyniki")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność (Accuracy): {acc:.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Bot']))

    if not os.path.exists('analysisResults'):
        os.makedirs('analysisResults')
        
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='weight', max_num_features=10)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    plt.savefig(plot_output)
    print(f"Zapisano wykres: {plot_output}")

    model.get_booster().save_model(model_output)
    joblib.dump(label_encoders, encoder_output)
    print(f"\nModel zapisany jako: {model_output}")
    print(f"Enkodery zapisane jako: {encoder_output}")
    
    print("\nAnaliza Heurystyczna")
    bot_avg_time = df[df['is_bot'] == 1]['apiTime'].mean()
    human_avg_time = df[df['is_bot'] == 0]['apiTime'].mean()
    print(f"Średni czas API bota: {bot_avg_time:.2f} ms")
    print(f"Średni czas API człowieka: {human_avg_time:.2f} ms")
    print("Jeśli w Node.js będzie seria zapytań GET z czasem >", 
          f"{(bot_avg_time + human_avg_time)/2:.0f} ms, to prawdopodobnie bot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trenowanie modelu XGBoost')
    parser.add_argument('--dir', type=str, required=True, help='Nazwa katalogu (np. H1U10-50-40-10)')
    args = parser.parse_args()

    input_file_path = f'../{args.dir}/merged_data.csv'
    
    model_output_path = f'bot_request_model{args.dir}.json'
    encoder_output_path = f'bot_request_encoders{args.dir}.pkl'
    plot_output_path = f'analysisResults/bot_request_importance{args.dir}.png'
    
    print(f"Uruchamianie dla katalogu: {args.dir}")
    
    train_model(input_file_path, model_output_path, encoder_output_path, plot_output_path)