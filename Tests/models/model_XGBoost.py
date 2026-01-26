import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
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
        'apiTime', 'applicationTime', 'databaseTime', 
        'cpuUsage_market', 'cpuUsage_trade', 
        'memoryUsage_trade', 'memoryUsage_market', 
        'endpointUrl', 'apiMethod'
    ]
    
    df_model = df[feature_cols].copy()
    num_cols = df_model.select_dtypes(include=[np.number]).columns
    df_model[num_cols] = df_model[num_cols].fillna(0)
    
    X = df_model
    y = df['is_bot']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train = X_train.copy()
    X_test = X_test.copy()

    url_counts = X_train['endpointUrl'].value_counts()
    
    def map_frequency(data_series, counts):
        return data_series.map(counts).fillna(0)

    X_train['endpointUrl'] = map_frequency(X_train['endpointUrl'], url_counts)
    X_test['endpointUrl'] = map_frequency(X_test['endpointUrl'], url_counts)
    
    print("Zastosowano Frequency Encoding dla endpointUrl.")

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(X_train[['apiMethod']])
    
    train_encoded = ohe.transform(X_train[['apiMethod']])
    test_encoded = ohe.transform(X_test[['apiMethod']])
    encoded_cols = ohe.get_feature_names_out(['apiMethod'])
    
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_cols, index=X_train.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_cols, index=X_test.index)
    
    X_train_final = pd.concat([X_train.drop(columns=['apiMethod']), train_encoded_df], axis=1)
    X_test_final = pd.concat([X_test.drop(columns=['apiMethod']), test_encoded_df], axis=1)
    
    X_train_final = X_train_final.astype(float)
    X_test_final = X_test_final.astype(float)

    print(f"Liczba cech po transformacji: {X_train_final.shape[1]}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    )
    
    model.fit(X_train_final, y_train)

    y_proba = model.predict_proba(X_test_final)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"Najlepszy próg: {best_threshold:.4f}")
    print(f"Maksymalny F1-Score: {f1_scores[best_idx]:.4f}")

    save_data = {
        'url_counts': url_counts,
        'ohe_encoder': ohe,
        'best_threshold': best_threshold 
    }

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Wybór optymalnego progu decyzyjnego')
    plt.savefig('analysisResults/threshold_tuning.png')
    print("Zapisano wykres doboru progu: analysisResults/threshold_tuning.png")

    print("\nWyniki na zbiorze TESTOWYM:")
    y_pred = model.predict(X_test_final)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność (Accuracy): {acc:.4f}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Bot']))

    if not os.path.exists('analysisResults'):
        os.makedirs('analysisResults')
        
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(model, importance_type='weight', max_num_features=15)
    plt.title('Feature Importance (XGBoost)')
    plt.tight_layout()
    plt.savefig(plot_output)
    print(f"Zapisano wykres ważności cech: {plot_output}")

    save_data = {
        'url_counts': url_counts,
        'ohe_encoder': ohe
    }
    
    model.save_model(model_output)
    joblib.dump(save_data, encoder_output)
    
    print(f"\nModel zapisany jako: {model_output}")
    print(f"Enkodery (URL counts + OHE) zapisane jako: {encoder_output}")
    
    bot_avg = df[df['is_bot'] == 1]['apiTime'].mean()
    human_avg = df[df['is_bot'] == 0]['apiTime'].mean()
    print("\nStatystyki (cały zbiór):")
    print(f"Śr. czas API - Bot: {bot_avg:.2f} ms | Human: {human_avg:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trenowanie modelu XGBoost')
    parser.add_argument('--dir', type=str, required=True, help='Nazwa katalogu (np. H1U10-50-40-10)')
    args = parser.parse_args()

    input_file_path = f'../{args.dir}/merged_data.csv'
    
    model_output_path = f'bot_xgboost_model_{args.dir}.json'
    encoder_output_path = f'bot_xgboost_encoders_{args.dir}.pkl'
    plot_output_path = f'analysisResults/xgboost_importance_{args.dir}.png'
    
    print(f"Uruchamianie dla katalogu: {args.dir}")
    
    train_model(input_file_path, model_output_path, encoder_output_path, plot_output_path)