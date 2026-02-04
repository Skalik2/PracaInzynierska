import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import joblib
import os
import sys

INPUT_FILE = '../M2U200-45-45-10/merged_data.csv'
MODEL_FILE = './weights/bot_xgboost_model_M2U200-45-45-10.json'
ENCODERS_FILE = './weights/bot_xgboost_encoders_M2U200-45-45-10.pkl'
OUTPUT_DIR = '../M2U200-45-45-10/visualizations'
OUTPUT_IMAGE = 'XGBoost_Boundary_Full.png'
OUTPUT_TXT = OUTPUT_IMAGE.replace('.png', '_Stats.txt')

SAMPLE_SIZE = 2000     
MESH_STEP = 0.05       

def load_data():
    if not os.path.exists(INPUT_FILE):
        print(f"BŁĄD: Nie znaleziono pliku {INPUT_FILE}")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, on_bad_lines='skip', low_memory=False)
    
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

    df_model = df[feature_cols].copy() 

    if len(df_model) > SAMPLE_SIZE:
        print(f"   Losowanie {SAMPLE_SIZE} wierszy do wizualizacji")
        indices = np.random.choice(df_model.index, SAMPLE_SIZE, replace=False)
        df_model = df_model.loc[indices]
        y_true = df.loc[indices, 'is_bot']
    else:
        y_true = df['is_bot']

    num_cols = df_model.select_dtypes(include=[np.number]).columns
    df_model[num_cols] = df_model[num_cols].fillna(0)

    return df_model, y_true

def prepare_features(df_input):
    if not os.path.exists(ENCODERS_FILE):
        print(f"BŁĄD: Brak pliku encoderów {ENCODERS_FILE}")
        sys.exit(1)

    print(f"Wczytywanie encoderów z: {ENCODERS_FILE}")
    encoders_data = joblib.load(ENCODERS_FILE)
    
    url_counts = encoders_data['url_counts']
    ohe_encoder = encoders_data['ohe_encoder']
    
    df_processed = df_input.copy()

    df_processed['endpointUrl'] = df_processed['endpointUrl'].map(url_counts).fillna(0)

    encoded_matrix = ohe_encoder.transform(df_processed[['apiMethod']])
    encoded_cols = ohe_encoder.get_feature_names_out(['apiMethod'])
    
    encoded_df = pd.DataFrame(encoded_matrix, columns=encoded_cols, index=df_processed.index)
    
    X_final = pd.concat([df_processed.drop(columns=['apiMethod']), encoded_df], axis=1)
    
    X_final = X_final.astype(float)
    
    print(f"Liczba cech po transformacji: {X_final.shape[1]}")
    return X_final

def load_xgboost_model():
    if not os.path.exists(MODEL_FILE):
        print(f"BŁĄD: Brak modelu {MODEL_FILE}")
        sys.exit(1)
        
    print(f"Wczytywanie modelu z: {MODEL_FILE}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model

def visualize(X, y_true, model):
    
    y_pred = model.predict(X)
    
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'],
        index=X.columns
    )
    sorted_loadings = loadings.sort_values(by='PC1', ascending=False)
    
    report_lines = []
    report_lines.append(f"RAPORT ANALIZY PCA I MODELU (XGBoost)")
    report_lines.append(f"Plik danych: {INPUT_FILE}")
    report_lines.append(f"Liczba próbek: {len(y_true)}")
    report_lines.append("-" * 40)
    report_lines.append(f"WYNIKI KLASYFIKACJI:")
    report_lines.append(f"Dokładność (Accuracy): {acc:.2%}")
    report_lines.append(f"Macierz pomyłek:")
    report_lines.append(f"  TP (Bot wykryty): {tp}")
    report_lines.append(f"  FP (Fałszywy alarm): {fp}")
    report_lines.append(f"  TN (Poprawny user): {tn}")
    report_lines.append(f"  FN (Bot niewykryty): {fn}")
    report_lines.append("-" * 40)
    report_lines.append(f"ANALIZA PCA:")
    report_lines.append(f"Wyjaśniona wariancja PC1: {pca.explained_variance_ratio_[0]:.2%}")
    report_lines.append(f"Wyjaśniona wariancja PC2: {pca.explained_variance_ratio_[1]:.2%}")
    report_lines.append("\nTOP 10 Cech wpływających na PC1 (oś X):")
    report_lines.append(sorted_loadings.head(5).to_string())
    report_lines.append("...")
    report_lines.append(sorted_loadings.tail(5).to_string())

    report_content = "\n".join(report_lines)
    print(report_content)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    txt_path = os.path.join(OUTPUT_DIR, OUTPUT_TXT)
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\nRaport tekstowy zapisano w: {txt_path}")
    except Exception as e:
        print(f"\nNie udało się zapisać pliku tekstowego: {e}")

    boundary_model = KNeighborsClassifier(n_neighbors=15, weights='distance')
    boundary_model.fit(X_pca, y_pred) 

    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP),
                         np.arange(y_min, y_max, MESH_STEP))

    Z = boundary_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.7, cmap=plt.cm.coolwarm)
    
    plt.scatter(X_pca[y_true == 0, 0], X_pca[y_true == 0, 1], 
                c='blue', label='ACTIVE_USER', s=25, alpha=0.6, edgecolors='w')
    
    plt.scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1], 
                c='red', label='SCRAPER_BOT', marker='X', s=50, alpha=0.8, edgecolors='k')

    errors = X_pca[y_true != y_pred]
    if len(errors) > 0:
        plt.scatter(errors[:, 0], errors[:, 1], 
                    facecolors='none', edgecolors='yellow', s=100, 
                    linewidth=2, label='Błędna klasyfikacja')

    plt.title(f'Granice decyzyjne XGBoost (PCA)', fontsize=16)
    plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)

    img_path = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE)
    plt.savefig(img_path, dpi=300)
    print(f"Wykres zapisano w: {img_path}")

if __name__ == "__main__":
    X_raw, y_true = load_data()
    X_prepared = prepare_features(X_raw)
    xgb_model = load_xgboost_model()
    visualize(X_prepared, y_true, xgb_model)