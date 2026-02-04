import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os
import sys

INPUT_FILE = '../M2U200-45-45-10/merged_data.csv'
MODEL_FILE = './weights/bot_occ_modelM2U200-45-45-10.pkl'
OUTPUT_DIR = '../M2U200-45-45-10/visualizations'
OUTPUT_IMAGE = 'OCC_Boundary_Full.png'
LOG_FILE = OUTPUT_IMAGE.replace('.png', '_ConsoleLog.txt')

SAMPLE_SIZE = 2000     
MESH_STEP = 0.05       

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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

    df_model = df[feature_cols + ['is_bot']].copy()

    if len(df_model) > SAMPLE_SIZE:
        print(f"   Losowanie {SAMPLE_SIZE} wierszy do wizualizacji")
        try:
            _, sample = train_test_split(df_model, test_size=SAMPLE_SIZE, stratify=df_model['is_bot'], random_state=42)
            df_model = sample
        except:
            indices = np.random.choice(df_model.index, SAMPLE_SIZE, replace=False)
            df_model = df_model.loc[indices]

    y_true = df_model['is_bot']
    X_raw = df_model.drop(columns=['is_bot'])

    return X_raw, y_true

def load_model_and_apply_encoders(X_df):
    if not os.path.exists(MODEL_FILE):
        print(f"BŁĄD: Brak modelu {MODEL_FILE}. Uruchom najpierw skrypt treningowy.")
        sys.exit(1)

    print(f"Wczytywanie modelu z: {MODEL_FILE}")
    data_pack = joblib.load(MODEL_FILE)
    
    model = data_pack['model']
    url_counts = data_pack['url_counts']
    ohe = data_pack['ohe_encoder']
    train_columns = data_pack['train_columns']
    
    X_df['endpointUrl'] = X_df['endpointUrl'].map(url_counts).fillna(0)
    
    encoded_matrix = ohe.transform(X_df[['apiMethod']])
    encoded_cols = ohe.get_feature_names_out(['apiMethod'])
    
    encoded_df = pd.DataFrame(encoded_matrix, columns=encoded_cols, index=X_df.index)
    
    X_processed = pd.concat([X_df.drop(columns=['apiMethod']), encoded_df], axis=1)
    
    X_processed = X_processed.astype(float).fillna(0)

    X_final = X_processed.reindex(columns=train_columns, fill_value=0)
            
    return X_final, model

def visualize(X, y_true, model):
    
    y_pred_raw = model.predict(X)
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    acc = accuracy_score(y_true, y_pred)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\nWYNIKI (Isolation Forest) na próbce wizualizacyjnej")
    print(f"Dokładność (Accuracy): {acc:.2%}")
    print(f"  TP (Bot poprawnie wykryty): {tp}")
    print(f"  FP (User błędnie zablokowany): {fp}")
    print(f"  TN (User poprawnie wpuszczony): {tn}")
    print(f"  FN (Bot niewykryty): {fn}")
    print("-" * 30)
    
    print(f"ANALIZA PCA:")
    print(f"  Wyjaśniona wariancja PC1 (Oś X): {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  Wyjaśniona wariancja PC2 (Oś Y): {pca.explained_variance_ratio_[1]:.2%}")
    
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
                c='blue', label='ACTIVE_USER i CURIOUS_USER', s=25, alpha=0.6, edgecolors='w')
    
    plt.scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1], 
                c='red', label='SCRAPER_BOT', marker='X', s=50, alpha=0.8, edgecolors='k')

    errors = X_pca[y_true != y_pred]
    if len(errors) > 0:
        plt.scatter(errors[:, 0], errors[:, 1], 
                    facecolors='none', edgecolors='yellow', s=100, 
                    linewidth=2, label='Błędna klasyfikacja')

    plt.title(f'Granice decyzyjne Isolation Forest (PCA)', fontsize=16)
    
    plt.xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    plt.ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
    
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    img_path = os.path.join(OUTPUT_DIR, OUTPUT_IMAGE)
    plt.savefig(img_path, dpi=300)
    print(f"Wykres zapisano w: {img_path}")
    print(f"Pełny log konsoli zapisano w: {os.path.join(OUTPUT_DIR, LOG_FILE)}")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    log_path = os.path.join(OUTPUT_DIR, LOG_FILE)
    sys.stdout = DualLogger(log_path)

    try:
        X_raw, y_true = load_data()
        X_encoded, occ_model = load_model_and_apply_encoders(X_raw)
        visualize(X_encoded, y_true, occ_model)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nWystąpił błąd: {e}")