import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import joblib
import os
import sys

INPUT_FILE = '../csv_output/merged_data.csv'
MODEL_XGB = './weights/bot_request_model.json'
MODEL_OCC = './weights/bot_occ_model.pkl'
ENCODERS_XGB = './weights/bot_request_encoders.pkl' 
OUTPUT_DIR = 'analysisResults'

MESH_STEP = 0.05
SAMPLE_SIZE = 1000

def load_data_and_encoders():
    if not os.path.exists(INPUT_FILE):
        print(f"Błąd: Brak pliku {INPUT_FILE}")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, on_bad_lines='skip', low_memory=False)
    
    df['is_bot'] = df['userPersona'].astype(str).apply(lambda x: 1 if 'SCRAPER_BOT' in x else 0)

    if len(df) > SAMPLE_SIZE:
        print(f"   Losowanie {SAMPLE_SIZE} requestów dla przejrzystości wykresu i testów...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        print(f"   Używanie wszystkich {len(df)} requestów.")

    feature_cols = ['apiTime', 'applicationTime', 'databaseTime', 'cpuUsage', 'endpointUrl', 'apiMethod']
    X_raw = df[feature_cols].copy().fillna(0)

    encoders = {}
    if os.path.exists(ENCODERS_XGB):
        encoders = joblib.load(ENCODERS_XGB)
        for col in ['endpointUrl', 'apiMethod']:
            X_raw[col] = X_raw[col].astype(str)
            le = encoders[col]
            known_classes = set(le.classes_)
            X_raw[col] = X_raw[col].apply(lambda x: le.transform([x])[0] if x in known_classes else 0)
    else:
        for col in ['endpointUrl', 'apiMethod']:
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
            encoders[col] = le

    return X_raw, df['is_bot']

def load_models():
    if not os.path.exists(MODEL_XGB) or not os.path.exists(MODEL_OCC):
        print("Brak plików modeli (.json lub .pkl).")
        sys.exit(1)
        
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(MODEL_XGB)
    
    occ_pack = joblib.load(MODEL_OCC)
    occ_model = occ_pack['model']
    
    return xgb_model, occ_model

def print_detailed_stats(model_name, y_true, y_pred):
    total = len(y_true)
    errors = np.sum(y_true != y_pred)
    accuracy = (total - errors) / total
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    print(f"\nWYNIKI {model_name}")
    print(f"  Liczba próbek: {total}")
    print(f"  Poprawnie sklasyfikowane: {total - errors}")
    print(f"  Błędnie sklasyfikowane:   {errors}")
    print(f"  Dokładność (Accuracy):    {accuracy:.2%}")
    print("-" * 30)
    print(f"  [FP] Fałszywe Alarmy (Zablokowani ludzie): {fp}")
    print(f"  [FN] Niewykryte Boty (Boty w systemie):    {fn}")
    print(f"  [TP] Prawidłowo wykryte Boty:               {tp}")
    print(f"  [TN] Prawidłowo zidentyfikowani Ludzie:     {tn}")
    print("==========================================")

def generate_decision_boundary_knn(X_pca, predictions):
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X_pca, predictions)
    
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, MESH_STEP),
                         np.arange(y_min, y_max, MESH_STEP))
    
    Z = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    return xx, yy, Z.reshape(xx.shape)

def plot_boundary(ax, title, Z, xx, yy, X_pca, y_true, model_preds, plot_type='full'):
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu_r, alpha=0.6, levels=np.linspace(0, 1, 11))
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    if plot_type == 'background': return

    if plot_type == 'full':
        ax.scatter(X_pca[y_true == 0, 0], X_pca[y_true == 0, 1], c='navy', s=20, label='Człowiek', edgecolors='white', alpha=0.7)
        ax.scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1], c='darkred', marker='X', s=40, label='Bot', edgecolors='black', alpha=0.8)
        ax.legend(loc='lower right')
        
    elif plot_type == 'errors':
        errors_mask = (model_preds != y_true)
        ax.scatter(X_pca[~errors_mask, 0], X_pca[~errors_mask, 1], c='gray', s=5, alpha=0.1) # Dobre
        if np.sum(errors_mask) > 0:
            ax.scatter(X_pca[errors_mask, 0], X_pca[errors_mask, 1], c='yellow', s=60, marker='o', edgecolors='black', label='Błąd modelu')
            ax.legend()

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    X_6d, y_true = load_data_and_encoders()
    xgb_model, occ_model = load_models()

    print("3. Generowanie predykcji i statystyk...")
    
    xgb_preds = xgb_model.predict(X_6d)
    print_detailed_stats("XGBoost (Binary)", y_true, xgb_preds)
    
    occ_raw = occ_model.predict(X_6d)
    occ_preds = np.where(occ_raw == -1, 1, 0)
    print_detailed_stats("Isolation Forest (OCC)", y_true, occ_preds)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_6d)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    components = pd.DataFrame(pca.components_, columns=X_6d.columns, index=['PC1 (Oś X)', 'PC2 (Oś Y)'])
    print("\nINTERPRETACJA OSI (PCA)")
    print("Jakie cechy najbardziej wpływają na układ kropek?")
    print(components.T.sort_values(by='PC1 (Oś X)', key=abs, ascending=False))
    print("----------------------------------------\n")
    
    xx_occ, yy_occ, Z_occ = generate_decision_boundary_knn(X_pca, occ_preds)
    xx_xgb, yy_xgb, Z_xgb = generate_decision_boundary_knn(X_pca, xgb_preds)

    configs = [("full", "1_Pelna_Wizualizacja.png"), ("background", "2_Tylko_Granice.png"), ("errors", "3_Bledne_Trafienia.png")]

    for p_type, fname in configs:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plot_boundary(axes[0], "A) Isolation Forest", Z_occ, xx_occ, yy_occ, X_pca, y_true, occ_preds, p_type)
        plot_boundary(axes[1], "B) XGBoost", Z_xgb, xx_xgb, yy_xgb, X_pca, y_true, xgb_preds, p_type)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/{fname}")
        plt.close()
    
    print(f"\nZapisano wykresy w folderze {OUTPUT_DIR}")

if __name__ == "__main__":
    main()