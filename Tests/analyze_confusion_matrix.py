import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def ensure_output_dir(base_dir, subfolder='confusion_analysis'):
    out_dir = os.path.join(base_dir, subfolder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def load_data(directory):
    files = {
        'users': 'users.csv',
        'blocked': 'blocked_user_log.csv',
        'allowed': 'market_log.csv'
    }
    dfs = {}
    print(f"--- Wczytywanie danych z {directory} ---")
    
    for key, filename in files.items():
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, low_memory=False)
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
                
                if 'userId' in df.columns:
                    df['userId'] = df['userId'].astype(str).str.replace(r'\.0$', '', regex=True)
                
                dfs[key] = df
                print(f"✅ Wczytano {filename}: {len(df)} wierszy")
            except Exception as e:
                print(f"Błąd wczytywania {filename}: {e}")
        else:
            pass
    
    return dfs

def get_user_labels(dfs):
    user_labels_map = {}
    
    users_df = dfs.get('users')
    if users_df is not None and 'id' in users_df.columns and 'userPersona' in users_df.columns:
        users_df['id'] = users_df['id'].astype(str).str.replace(r'\.0$', '', regex=True)
        for _, row in users_df.iterrows():
            user_labels_map[row['id']] = str(row['userPersona'])

    market_df = dfs.get('allowed')
    if market_df is not None and 'userId' in market_df.columns and 'userPersona' in market_df.columns:
        valid_market = market_df[
            (market_df['userId'] != '\\N') & 
            (market_df['userId'] != 'nan') &
            (market_df['userPersona'].notna()) &
            (market_df['userPersona'] != 'UNKNOWN')
        ]
        
        unique_users = valid_market[['userId', 'userPersona']].drop_duplicates()
        count_new = 0
        for _, row in unique_users.iterrows():
            uid = row['userId']
            persona = str(row['userPersona'])
            
            if uid not in user_labels_map:
                user_labels_map[uid] = persona
                count_new += 1
        
        if count_new > 0:
            print(f"Znaleziono {count_new} nowych etykiet użytkowników w market_log.csv")

    def is_bot(persona):
        s = str(persona).upper()
        return 1 if 'BOT' in s or 'SCRAPER' in s else 0
    
    final_map = {k: is_bot(v) for k, v in user_labels_map.items()}
    print(f"Łącznie zidentyfikowano {len(final_map)} użytkowników.")
    return final_map

def analyze_confusion_over_time(dfs, user_labels, time_freq='1s'):
    print("\nAnaliza macierzy pomyłek")
    
    if not user_labels:
        return None

    def map_is_bot(series):
        return series.map(user_labels)

    blocked_df = dfs.get('blocked')
    blocked_resampled = pd.DataFrame()
    
    if blocked_df is not None and not blocked_df.empty:
        df = blocked_df.copy()
        df['is_bot_truth'] = map_is_bot(df['userId'])
        
        unknown_count = df['is_bot_truth'].isna().sum()
        if unknown_count > 0:
            print(f"Zignorowano {unknown_count} blokad dla nieznanych użytkowników.")
            df = df.dropna(subset=['is_bot_truth'])

        if not df.empty:
            df['TP'] = (df['is_bot_truth'] == 1).astype(int)
            df['FP'] = (df['is_bot_truth'] == 0).astype(int)
            if 'timestamp' in df.columns:
                blocked_resampled = df.set_index('timestamp').resample(time_freq)[['TP', 'FP']].sum()

    allowed_df = dfs.get('allowed')
    allowed_resampled = pd.DataFrame()
    
    if allowed_df is not None and not allowed_df.empty:
        df = allowed_df.copy()
        
        df = df[df['userId'] != '\\N']
        df['is_bot_truth'] = map_is_bot(df['userId'])
        
        df = df.dropna(subset=['is_bot_truth'])

        if not df.empty:
            df['FN'] = (df['is_bot_truth'] == 1).astype(int)
            df['TN'] = (df['is_bot_truth'] == 0).astype(int)
            if 'timestamp' in df.columns:
                allowed_resampled = df.set_index('timestamp').resample(time_freq)[['FN', 'TN']].sum()

    if blocked_resampled.empty and allowed_resampled.empty:
        print("Brak danych do analizy confusion matrix (obie grupy puste).")
        return None

    result = pd.concat([blocked_resampled, allowed_resampled], axis=1).fillna(0)
    
    for col in ['TP', 'FP', 'FN', 'TN']:
        if col not in result.columns:
            result[col] = 0.0

    result['Precision'] = result['TP'] / (result['TP'] + result['FP'] + 1e-9)
    result['Recall'] = result['TP'] / (result['TP'] + result['FN'] + 1e-9)
    result['F1'] = 2 * (result['Precision'] * result['Recall']) / (result['Precision'] + result['Recall'] + 1e-9)
    
    return result.fillna(0)

def plot_confusion_timeline(result, output_dir):
    plt.figure(figsize=(15, 8))
    plt.plot(result.index, result['TP'], label='TP (Bot Zablokowany)', color='green')
    plt.plot(result.index, result['FP'], label='FP (Człowiek Zablokowany)', color='red', linestyle='--')
    plt.plot(result.index, result['FN'], label='FN (Bot Przepuszczony)', color='orange', linestyle=':')
    plt.plot(result.index, result['TN'], label='TN (Człowiek Przepuszczony)', color='blue', alpha=0.3)
    plt.legend()
    plt.title("Confusion Matrix w Czasie")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_timeline.png'))
    plt.close()

    plt.figure(figsize=(15, 6))
    plt.plot(result.index, result['Precision'], label='Precision')
    plt.plot(result.index, result['Recall'], label='Recall')
    plt.plot(result.index, result['F1'], label='F1')
    plt.legend()
    plt.title("Metryki (0-1)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_timeline.png'))
    plt.close()

def generate_report(result, output_dir):
    total = result[['TP', 'FP', 'FN', 'TN']].sum()
    tp, fp, fn, tn = total['TP'], total['FP'], total['FN'], total['TN']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    report = f"""
TP: {int(tp)} | FP: {int(fp)} | FN: {int(fn)} | TN: {int(tn)}
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1 Score:  {f1:.4f}
"""
    with open(os.path.join(output_dir, 'confusion_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='csv_output')
    parser.add_argument('--out', type=str, default='confusion_results')
    parser.add_argument('--step', type=str, default='1s')
    args = parser.parse_args()

    dfs = load_data(args.dir)
    user_labels = get_user_labels(dfs)
    
    result = analyze_confusion_over_time(dfs, user_labels, args.step)
    
    if result is not None and not result.empty:
        out_dir = ensure_output_dir(args.dir, args.out)
        result.to_csv(os.path.join(out_dir, 'confusion_matrix_timeseries.csv'))
        plot_confusion_timeline(result, out_dir)
        generate_report(result, out_dir)
    else:
        print("Nie wygenerowano wyników (pusty DataFrame).")

if __name__ == "__main__":
    main()