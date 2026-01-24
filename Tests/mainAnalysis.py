import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import xgboost as xgb
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

OUTPUT_DIR = 'EDA'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_text_report(filename, content):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Zapisano raport: {filepath}")

def sepReplicas(source_path):
    print("Agregacja zasobów Trade (CPU + RAM)")
    input_file = os.path.join(source_path, 'trade_cpu.csv')
    output_file = os.path.join(source_path, 'trade_cpu_pivot.csv')
    
    try:
        if not os.path.exists(input_file):
            print(f"Pominięto: brak pliku {input_file}")
            return

        trade_cpu = pd.read_csv(input_file)
        
        if 'timestamp' in trade_cpu.columns:
            trade_cpu['timestamp'] = pd.to_datetime(trade_cpu['timestamp'])
            trade_cpu['timestamp_sec'] = trade_cpu['timestamp'].dt.floor('1s')
            
            agg_trade = trade_cpu.groupby('timestamp_sec')[['cpuUsage', 'memoryUsage']].mean().reset_index()
            
            agg_trade = agg_trade.rename(columns={
                'timestamp_sec': 'timestamp', 
                'cpuUsage': 'cpuUsage_trade',
                'memoryUsage': 'memoryUsage_trade'
            })
            
            agg_trade = agg_trade.sort_values('timestamp')
            
            agg_trade.to_csv(output_file, index=False)
            print(f"Stworzono zagregowany plik Trade: {output_file}")
        else:
            pass
    except Exception as e:
        print(e)

def mergeData(source_path):
    print("\nŁączenie danych")
    try:
        files_map = {
            'market_log': 'complete_market_log_csv.csv',
            'trade_log': 'sum_trade_log.csv',
            'market_cpu': 'market_cpu.csv',
            'trade_cpu_pivot': 'trade_cpu_pivot.csv',
            'traffic_cpu': 'traffic_cpu.csv'
        }
        
        dfs = {}
        for key, filename in files_map.items():
            filepath = os.path.join(source_path, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, na_values=['\\N'])
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
                    df = df.sort_values('timestamp')
                    dfs[key] = df

        if 'market_log' not in dfs:
            return False

        merged_df = dfs['market_log']

        if 'traffic_cpu' in dfs:
            t_df = dfs['traffic_cpu'].rename(columns={
                'cpuUsage': 'cpuUsage_traffic', 
                'memoryUsage': 'memoryUsage_traffic'
            })
            if 'id' in t_df.columns: t_df = t_df.drop(columns=['id'])
            merged_df = pd.merge_asof(merged_df, t_df, on='timestamp', direction='nearest', tolerance=pd.Timedelta('30s'))

        if 'trade_log' in dfs:
            merged_df = pd.merge_asof(merged_df, dfs['trade_log'], on='timestamp', direction='backward', tolerance=pd.Timedelta('1min'), suffixes=('', '_trade_log'))

        if 'market_cpu' in dfs:
            m_cpu = dfs['market_cpu'].rename(columns={
                'cpuUsage': 'cpuUsage_market',
                'memoryUsage': 'memoryUsage_market'
            })
            cols_to_drop = [c for c in m_cpu.columns if 'id' in c]
            m_cpu = m_cpu.drop(columns=cols_to_drop)
            merged_df = pd.merge_asof(merged_df, m_cpu, on='timestamp', direction='nearest', tolerance=pd.Timedelta('30s'))

        if 'trade_cpu_pivot' in dfs:
            merged_df = pd.merge_asof(merged_df, dfs['trade_cpu_pivot'], on='timestamp', direction='nearest', tolerance=pd.Timedelta('30s'))

        merged_df = merged_df.dropna(subset=['applicationTime']) 
        if 'userPersona' in merged_df.columns:
            merged_df['userPersona'] = merged_df['userPersona'].fillna('UNKNOWN')

        output_file = os.path.join(source_path, 'merged_data.csv')
        merged_df.to_csv(output_file, index=False)
        print(f"Zapisano merged_data.csv. Kolumny: {list(merged_df.columns)}")
        return not merged_df.empty
        
    except Exception as e:
        print(e)
        return False

def focusedPersonaCorrelation(source_path):
    print("\nDedykowana analiza korelacji")
    input_file = os.path.join(source_path, 'merged_data.csv')
    try:
        df = pd.read_csv(input_file)
        
        if 'userPersona' not in df.columns:
            return

        persona_dummies = pd.get_dummies(df['userPersona'], prefix='Persona')
        
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                        if col not in ['id', 'userId', 'id_trade_log', 'timestamp']]
        
        if 'apiMethod' in df.columns:
             method_dummies = pd.get_dummies(df['apiMethod'], prefix='Method')
             analysis_df = pd.concat([persona_dummies, df[numeric_cols], method_dummies], axis=1)
        else:
             analysis_df = pd.concat([persona_dummies, df[numeric_cols]], axis=1)

        corr_matrix = analysis_df.corr(method='pearson')
        
        persona_rows = [c for c in corr_matrix.index if c.startswith('Persona_')]
        other_cols = [c for c in corr_matrix.columns if not c.startswith('Persona_')]
        
        focused_corr = corr_matrix.loc[persona_rows, other_cols]
        
        focused_corr = focused_corr.loc[:, (focused_corr != 0).any(axis=0)]

        if focused_corr.empty:
            print("Brak istotnych korelacji.")
            return

        plt.figure(figsize=(16, 8))
        sns.heatmap(focused_corr, annot=True, cmap='RdBu_r', center=0, fmt=".2f", linewidths=.5)
        plt.title('Klasa użytkownika względem parametrów systemowych i typów API')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_img = os.path.join(OUTPUT_DIR, 'persona_correlation_matrix.png')
        plt.savefig(output_img)
        plt.close()
        
        report = "Najsilniejsze korelacje dla Person (abs > 0.1):\n"
        for persona in persona_rows:
            series = focused_corr.loc[persona]
            strong_corrs = series[series.abs() > 0.1].sort_values(key=abs, ascending=False)
            if not strong_corrs.empty:
                report += f"\n[{persona}]:\n"
                report += strong_corrs.to_string() + "\n"
        
        save_text_report('persona_correlation_report.txt', report)

    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()

def personaAnalysis(source_path):
    print("\nAnaliza wydajności per klasa użytkownika")
    input_file = os.path.join(source_path, 'merged_data.csv')
    try:
        df = pd.read_csv(input_file)
        if 'userPersona' not in df.columns or 'apiTime' not in df.columns:
            return

        plt.figure(figsize=(12, 8))
        sns.boxplot(x='userPersona', y='apiTime', data=df, palette="Set3")
        plt.title('Rozkład czasu odpowiedzi API (ms) wg typu użytkownika')
        plt.yscale('log')
        plt.ylabel('Czas API (ms) - skala log')
        plt.savefig(os.path.join(OUTPUT_DIR, 'persona_performance.png'), bbox_inches='tight')
        plt.close()
        print("Zapisano: persona_performance.png")

        report = df.groupby('userPersona')[['apiTime', 'databaseTime', 'applicationTime']].describe().to_string()
        save_text_report('persona_stats.txt', report)

    except Exception as e:
        print(e)

def association(source_path):
    print("\nReguły asocjacji")
    input_file = os.path.join(source_path, 'merged_data.csv')
    
    try:
        df = pd.read_csv(input_file)
        if 'userPersona' not in df.columns:
            return
        
        dataset = pd.DataFrame()
        dataset = pd.concat([dataset, pd.get_dummies(df['userPersona'], prefix='Persona')], axis=1)

        if 'apiMethod' in df.columns:
             dataset = pd.concat([dataset, pd.get_dummies(df['apiMethod'], prefix='Method')], axis=1)

        numeric_cols = [
            'applicationTime', 'databaseTime', 'apiTime', 
            'cpuUsage_traffic', 'memoryUsage_traffic', 
            'cpuUsage_market', 'memoryUsage_market',
            'cpuUsage_trade', 'memoryUsage_trade'
        ]
        
        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if df[col].var() == 0: continue
                threshold = df[col].quantile(0.85)
                dataset[f'H_{col}'] = (df[col] > threshold).astype(int)

        dataset = dataset.astype(bool)

        frequent_itemsets = apriori(dataset, min_support=0.05, use_colnames=True, max_len=3)

        if frequent_itemsets.empty:
            return

        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
        
        if rules.empty:
            return
        
        persona_rules = rules[
            rules['antecedents'].apply(lambda x: any('Persona_' in s for s in x)) | 
            rules['consequents'].apply(lambda x: any('Persona_' in s for s in x))
        ].copy()

        persona_rules['pair_key'] = persona_rules.apply(lambda x: frozenset(list(x['antecedents']) + list(x['consequents'])), axis=1)
        persona_rules = persona_rules.sort_values(by='confidence', ascending=False).drop_duplicates(subset=['pair_key'])

        def clean_label(item_set):
            items = list(item_set)
            cleaned_items = [str(i).replace('Persona_', '') for i in items]
            return ', '.join(cleaned_items)

        persona_rules['antecedents'] = persona_rules['antecedents'].apply(clean_label)
        persona_rules['consequents'] = persona_rules['consequents'].apply(clean_label)
        
        persona_rules = persona_rules.sort_values(by='lift', ascending=False)
        cols_out = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        persona_rules[cols_out] = persona_rules[cols_out].round(4)

        if not persona_rules.empty:
            print(persona_rules[cols_out].head(15).to_string(index=False))
            save_text_report('association_rules_persona.txt', persona_rules[cols_out].to_string(index=False))

            plt.figure(figsize=(14, 10))
            G = nx.DiGraph()
            for _, row in persona_rules.head(20).iterrows():
                G.add_edge(row['antecedents'], row['consequents'], weight=row['lift'])
            
            pos = nx.spring_layout(G, k=2.0, seed=42)
            
            node_colors = []
            for node in G.nodes():
                if any(x in node for x in ['SCRAPER', 'TRADER', 'USER']):
                    node_colors.append('#ff9999')
                elif 'H_' in node: 
                    node_colors.append('#99ff99') 
                else: 
                    node_colors.append('#99ccff')

            nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                    node_size=3500, font_size=10, font_weight='bold', arrowsize=20)
            
            plt.title("Analiza asocjacji dla klas użytkowników")
            plt.savefig(os.path.join(OUTPUT_DIR, 'association_graph_persona.png'), bbox_inches='tight')
            plt.close()
        else:
            pass

    except Exception as e:
        print(e)
        import traceback
        traceback.print_exc()

def xgbTree(source_path, target_col):
    print(f"\nXGBoost dla {target_col}")
    input_file = os.path.join(source_path, 'merged_data.csv')
    try:
        df = pd.read_csv(input_file)
        
        if 'userPersona' in df.columns:
            df = pd.concat([df, pd.get_dummies(df['userPersona'], prefix='usr')], axis=1)

        df_numeric = df.select_dtypes(include=[np.number])

        target = target_col
        if target not in df_numeric.columns:
            candidates = [c for c in df_numeric.columns if target_col in c]
            if candidates:
                target = candidates[0]
            else:
                return

        df_numeric = df_numeric.dropna(subset=[target])
        df_numeric = df_numeric.fillna(method='ffill').fillna(0)

        if len(df_numeric) < 50: return

        X = df_numeric.drop(columns=[target])
        y = df_numeric[target]
        X = X.loc[:, X.var() > 0]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        report = f"Target: {target}\nMSE: {mse}\nR2: {r2}\n\nFeature Importance:\n"
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in range(min(15, X.shape[1])):
            report += f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}\n"
            
        save_text_report(f'xgb_results_{target}.txt', report)
        
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(model, max_num_features=20, title=f'Feature Importance ({target})')
        plt.savefig(os.path.join(OUTPUT_DIR, f'xgb_importance_{target}.png'), bbox_inches='tight')
        plt.close()
        print(f"Zapisano: xgb_importance_{target}.png")

    except Exception as e:
        print(e)

def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='.', help='Katalog z plikami CSV')
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        return

    OUTPUT_DIR = os.path.join(args.dir, 'EDA')
    ensure_dir(OUTPUT_DIR)
    
    sepReplicas(args.dir)
    
    if mergeData(args.dir):
        focusedPersonaCorrelation(args.dir)
        
        personaAnalysis(args.dir)
        association(args.dir)
        
        xgbTree(args.dir, 'databaseTime')
        xgbTree(args.dir, 'cpuUsage')
    else:
        pass


if __name__ == "__main__":
    main()