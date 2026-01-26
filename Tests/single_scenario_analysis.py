import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def ensure_output_dir(base_dir):
    plots_dir = os.path.join(base_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

def load_data(directory):
    required_files = {
        'complete_market_log': 'complete_market_log_csv.csv',
        'market_cpu': 'market_cpu.csv',
        'market_log': 'market_log.csv',
        'traffic_cpu': 'traffic_cpu.csv',
        'sum_trade_log': 'sum_trade_log.csv',
        'trade_cpu': 'trade_cpu.csv',
        'trade_log': 'trade_log.csv'
    }
    
    data = {}
    missing_files = []

    print(f"Wczytywanie danych z katalogu: {directory}")

    for key, filename in required_files.items():
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath, na_values=['\\N'])
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
                    df = df.sort_values('timestamp')
                data[key] = df
                print(f"Załadowano: {filename}")
            except Exception as e:
                print(e)
        else:
            missing_files.append(filename)

    return data

def draw_api_time_and_cpu(market_log, market_cpu, output_dir):
    if market_log is None or market_cpu is None: return

    merged_data = pd.merge_asof(market_log, market_cpu, on='timestamp', direction='nearest')

    for method in ['GET', 'POST']:
        method_data = merged_data[merged_data['apiMethod'] == method]
        if method_data.empty: continue

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Czas')
        ax1.set_ylabel(f'Czas odpowiedz API {method} (ms)', color='blue')
        ax1.plot(method_data['timestamp'], method_data['apiTime'], color='blue', label='API Time', alpha=0.6)
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Użycie CPU (%)', color='red')
        ax2.plot(method_data['timestamp'], method_data['cpuUsage'], color='red', label='CPU Usage', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title(f'{method} Requests: Czas odpowiedzi API vs Użycie CPU')
        
        filename = f'api_time_cpu_{method}.png'
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def draw_app_db_time(market_log, market_cpu, output_dir):
    if market_log is None or market_cpu is None: return

    ml_resampled = market_log.resample('15s', on='timestamp').mean(numeric_only=True).reset_index()
    mc_resampled = market_cpu.resample('15s', on='timestamp').mean(numeric_only=True).reset_index()

    merged_data = pd.merge_asof(ml_resampled, mc_resampled, on='timestamp', direction='nearest').dropna()

    if merged_data.empty: return

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Czas')
    ax1.set_ylabel('Czas (ms)', color='blue')
    ax1.plot(merged_data['timestamp'], merged_data['applicationTime'], label='Application Time', color='blue')
    ax1.plot(merged_data['timestamp'], merged_data['databaseTime'], label='Database Time', color='green', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Użycie CPU (%)', color='red')
    ax2.plot(merged_data['timestamp'], merged_data['cpuUsage'], color='red', label='CPU Usage', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='red')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.title('Średni czas Aplikacji/Bazy vs CPU')
    plt.savefig(os.path.join(output_dir, 'app_db_time_cpu.png'))
    plt.close()

def draw_persona_metrics(complete_market_log, output_dir):
    print("Generowanie wykresów dla klas użytkowników")
    if complete_market_log is None or 'userPersona' not in complete_market_log.columns:
        return

    df = complete_market_log.copy()
    
    df_count = df.groupby([pd.Grouper(key='timestamp', freq='1min'), 'userPersona']).size().reset_index(name='count')
    
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_count, x='timestamp', y='count', hue='userPersona', marker='o')
    
    plt.title('Natężenie ruchu (Liczba zapytań/min) wg klasy użytkownika', fontsize=20, fontweight='bold')
    
    plt.ylabel('Liczba zapytań', fontsize=16)
    
    plt.xticks(fontsize=14, rotation=45) 
    plt.yticks(fontsize=14)
    
    plt.legend(title='Klasa użytkownika', fontsize=14, title_fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'persona_traffic_load.png'))
    plt.close()

def draw_offers_and_cpu(trade_log, trade_cpu, market_cpu, output_dir):
    if trade_log is None or trade_cpu is None or market_cpu is None:
        return

    m_cpu_ready = market_cpu.copy().sort_values('timestamp')
    m_cpu_ready = m_cpu_ready.rename(columns={'cpuUsage': 'cpuUsage_market'})

    if 'numberOfSellOffers' in trade_log.columns:
        trade_log['totalOffers'] = trade_log['numberOfSellOffers'] + trade_log['numberOfBuyOffers']
    else:
        return

    replicas = trade_log['replicaId'].unique() if 'replicaId' in trade_log.columns else [None]

    for replica in replicas:
        if replica is not None:
            t_log = trade_log[trade_log['replicaId'] == replica]
            t_cpu = trade_cpu[trade_cpu['replicaId'] == replica] if 'replicaId' in trade_cpu.columns else trade_cpu
            suffix = f"_replica_{replica}"
        else:
            t_log = trade_log
            t_cpu = trade_cpu
            suffix = ""

        t_cpu_renamed = t_cpu.copy().rename(columns={'cpuUsage': 'cpuUsage_trade'})

        merged_step1 = pd.merge_asof(t_log, t_cpu_renamed, on='timestamp', direction='nearest')
        
        merged_data = pd.merge_asof(merged_step1, m_cpu_ready, on='timestamp', direction='nearest')

        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Czas')
        ax1.set_ylabel('Suma ofert (Trade)', color='blue')
        ax1.plot(merged_data['timestamp'], merged_data['totalOffers'], color='blue', label='Total Offers', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('CPU Usage (%)', color='black')
        
        ax2.plot(merged_data['timestamp'], merged_data['cpuUsage_trade'], color='red', label='Trade CPU', alpha=0.7)
        
        ax2.plot(merged_data['timestamp'], merged_data['cpuUsage_market'], color='green', linestyle='--', label='Market CPU', alpha=0.7)
        
        ax2.tick_params(axis='y', labelcolor='black')

        plt.title(f'Liczba ofert a CPU Trade, CPU Market {suffix}')
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.savefig(os.path.join(output_dir, f'offers_cpu_market{suffix}.png'))
        plt.close()

def draw_average_response_time(complete_market_log, output_dir):
    if complete_market_log is None: return

    if 'userPersona' in complete_market_log.columns:
        agg_data = complete_market_log.groupby(['endpointUrl', 'userPersona'])['apiTime'].mean().reset_index()
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='endpointUrl', y='apiTime', hue='userPersona', data=agg_data)
        
        plt.title('Średni czas odpowiedzi API dla Endpoint\'u i Klasy użytkownika', fontsize=20, fontweight='bold')
        
        plt.ylabel('Średni czas (ms)', fontsize=16)
        
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.legend(title='Klasa użytkownika', fontsize=14, title_fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avg_response_time_persona.png'))
        plt.close()
    
    agg_data_simple = complete_market_log.groupby(['apiMethod', 'endpointUrl'])['apiTime'].mean().reset_index()
    plt.figure(figsize=(14, 7))
    sns.barplot(x='apiMethod', y='apiTime', hue='endpointUrl', data=agg_data_simple)
    plt.title('Średni czas odpowiedzi API (Total)')
    plt.savefig(os.path.join(output_dir, 'avg_response_time.png'))
    plt.close()

def draw_traffic_cpu(traffic_cpu, output_dir):
    if traffic_cpu is None: return

    plt.figure(figsize=(12, 6))
    plt.plot(traffic_cpu['timestamp'], traffic_cpu['cpuUsage'], label='CPU Usage')
    plt.plot(traffic_cpu['timestamp'], traffic_cpu['memoryUsage'], label='Memory Usage')
    plt.title('Zużycie zasobów - Traffic Service')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'traffic_resources.png'))
    plt.close()

def draw_offers_per_min(trade_log, market_cpu, output_dir):
    if trade_log is None or market_cpu is None: return

    tl_resampled = trade_log.resample('1min', on='timestamp').sum(numeric_only=True).reset_index()
    mc_resampled = market_cpu.resample('1min', on='timestamp').mean(numeric_only=True).reset_index()

    merged = pd.merge(tl_resampled, mc_resampled, on='timestamp')

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_xlabel('Czas (minuty)')
    ax1.set_ylabel('CPU Market (%)', color='blue')
    ax1.plot(merged['timestamp'], merged['cpuUsage'], color='blue', label='CPU Market')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Liczba ofert / min', color='green')
    ax2.plot(merged['timestamp'], merged['numberOfBuyOffers'], color='orange', label='Buy Offers')
    ax2.plot(merged['timestamp'], merged['numberOfSellOffers'], color='red', label='Sell Offers')
    ax2.tick_params(axis='y', labelcolor='green')

    plt.title('CPU Market vs Ofert na minutę')
    plt.savefig(os.path.join(output_dir, 'throughput_per_minute.png'))
    plt.close()

def draw_market_ram(market_cpu, output_dir):
    print("Generowanie wykresu RAM")
    
    if market_cpu is None:
        return
        
    if 'memoryUsage' not in market_cpu.columns:
        return

    plt.figure(figsize=(12, 6))
    
    sns.lineplot(data=market_cpu, x='timestamp', y='memoryUsage', color='purple', label='Memory Usage')
    
    plt.title('Zużycie pamięci RAM - Market')
    plt.xlabel('Czas')
    plt.ylabel('Zużycie RAM (%)')
    plt.legend()
    plt.grid(True)
    
    filename = 'market_ram.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    plt.close()
    
    print(f"Zapisano wykres: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Ścieżka do katalogu z CSV')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Błąd: Katalog '{args.dir}' nie istnieje.")
        return

    output_dir = ensure_output_dir(args.dir)
    data = load_data(args.dir)

    print("\n--- Generowanie wykresów ---")
    draw_market_ram(data.get('market_cpu'), output_dir)
    draw_api_time_and_cpu(data.get('complete_market_log'), data.get('market_cpu'), output_dir)
    draw_app_db_time(data.get('market_log'), data.get('market_cpu'), output_dir)
    draw_persona_metrics(data.get('complete_market_log'), output_dir)
    draw_offers_and_cpu(data.get('sum_trade_log'), data.get('trade_cpu'), data.get('market_cpu'), output_dir)
    draw_average_response_time(data.get('complete_market_log'), output_dir)
    draw_traffic_cpu(data.get('traffic_cpu'), output_dir)
    draw_offers_per_min(data.get('trade_log'), data.get('market_cpu'), output_dir)

    print(f"\nZakończono. Wykresy zapisane w: {output_dir}")

if __name__ == "__main__":
    main()