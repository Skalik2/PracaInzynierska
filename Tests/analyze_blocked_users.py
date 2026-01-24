import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

sns.set(style="whitegrid")

def ensure_output_dir(base_dir, subfolder='blocked_analysis'):
    out_dir = os.path.join(base_dir, subfolder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def load_blocked_data(directory, filename='blocked_user_log.csv'):
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        if 'userId' in df.columns:
            df['userId'] = df['userId'].astype(str)
        return df
    except Exception as e:
        print(e)
        return None

def load_user_classes(directory, filename='users_classes.csv'):
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        return None

    try:
        df = pd.read_csv(filepath)
        if 'userId' in df.columns:
            df['userId'] = df['userId'].astype(str)
        return df
    except Exception as e:
        print(e)
        return None

def generate_filtered_ban_plot(df_logs, df_classes, output_dir):
    if df_classes is None or df_logs is None:
        return

    ban_logs = df_logs[df_logs['reason'].str.contains("BOT DETECTED", na=False)].copy()
    ban_logs = ban_logs.sort_values('timestamp')
    unique_bans = ban_logs.drop_duplicates(subset=['userId'], keep='first')

    merged = pd.merge(unique_bans, df_classes, on='userId', how='left')
    merged['isBot'] = merged['isBot'].fillna(-1)

    bots_banned = merged[merged['isBot'] == 1].copy()
    humans_banned = merged[merged['isBot'] == 0].copy()

    plt.figure(figsize=(12, 7))

    def plot_cumulative(data, label, color, linestyle='-'):
        if not data.empty:
            data = data.sort_values('timestamp')
            data['count'] = range(1, len(data) + 1)
            plt.plot(data['timestamp'], data['count'], label=label, color=color, linestyle=linestyle, marker='o', markersize=3)
        else:
            plt.plot([], [], label=label + " (Brak)", color=color, linestyle=linestyle)

    plot_cumulative(bots_banned, f'Unikalne zablokowane Boty ({len(bots_banned)})', 'green')
    plot_cumulative(humans_banned, f'Unikalni zablokowani Ludzie ({len(humans_banned)})', 'red')

    plt.title("Przyrost zablokowanych użytkowników")
    plt.xlabel("Czas")
    plt.ylabel("Liczba użytkowników")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'bans_timeline_unique.png')
    plt.savefig(filename)
    plt.close()

def generate_blocked_attempts_timeline(df_logs, df_classes, output_dir):
    
    if df_classes is not None:
        merged = pd.merge(df_logs, df_classes, on='userId', how='left')
        merged['isBot'] = merged['isBot'].fillna(-1) 
    else:
        merged = df_logs.copy()
        merged['isBot'] = -1

    merged = merged.set_index('timestamp')
    
    resample_interval = '5s'
    
    bots = merged[merged['isBot'] == 1].resample(resample_interval).size()
    humans = merged[merged['isBot'] == 0].resample(resample_interval).size()
    unknown = merged[merged['isBot'] == -1].resample(resample_interval).size()
    
    plt.figure(figsize=(14, 7))
    
    if not bots.empty and bots.sum() > 0:
        plt.plot(bots.index, bots.values, color='green', label=f'Żądania od Botów ({bots.sum()})', linewidth=1.5)
        plt.fill_between(bots.index, bots.values, color='green', alpha=0.1)

    if not humans.empty and humans.sum() > 0:
        plt.plot(humans.index, humans.values, color='red', label=f'Żądania od Ludzi ({humans.sum()})', linewidth=1.5)
        plt.fill_between(humans.index, humans.values, color='red', alpha=0.1)

    if not unknown.empty and unknown.sum() > 0:
        plt.plot(unknown.index, unknown.values, color='gray', label=f'Nieznane ({unknown.sum()})', linestyle=':', alpha=0.5)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()

    plt.title(f"Intensywność zablokowanych żądań (wszystkie próby, interwał {resample_interval})")
    plt.xlabel("Czas")
    plt.ylabel("Liczba żądań")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = os.path.join(output_dir, 'blocked_attempts_timeline.png')
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Zapisano wykres timeline: {output_path}")

def generate_text_report(df_logs, df_classes, output_dir):
    report_path = os.path.join(output_dir, 'blocked_summary_stats.txt')
    
    if df_classes is not None:
        merged = pd.merge(df_logs, df_classes, on='userId', how='left')
        
        total_reqs = len(merged)
        bot_reqs = len(merged[merged['isBot'] == 1])
        human_reqs = len(merged[merged['isBot'] == 0])
        
        unique_bans = merged[merged['reason'].str.contains("BOT DETECTED", na=False)].drop_duplicates(subset=['userId'])
        unique_bots = len(unique_bans[unique_bans['isBot'] == 1])
        unique_humans = len(unique_bans[unique_bans['isBot'] == 0])
    else:
        total_reqs = len(df_logs)
        bot_reqs = 0
        human_reqs = 0
        unique_bots = 0
        unique_humans = 0

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Łączna liczba zablokowanych żądań: {total_reqs}\n")
        f.write(f" - Od SCRAPER_BOT: {bot_reqs}\n")
        f.write(f" - Od ACTIVE_USER i CAUTIOUS_USER (FP): {human_reqs}\n\n")
        
        f.write("--- UNIKALNE BANY (Pierwsze 'BOT DETECTED') ---\n")
        f.write(f"Zablokowane unikalne SCRAPER_BOT: {unique_bots}\n")
        f.write(f"Zablokowani unikalni ACTIVE_USER i CAUTIOUS_USER: {unique_humans}\n")
        
    print(f"Zapisano raport: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Analiza logów zablokowanych użytkowników.")
    parser.add_argument('--dir', type=str, default='.', help='Katalog z plikami CSV')
    parser.add_argument('--out', type=str, default='blocked_analysis', help='Katalog wyjściowy')
    args = parser.parse_args()

    df_logs = load_blocked_data(args.dir)
    df_classes = load_user_classes(args.dir)
    
    if df_logs is not None and not df_logs.empty:
        out_dir = ensure_output_dir(args.dir, args.out)
        
        generate_blocked_attempts_timeline(df_logs, df_classes, out_dir)
        generate_filtered_ban_plot(df_logs, df_classes, out_dir)
        generate_text_report(df_logs, df_classes, out_dir)
        
    else:
        print("Brak danych logów")

if __name__ == "__main__":
    main()