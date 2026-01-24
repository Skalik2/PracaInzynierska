import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

INPUT_FILE = '../csv_output/merged_data.csv'
OUTPUT_DIR = 'behavioral_analysis'

def analyze_cpu_impact():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(INPUT_FILE):
        print(f"Błąd: Nie znaleziono pliku {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, on_bad_lines='skip', low_memory=False)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df['is_bot'] = df['userPersona'].astype(str).apply(lambda x: 1 if 'SCRAPER_BOT' in x else 0)

    df['second'] = df['timestamp'].dt.floor('S')
    agg_df = df.groupby('second').agg({
        'id': 'count',
        'cpuUsage': 'mean',
        'is_bot': 'max'
    }).rename(columns={'id': 'RPS'})

    agg_df['Typ Użytkownika'] = agg_df['is_bot'].map({
        0: 'CAUTIOUS_USER, ACTIVE_TRADER', 
        1: 'SCRAPER_BOT'
    })

    plt.figure(figsize=(11, 7))
    
    sns.scatterplot(
        data=agg_df, 
        x='RPS', 
        y='cpuUsage', 
        hue='Typ Użytkownika', 
        style='Typ Użytkownika',
        palette={'CAUTIOUS_USER, ACTIVE_TRADER': '#1f77b4', 'SCRAPER_BOT': '#d62728'},
        s=80,
        alpha=0.7
    )
    
    plt.title('Wpływ liczby zapytań (RPS) na obciążenie CPU', fontsize=14)
    plt.xlabel('Liczba zapytań na sekundę (RPS)', fontsize=12)
    plt.ylabel('Średnie zużycie CPU [%]', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Grupa')
    
    output_path = f"{OUTPUT_DIR}/cpu_impact.png"
    plt.savefig(output_path)

if __name__ == "__main__":
    analyze_cpu_impact()