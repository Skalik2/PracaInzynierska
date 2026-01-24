import argparse
import pandas as pd
import os

def generate_user_labels(target_dir):
    input_path = os.path.join(target_dir, 'market_log.csv')
    output_path = os.path.join(target_dir, 'users_classes.csv')

    if not os.path.exists(input_path):
        print(f"Błąd: Plik '{input_path}' nie istnieje.")
        return

    try:
        df = pd.read_csv(input_path, dtype={'userId': str, 'userPersona': str})
        
        df_users = df[ (df['userId'] != '\\N') & (df['userId'].notna()) ].copy()
        
        if df_users.empty:
            return

        def is_entry_bot(persona):
            if pd.isna(persona):
                return 0
            if 'BOT' in str(persona).upper():
                return 1
            return 0

        df_users['isBotEntry'] = df_users['userPersona'].apply(is_entry_bot)

        user_classes = df_users.groupby('userId')['isBotEntry'].max().reset_index()
        
        user_classes.rename(columns={'isBotEntry': 'isBot'}, inplace=True)
        
        try:
            user_classes['sort_key'] = pd.to_numeric(user_classes['userId'])
        except ValueError:
            user_classes['sort_key'] = user_classes['userId']
            
        user_classes = user_classes.sort_values(by='sort_key').drop(columns=['sort_key'])
        
        user_classes.to_csv(output_path, index=False)
        
        print(f"Liczba użytkowników: {len(user_classes)}")
        print(f"Liczba wykrytych botów: {user_classes['isBot'].sum()}")

    except Exception as e:
        print(e)

def main():
    parser = argparse.ArgumentParser(description="Klasyfikator Bot/Human na podstawie logów.")
    parser.add_argument('--dir', type=str, default='.', help='Katalog z market_log.csv')
    args = parser.parse_args()

    target_directory = args.dir
    if not os.path.exists(target_directory):
        print(f"Katalog '{target_directory}' nie istnieje.")
        return

    generate_user_labels(target_directory)

if __name__ == "__main__":
    main()