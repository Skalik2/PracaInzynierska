import re
import csv
import os
import argparse

def sql_copy_to_csv(sql_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with open(sql_file, 'r', encoding='utf-8') as file:
            sql_data = file.read()
    except Exception as e:
        print(e)
        return
    
    copy_statements = re.findall(r"COPY ([\w\.]+) \((.*?)\) FROM stdin;\n(.*?)(?=\\\.)", sql_data, re.DOTALL)
    
    if not copy_statements:
        return

    for table_name, columns, values in copy_statements:
        clean_table_name = table_name.split('.')[-1]
        print(f"Przetwarzanie tabeli: {clean_table_name} (z {table_name})")
        
        columns_list = [col.strip().strip('"') for col in columns.split(",")]
        rows = [row.split("\t") for row in values.strip().split("\n")]
        
        csv_file_path = os.path.join(output_folder, f"{clean_table_name}.csv")
        
        try:
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(columns_list)
                writer.writerows(rows)
        except Exception as e:
            print(e)

    print("\nZakończono konwersję")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--out', type=str, default='csv_output')
    args = parser.parse_args()
    
    sql_file_path = args.file

    if not sql_file_path:
        current_files = os.listdir('.')
        sql_files = [f for f in current_files if f.endswith('.sql')]
        if len(sql_files) >= 1:
            sql_file_path = sql_files[0]
            print(f"Wykryto plik SQL: {sql_file_path}")
        else:
            print("BŁĄD: Nie znaleziono pliku .sql.")
            return

    sql_copy_to_csv(sql_file_path, args.out)

if __name__ == "__main__":
    main()