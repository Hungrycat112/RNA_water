import os
import pandas as pd
import argparse
import logging

def find_csv_files(directory):
    """Recursively find all CSV files in the given directory."""
    return [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files]

def combine_and_save(input_dir, output_file):
    """Combine all CSV files from the given directory (including subdirectories) into one CSV file."""
    if not os.path.isdir(input_dir):
        logging.error(f"Invalid directory: {input_dir}")
        return

    csv_files = find_csv_files(input_dir)

    if not csv_files:
        logging.warning("No CSV files found. Output file not created.")
        return

    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, low_memory=False).dropna(how='all')
            dfs.append(df)
            logging.info(f"Loaded: {file_path} ({len(df)} rows)")
        except Exception as e:
            logging.error(f"Failed to read {file_path}: {e}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        logging.info(f"Combined CSV saved to {output_file} ({len(combined_df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine all CSV files from a directory (including subdirectories).")
    parser.add_argument("input_dir", help="Directory containing CSV files")
    parser.add_argument("output_file", help="Output CSV file path")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    combine_and_save(args.input_dir, args.output_file)
