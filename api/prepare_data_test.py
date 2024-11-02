import pandas as pd
from pathlib import Path

# Constants for file paths
INPUT_DIR = Path('../input_data')
OUTPUT_DIR = Path('test_data')
ANEXO1_FILE = INPUT_DIR / 'anexo1_daily.parquet'
ANEXO3_FILE = INPUT_DIR / 'anexo3_daily.parquet'
MOSSORO_FILE = INPUT_DIR / 'mossoro_daily.parquet'
PARNAMIRIM_FILE = INPUT_DIR / 'parnamirim_daily.parquet'
TRAIN_FILE = OUTPUT_DIR / 'df_train.parquet'
TEST_FILE = OUTPUT_DIR / 'df_test.parquet'

def load_data(file_path):
    """Load data from a parquet file."""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def rename_columns(df, suffix):
    """Rename columns by adding a suffix."""
    df.columns = [f"{c}_{suffix}" for c in df.columns]
    return df

def prepare_data():
    """Prepare and save train and test datasets."""
    # Load data
    df_anexo1 = load_data(ANEXO1_FILE)
    df_anexo3 = load_data(ANEXO3_FILE)
    df_mossoro = load_data(MOSSORO_FILE)
    df_parnamirim = load_data(PARNAMIRIM_FILE)

    # Rename columns
    df_anexo1 = rename_columns(df_anexo1, 'anexo1')
    df_anexo3 = rename_columns(df_anexo3, 'anexo3')
    df_mossoro = rename_columns(df_mossoro, 'mossoro')
    df_parnamirim = rename_columns(df_parnamirim, 'parnamirim')

    # Concatenate dataframes
    df = pd.concat([df_anexo1, df_anexo3, df_mossoro, df_parnamirim], axis=1)

    # Reindex to fill missing dates
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(all_days, fill_value=0)

    # Split into train and test sets
    df_train = df.iloc[:-15]
    df_test = df.iloc[-15:]

    # Save to parquet files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_parquet(TRAIN_FILE)
    df_test.to_parquet(TEST_FILE)

if __name__ == "__main__":
    prepare_data()