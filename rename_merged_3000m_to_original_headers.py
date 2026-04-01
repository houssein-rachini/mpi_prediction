import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FILES = [
    BASE_DIR / 'merged_all_vars_3000m_original_ref_gaul.csv',
]

RENAME_MAP = {
    'Mean Population': 'Mean_Pop',
    'Total Population': 'Total_Pop',
    'Min Population': 'Min_Pop',
    'Max Population': 'Max_Pop',
    'Median Population': 'Median_Pop',
    'Std Dev Population': 'StdDev_Pop',
    'Max GPP': 'Max_GPP',
    'Median GPP': 'Median_GPP',
    'Min GPP': 'Min_GPP',
    'Std Dev GPP': 'StdDev_GPP',
    'Total GPP': 'Sum_GPP',
    'Mean GPP': 'Mean_GPP',
    'Max LST (K)': 'Max_LST',
    'Mean LST (K)': 'Mean_LST',
    'Median LST (K)': 'Median_LST',
    'Min LST (K)': 'Min_LST',
    'Std Dev LST': 'StdDev_LST',
    'Total LST': 'Sum_LST',
    'Max NTL': 'Max_NTL',
    'Mean NTL': 'Mean_NTL',
    'Median NTL': 'Median_NTL',
    'Min NTL': 'Min_NTL',
    'Std Dev NTL': 'StdDev_NTL',
    'Total NTL': 'Sum_NTL',
    'Max NDVI': 'Max_NDVI',
    'Mean NDVI': 'Mean_NDVI',
    'Median NDVI': 'Median_NDVI',
    'Min NDVI': 'Min_NDVI',
    'Std Dev NDVI': 'StdDev_NDVI',
    'Total NDVI': 'Sum_NDVI',
}

COLUMN_ORDER = [
    'Country', 'Region', 'Year',
    'Max_GPP', 'Median_GPP', 'Min_GPP', 'StdDev_GPP', 'Sum_GPP', 'Mean_GPP',
    'Total_Pop', 'Mean_Pop', 'Min_Pop', 'Max_Pop', 'Median_Pop', 'StdDev_Pop',
    'Max_LST', 'Mean_LST', 'Median_LST', 'Min_LST', 'StdDev_LST', 'Sum_LST',
    'Max_NTL', 'Mean_NTL', 'Median_NTL', 'Min_NTL', 'StdDev_NTL', 'Sum_NTL',
    'Max_NDVI', 'Mean_NDVI', 'Median_NDVI', 'Min_NDVI', 'StdDev_NDVI', 'Sum_NDVI',
    'MPI',
]


def main() -> None:
    for path in FILES:
        df = pd.read_csv(path, encoding='utf-8')
        df = df.rename(columns=RENAME_MAP)
        missing = [col for col in COLUMN_ORDER if col not in df.columns]
        if missing:
            raise ValueError(f'{path.name} missing columns after rename: {missing}')
        df = df[COLUMN_ORDER]
        df.to_csv(path, index=False, encoding='utf-8')
        print({'file': str(path), 'rows': int(len(df))})


if __name__ == '__main__':
    main()
