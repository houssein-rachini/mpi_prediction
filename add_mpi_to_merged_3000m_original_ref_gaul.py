import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ORIGINAL_FILE = BASE_DIR / 'unmasked_Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv'
MERGED_FILE = BASE_DIR / 'merged_all_vars_3000m_original_ref_gaul.csv'

COUNTRY_NAME_MAP = {
    'Tanzania': 'United Republic of Tanzania',
    'Tanzania, United Republic of': 'United Republic of Tanzania',
    'Bolivia, Plurinational State of': 'Bolivia',
    'Congo, Democratic Republic of the': 'Congo',
    'Democratic Republic of Congo': 'Congo',
    'Republic of Congo': 'Congo',
    'Congo, Republic of': 'Congo',
    "Cote d'Ivoire": "C?te d'Ivoire",
    'Lao': "Lao People's Democratic Republic",
    'Lao PDR': "Lao People's Democratic Republic",
    'Laos': "Lao People's Democratic Republic",
    'Macedonia': 'The former Yugoslav Republic of Macedonia',
    'TFYR of Macedonia': 'The former Yugoslav Republic of Macedonia',
    'North Macedonia': 'The former Yugoslav Republic of Macedonia',
    'Moldova': 'Moldova, Republic of',
    'Timor Leste': 'Timor-Leste',
    'Vietnam': 'Viet Nam',
}

KEY_COLS = ['Country', 'Region', 'Year']


def mpi_to_gaul(country: str) -> str:
    return COUNTRY_NAME_MAP.get(country, country)


def main() -> None:
    merged = pd.read_csv(MERGED_FILE, encoding='utf-8')
    original = pd.read_csv(ORIGINAL_FILE, encoding='utf-8')

    original = original[['Country', 'Region', 'Year', 'MPI']].copy()
    original['Country'] = original['Country'].map(mpi_to_gaul)
    original['Year'] = pd.to_numeric(original['Year'], errors='coerce').astype('Int64')
    merged['Year'] = pd.to_numeric(merged['Year'], errors='coerce').astype('Int64')

    original = original.drop_duplicates(KEY_COLS)

    if 'MPI' in merged.columns:
        merged = merged.drop(columns=['MPI'])

    merged = merged.merge(original, on=KEY_COLS, how='left')
    total_rows_before_filter = len(merged)
    merged = merged.dropna(subset=['MPI']).reset_index(drop=True)
    merged.to_csv(MERGED_FILE, index=False, encoding='utf-8')

    print({
        'output': str(MERGED_FILE),
        'rows': int(len(merged)),
        'removed_rows_without_mpi': int(total_rows_before_filter - len(merged)),
    })


if __name__ == '__main__':
    main()
