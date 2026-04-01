import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / 'gee_all_vars_3000m_original_ref_gaul'
OUTPUT_FILE = BASE_DIR / 'merged_all_vars_3000m_original_ref_gaul.csv'

FILES = {
    'population': INPUT_DIR / 'all_82_population_3000m_original_ref_gaul_actual.csv',
    'gpp': INPUT_DIR / 'all_82_gpp_3000m_original_ref_gaul.csv',
    'lst': INPUT_DIR / 'all_82_lst_3000m_original_ref_gaul.csv',
    'ntl': INPUT_DIR / 'all_82_ntl_3000m_original_ref_gaul.csv',
    'ndvi': INPUT_DIR / 'all_82_ndvi_3000m_original_ref_gaul.csv',
}

EXTRA_COLS = ['system:index', '.geo']
KEY_COLS = ['Country', 'Region', 'Year']


def load_metric(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    df = df.drop(columns=[c for c in EXTRA_COLS if c in df.columns], errors='ignore')
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    return df


def main() -> None:
    missing = [str(path) for path in FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing input files: ' + '; '.join(missing))

    merged = load_metric(FILES['population'])

    for metric in ['gpp', 'lst', 'ntl', 'ndvi']:
        df = load_metric(FILES[metric])
        merged = merged.merge(df, on=KEY_COLS, how='outer')

    merged = merged.sort_values(KEY_COLS).reset_index(drop=True)
    merged.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

    print({
        'output': str(OUTPUT_FILE),
        'rows': int(len(merged)),
        'columns': list(merged.columns),
    })


if __name__ == '__main__':
    main()
