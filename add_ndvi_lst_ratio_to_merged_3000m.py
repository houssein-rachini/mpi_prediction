import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FILES = [
    BASE_DIR / 'merged_all_vars_3000m_original_ref_gaul.csv',
]

NUM_COL = 'Median_NDVI'
DEN_COL = 'Mean_LST'
NEW_COL = 'ndvi_lst_ratio'


def main() -> None:
    for path in FILES:
        df = pd.read_csv(path, encoding='utf-8')
        if NUM_COL not in df.columns or DEN_COL not in df.columns:
            raise ValueError(f'{path.name} missing required columns')
        num = pd.to_numeric(df[NUM_COL], errors='coerce')
        den = pd.to_numeric(df[DEN_COL], errors='coerce')
        ratio = num / den
        ratio = ratio.where(den != 0)
        if NEW_COL in df.columns:
            df = df.drop(columns=[NEW_COL])
        insert_at = len(df.columns)
        if 'MPI' in df.columns:
            insert_at = df.columns.get_loc('MPI')
        df.insert(insert_at, NEW_COL, ratio)
        df.to_csv(path, index=False, encoding='utf-8')
        print({'file': str(path), 'rows': int(len(df)), 'non_null_ratio_rows': int(df[NEW_COL].notna().sum())})


if __name__ == '__main__':
    main()
