import pandas as pd
import numpy as np
from decimal import Decimal
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / 'gee_all_vars_3000m_original_ref_gaul' / 'all_82_population_3000m_original_ref_gaul_actual.csv'
TARGET_YEARS = [2021, 2022, 2023]
METRIC_COLS = [
    'Mean Population',
    'Total Population',
    'Min Population',
    'Max Population',
    'Median Population',
    'Std Dev Population',
]
ID_COLS = ['Country', 'Region', 'Year']
OPTIONAL_COLS = ['system:index', '.geo']


def normalize_precision(value):
    if pd.isna(value):
        return np.nan
    text = format(float(value), '.15f').rstrip('0').rstrip('.')
    if '.' not in text:
        return float(value)
    decimals = len(text.split('.', 1)[1])
    if decimals <= 6:
        return float(value)
    return round(float(value), 6)


def extrapolate(values, years, target_year):
    values = np.asarray(values, dtype=float)
    years = np.asarray(years, dtype=int)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    valid_vals = values[mask]
    valid_years = years[mask]
    year_diffs = np.diff(valid_years)
    value_diffs = np.diff(valid_vals)
    valid_growth = year_diffs != 0
    if not valid_growth.any():
        return np.nan
    growth = np.mean(value_diffs[valid_growth] / year_diffs[valid_growth])
    return normalize_precision(valid_vals[-1] + growth * (target_year - valid_years[-1]))


def main() -> None:
    df = pd.read_csv(FILE, encoding='utf-8')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')

    # Remove any existing target-year rows before rebuilding them.
    df = df[~df['Year'].isin(TARGET_YEARS)].copy()

    base = df[df['Year'] <= 2020].copy()
    new_rows = []

    for (country, region), group in base.groupby(['Country', 'Region'], dropna=False):
        group = group.sort_values('Year')
        years = group['Year'].astype(int).to_numpy()
        for target_year in TARGET_YEARS:
            row = {col: np.nan for col in df.columns}
            row['Country'] = country
            row['Region'] = region
            row['Year'] = target_year
            for col in OPTIONAL_COLS:
                if col in df.columns:
                    row[col] = np.nan
            for col in METRIC_COLS:
                row[col] = extrapolate(group[col].values, years, target_year)
            new_rows.append(row)

    if new_rows:
        add_df = pd.DataFrame(new_rows)
        df = pd.concat([df, add_df], ignore_index=True)

    if 'system:index' in df.columns or '.geo' in df.columns:
        ordered = [c for c in ['system:index', 'Country', 'Max Population', 'Mean Population', 'Median Population', 'Min Population', 'Region', 'Std Dev Population', 'Total Population', 'Year', '.geo'] if c in df.columns]
        other = [c for c in df.columns if c not in ordered]
        df = df[ordered + other]

    df = df.sort_values(['Country', 'Region', 'Year']).reset_index(drop=True)
    df.to_csv(FILE, index=False, encoding='utf-8')
    print({'output': str(FILE), 'rows': int(len(df)), 'max_year': int(df['Year'].max())})


if __name__ == '__main__':
    main()
