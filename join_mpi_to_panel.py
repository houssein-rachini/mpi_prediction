"""
join_mpi_to_panel.py

Joins Combined_MPI_2012_2023.csv onto all_82_merged_500m_with_anomalies.csv.

Matching strategy:
  1. Normalize country names (both sides) to a common key.
  2. For each country, build a region mapping: MPI region -> panel region
     using exact match first, then rapidfuzz token_sort_ratio >= 70.
  3. Apply manual region overrides (REGION_FIX) for known low-score-but-correct pairs.
  4. Join on (country_key, panel_region_key, Year).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

FOLDER = Path(__file__).resolve().parent

PANEL_FIX = {
    'the former yugoslav republic of macedonia': 'north macedonia',
    'united republic of tanzania': 'tanzania',
}
MPI_FIX = {
    'tanzania, united republic of': 'tanzania',
    'tfyr of macedonia': 'north macedonia',
    'macedonia': 'north macedonia',
    "côte d'ivoire": "cote d'ivoire",
    "c?te d'ivoire": "cote d'ivoire",
    'lao': "lao people's democratic republic",
    'lao pdr': "lao people's democratic republic",
    'laos': "lao people's democratic republic",
    'congo, republic of': 'congo',
    'republic of congo': 'congo',
    'democratic republic of congo': 'congo, democratic republic of the',
    'eswatini': 'swaziland',
    'moldova': 'moldova, republic of',
    'timor leste': 'timor-leste',
    'syria': 'syrian arab republic',
    'bolivia, plurinational state of': 'bolivia',
    'viet nam': 'vietnam',
}

# Manual region overrides: {(country_key, mpi_region_key): panel_region_key}
# Used when fuzzy score falls below threshold but the match is unambiguous.
REGION_FIX: dict[tuple[str, str], str] = {
    # Angola — accent variant
    ('angola', 'bié'): 'bie',
    # Bangladesh — renamed/newly created divisions
    ('bangladesh', 'chattogram'): 'chittagong',
    ('bangladesh', 'mymensingh'): 'dhaka',
    # Benin — accent / spelling
    ('benin', 'ouémé'): 'oueme',
    ('benin', 'atacora'): 'atakora',
    # Burkina Faso
    ('burkina faso', 'hauts basins'): 'hauts-bassins',
    ('burkina faso', 'hauts-bassins'): 'hauts-bassins',
    # Cambodia — merged survey zones
    ('cambodia', 'mondol kiri/rattanak kiri'): 'ratanak kiri',
    ('cambodia', 'otdar mean chey'): 'otdar meanchey',
    ('cambodia', 'preah vihear/steung treng'): 'preah vihear',
    # Guinea — accent
    ('guinea', 'nézérékoré'): 'nzerekore',
    # Haiti — groupings
    ('haiti', 'aire métropolitaine/ouest'): 'ouest',
    ("haiti", "grand'anse"): 'grande anse',
    # Ethiopia — SNNPR long form → abbreviation
    ('ethiopia', "southern nations, nationalities, and people's region"): 'snnpr',
    # Cameroon — city-level surveys mapped to closest region
    ('cameroon', 'douala'): 'littoral',
    ('cameroon', 'yaounde'): 'centre',
    # Belize — city-level variants
    ('belize', 'belize (ex. belize city south side)'): 'belize',
    ('belize', 'belize (excluding belize city south side)'): 'belize',
    ('belize', 'belize city south side'): 'belize',
    # Azerbaijan
    ('azerbaijan', 'baku'): 'absheron',
    # Djibouti
    ('djibouti', 'other districts'): 'djibouti',
}

FUZZY_THRESHOLD = 70


def build_region_map(
    mpi_regions: list[str],
    panel_regions: list[str],
    country: str,
) -> dict[str, str]:
    """Return {mpi_region -> panel_region} for one country."""
    mapping: dict[str, str] = {}
    panel_set = set(panel_regions)
    for mr in mpi_regions:
        # 1. Manual override
        override = REGION_FIX.get((country, mr))
        if override and override in panel_set:
            mapping[mr] = override
            continue
        # 2. Exact match
        if mr in panel_set:
            mapping[mr] = mr
            continue
        # 3. Fuzzy match
        result = process.extractOne(mr, panel_regions, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= FUZZY_THRESHOLD:
            mapping[mr] = result[0]
    return mapping


def main() -> None:
    panel = pd.read_csv(
        FOLDER / 'gee_all_vars_added' / 'all_82_merged_500m_with_anomalies_MPI.csv',
        encoding='utf-8',
    )
    mpi = pd.read_csv(FOLDER / 'Combined_MPI_2012_2023.csv', encoding='utf-8')

    # Normalise country keys
    panel['_ckey'] = panel['Country'].str.strip().str.lower().map(lambda x: PANEL_FIX.get(x, x))
    panel['_rkey'] = panel['Region'].str.strip().str.lower()
    mpi['_ckey'] = mpi['Country'].str.strip().str.lower().map(lambda x: MPI_FIX.get(x, x))
    mpi['_rkey'] = mpi['Region'].str.strip().str.lower()

    # Build per-country region map and add _rkey_panel to mpi
    common_countries = set(mpi['_ckey'].unique()) & set(panel['_ckey'].unique())
    rkey_panel_map: dict[tuple[str, str], str] = {}

    for country in sorted(common_countries):
        panel_regs = list(panel[panel['_ckey'] == country]['_rkey'].unique())
        mpi_regs   = list(mpi[mpi['_ckey'] == country]['_rkey'].unique())
        reg_map = build_region_map(mpi_regs, panel_regs, country)
        for mr, pr in reg_map.items():
            rkey_panel_map[(country, mr)] = pr

    mpi['_rkey_panel'] = mpi.apply(
        lambda r: rkey_panel_map.get((r['_ckey'], r['_rkey'])), axis=1
    )

    mpi_rows_matched = mpi['_rkey_panel'].notna().sum()
    print(f'MPI rows with region match: {mpi_rows_matched}/{len(mpi)} ({mpi_rows_matched/len(mpi)*100:.1f}%)')

    # Join: panel._rkey matches mpi._rkey_panel
    mpi_to_join = (
        mpi[mpi['_rkey_panel'].notna()][['_ckey', '_rkey_panel', 'Year', 'MPI']]
        .rename(columns={'_rkey_panel': '_rkey'})
        .drop_duplicates(['_ckey', '_rkey', 'Year'])
    )

    if 'MPI' in panel.columns:
        panel = panel.drop(columns=['MPI'])

    merged = panel.merge(mpi_to_join, on=['_ckey', '_rkey', 'Year'], how='left')
    matched = merged['MPI'].notna().sum()
    print(f'Panel rows with MPI: {matched}/{len(merged)} ({matched/len(merged)*100:.1f}%)')
    print(f'Unique matched entities: {merged[merged["MPI"].notna()][["_ckey","_rkey"]].drop_duplicates().shape[0]}')

    unmatched_countries = merged[merged['MPI'].isna()]['Country'].nunique()
    print(f'Countries with no MPI for any row: {unmatched_countries}')

    merged = merged.drop(columns=['_ckey', '_rkey'])

    # Place MPI right after Year
    fixed = ['Country', 'Region', 'adm1_code', 'Year', 'MPI']
    rest = [c for c in merged.columns if c not in fixed]
    merged = merged[fixed + rest]

    out = FOLDER / 'gee_all_vars_added' / 'all_82_merged_500m_with_anomalies_MPI.csv'
    merged.to_csv(out, index=False, encoding='utf-8')
    print(f'Saved: {len(merged)} rows x {len(merged.columns)} cols -> {out.name}')


if __name__ == '__main__':
    main()
