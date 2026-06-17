"""
Download WorldPop Global 2 (R2025A, constrained, 100m) GeoTIFFs
for all 82 GAUL training countries, years 2021-2023.

Output folder: worldpop_g2_100m/{year}/{ISO3}/
File pattern:  {iso3_lower}_pop_{year}_CN_100m_R2025A_v1.tif

Estimated total download: 30-50 GB.

Run (dry-run first to see sizes):
    python download_worldpop_g2.py --dry-run
Run (actual download, 4 parallel workers):
    python download_worldpop_g2.py
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import country_converter as coco
import requests
from tqdm import tqdm

BASE_DIR   = Path(__file__).resolve().parent
OUT_DIR    = BASE_DIR / "worldpop_g2_100m"
TARGET_YEARS = [2021, 2022, 2023]
MAX_WORKERS  = 4

GAUL_COUNTRIES = [
    "Afghanistan", "Albania", "Angola", "Azerbaijan", "Bangladesh",
    "Belize", "Benin", "Bhutan", "Bolivia", "Botswana", "Brazil",
    "Burkina Faso", "Burundi", "Cambodia", "Cameroon",
    "Central African Republic", "Chad", "Congo", "Costa Rica", "Cuba",
    "Djibouti", "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Ethiopia", "Fiji", "Gabon", "Ghana", "Guatemala", "Guinea",
    "Guinea-Bissau", "Haiti", "Honduras", "India", "Indonesia", "Iraq",
    "Jamaica", "Jordan", "Kenya", "Kyrgyzstan",
    "Lao People's Democratic Republic", "Lesotho", "Liberia", "Madagascar",
    "Mali", "Mauritania", "Mexico", "Moldova, Republic of", "Mongolia",
    "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal", "Nicaragua",
    "Niger", "Nigeria", "Pakistan", "Papua New Guinea", "Paraguay", "Peru",
    "Senegal", "Sierra Leone", "South Sudan", "Sri Lanka", "Sudan",
    "Suriname", "Swaziland", "Syrian Arab Republic", "Tajikistan",
    "Thailand", "Timor-Leste", "Togo", "Trinidad and Tobago", "Tunisia",
    "Turkmenistan", "Uganda", "Uzbekistan", "Yemen", "Zambia", "Zimbabwe",
]

NAME_FIX = {
    "Congo":                             "Democratic Republic of the Congo",
    "Lao People's Democratic Republic":  "Laos",
    "Moldova, Republic of":              "Moldova",
    "Swaziland":                         "Eswatini",
    "Syrian Arab Republic":              "Syria",
    "South Sudan":                       "South Sudan",
    "Guinea-Bissau":                     "Guinea-Bissau",
    "Timor-Leste":                       "Timor-Leste",
    "Trinidad and Tobago":               "Trinidad and Tobago",
}

WP_BASE = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A"


def build_iso3_map() -> dict[str, str]:
    cc = coco.CountryConverter()
    result = {}
    for name in GAUL_COUNTRIES:
        lookup = NAME_FIX.get(name, name)
        iso3   = cc.convert(lookup, to="ISO3", not_found=None)
        if iso3 is None or iso3 == "not found":
            print(f"  WARNING: ISO3 not found for '{name}' (tried '{lookup}')")
        else:
            result[name] = iso3
    return result


def make_url(iso3: str, year: int) -> str:
    return (f"{WP_BASE}/{year}/{iso3}/v1/100m/constrained/"
            f"{iso3.lower()}_pop_{year}_CN_100m_R2025A_v1.tif")


def make_path(iso3: str, year: int) -> Path:
    return OUT_DIR / str(year) / iso3 / f"{iso3.lower()}_pop_{year}_CN_100m_R2025A_v1.tif"


def download_file(url: str, dest: Path) -> tuple[str, bool, str]:
    """Download url → dest. Returns (url, success, message)."""
    if dest.exists():
        return url, True, f"already exists ({dest.stat().st_size / 1e6:.0f} MB)"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    f.write(chunk)
        size_mb = dest.stat().st_size / 1e6
        return url, True, f"OK ({size_mb:.0f} MB)"
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        return url, False, str(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show URLs and sizes without downloading.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--years", nargs="+", type=int, default=TARGET_YEARS,
                        help="Years to download (default: 2021 2022 2023).")
    args = parser.parse_args()

    print("Building ISO3 map ...")
    iso3_map = build_iso3_map()
    print(f"  {len(iso3_map)} countries resolved.\n")

    jobs = [
        (country, iso3, year)
        for country, iso3 in sorted(iso3_map.items())
        for year in args.years
    ]

    if args.dry_run:
        print(f"{'Country':<35} {'ISO3':<6} {'Year':<6} {'URL'}")
        print("-" * 100)
        total_mb = 0
        for country, iso3, year in jobs:
            url = make_url(iso3, year)
            try:
                r = requests.head(url, timeout=10)
                mb = int(r.headers.get("Content-Length", 0)) / 1e6
                status = f"{mb:.0f} MB" if r.status_code == 200 else f"HTTP {r.status_code}"
                total_mb += mb
            except Exception as e:
                status = f"ERROR: {e}"
            print(f"{country:<35} {iso3:<6} {year:<6} {status}")
        print(f"\nEstimated total: {total_mb / 1024:.1f} GB")
        return

    # Real download
    print(f"Downloading {len(jobs)} files to {OUT_DIR}")
    print(f"Workers: {args.workers} | Years: {args.years}\n")

    futures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for country, iso3, year in jobs:
            url  = make_url(iso3, year)
            dest = make_path(iso3, year)
            fut  = ex.submit(download_file, url, dest)
            futures[fut] = (country, iso3, year)

        failed = []
        with tqdm(total=len(jobs), unit="file") as pbar:
            for fut in as_completed(futures):
                country, iso3, year = futures[fut]
                url, ok, msg = fut.result()
                status = "OK" if ok else "FAIL"
                pbar.set_postfix(country=country, year=year, status=status)
                if not ok:
                    failed.append((country, iso3, year, msg))
                pbar.update(1)

    print(f"\nDone. {len(jobs) - len(failed)}/{len(jobs)} succeeded.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for country, iso3, year, msg in failed:
            print(f"  {country} ({iso3}) {year}: {msg}")


if __name__ == "__main__":
    main()
