"""
gee_export_mask_area_comparison.py

Compares building-mask coverage per ADM2 between two GHSL epochs
(both from the P2023A release -- P2025A is not yet available on GEE):
  - 2018 epoch: JRC/GHSL/P2023A/GHS_BUILT_H/2018  (currently used)
  - 2020 epoch: JRC/GHSL/P2023A/GHS_BUILT_H/2020  (most recent available)

Mask definition (same as production): built_height >= 2.5 m, focalMax 500 m.

For each ADM2 in the 9 training countries, exports:
  masked_area_m2  -- sum of pixel areas (m2) inside the dilated mask
  total_area_m2   -- total ADM2 geometry area (m2)
  mask_fraction   -- masked_area_m2 / total_area_m2

Output Drive folder: gaul9_mask_area_comparison
  adm2_mask_area_2018.csv
  adm2_mask_area_2020.csv

Run:
    python gee_export_mask_area_comparison.py --start-tasks
"""

from __future__ import annotations

import argparse
import ee

COUNTRIES = [
    "Bosnia and Herzegovina", "Egypt", "Jordan", "Kyrgyzstan",
    "Montenegro", "Morocco", "Tajikistan", "Tunisia", "Turkey",
]
DRIVE_FOLDER = "gaul9_mask_area_comparison"
BUFFER_M = 500
HEIGHT_THRESH = 2.5


def initialize_ee() -> None:
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def build_mask(epoch: int) -> ee.Image:
    return (
        ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_H/{epoch}")
        .select("built_height")
        .gte(HEIGHT_THRESH)
        .focalMax(kernel=ee.Kernel.circle(radius=BUFFER_M, units="meters"))
        .selfMask()
    )


def regions_fc() -> ee.FeatureCollection:
    gaul2 = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2")
    return (
        gaul2
        .filter(ee.Filter.inList("ADM0_NAME", COUNTRIES))
        .select(["ADM0_NAME", "ADM1_CODE", "ADM2_CODE", "ADM2_NAME"])
    )


def export_mask_area(
    mask: ee.Image,
    regions: ee.FeatureCollection,
    description: str,
    folder: str,
) -> ee.batch.Task:
    """Compute masked_area_m2, total_area_m2, mask_fraction per ADM2."""
    pixel_area = ee.Image.pixelArea().updateMask(mask)

    reduced = pixel_area.rename("masked_area_m2").reduceRegions(
        collection=regions,
        reducer=ee.Reducer.sum(),
        scale=10,
        tileScale=4,
    )

    def _add_fields(f: ee.Feature) -> ee.Feature:
        total  = f.geometry().area(1)
        masked = ee.Number(f.get("sum"))
        frac   = masked.divide(total)
        return ee.Feature(None, {
            "Country":        f.get("ADM0_NAME"),
            "ADM1_CODE":      f.get("ADM1_CODE"),
            "ADM2_CODE":      f.get("ADM2_CODE"),
            "ADM2_NAME":      f.get("ADM2_NAME"),
            "masked_area_m2": masked,
            "total_area_m2":  total,
            "mask_fraction":  frac,
        })

    return ee.batch.Export.table.toDrive(
        collection=reduced.map(_add_fields),
        description=description,
        folder=folder,
        fileNamePrefix=description,
        fileFormat="CSV",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-tasks", action="store_true",
                        help="Start export tasks immediately.")
    parser.add_argument("--drive-folder", default=DRIVE_FOLDER)
    args = parser.parse_args()

    print("Initializing GEE ...")
    initialize_ee()

    regions = regions_fc()
    print(f"Regions: {len(COUNTRIES)} countries")

    print("Building masks (P2023A, epochs 2018 and 2020) ...")
    mask_2018 = build_mask(2018)
    mask_2020 = build_mask(2020)

    task_2018 = export_mask_area(mask_2018, regions,
                                  "adm2_mask_area_2018", args.drive_folder)
    task_2020 = export_mask_area(mask_2020, regions,
                                  "adm2_mask_area_2020", args.drive_folder)

    print("Tasks created:")
    print("  adm2_mask_area_2018  (P2023A / 2018 epoch)")
    print("  adm2_mask_area_2020  (P2023A / 2020 epoch)")
    print(f"  Drive folder: {args.drive_folder}")

    if args.start_tasks:
        task_2018.start()
        task_2020.start()
        print("Both tasks started -- monitor at code.earthengine.google.com/tasks")
    else:
        print("Re-run with --start-tasks to launch.")


if __name__ == "__main__":
    main()
