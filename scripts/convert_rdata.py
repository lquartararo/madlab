"""
One-time script to convert all .RData files from the mRchmadness R package
into .parquet (dataframes) or .json (bracket vectors) files.

Usage:
    uv run scripts/convert_rdata.py --rdata-dir ../mRchmadness/data --out-dir src/marchmadness/data
"""

import argparse
import json
from pathlib import Path

import pyreadr
import polars as pl


BRACKET_STEMS = {s for s in [
    "bracket.men.2017", "bracket.men.2018", "bracket.men.2019",
    "bracket.men.2021", "bracket.men.2022", "bracket.men.2023",
    "bracket.men.2024", "bracket.men.2025",
    "bracket.women.2017", "bracket.women.2018", "bracket.women.2021",
    "bracket.women.2022", "bracket.women.2023", "bracket.women.2024",
    "bracket.women.2025",
]}


def convert_rdata(rdata_path: Path, out_dir: Path) -> None:
    stem = rdata_path.stem
    result = pyreadr.read_r(str(rdata_path))

    for var_name, df in result.items():
        if df is None:
            print(f"  SKIP {rdata_path.name}: null object")
            continue

        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            print(f"  SKIP {rdata_path.name}: unexpected type {type(df)}")
            continue

        if stem in BRACKET_STEMS:
            # Bracket: single-column DataFrame -> JSON list of team-id strings
            values = df.iloc[:, 0].astype(str).tolist()
            out_path = out_dir / f"{stem}.json"
            with open(out_path, "w") as f:
                json.dump(values, f)
            print(f"  {rdata_path.name} -> {out_path.name}  (bracket, len={len(values)})")
        else:
            # Dataframes: normalize column names (dots to underscores)
            df.columns = [c.replace(".", "_") for c in df.columns]
            pl_df = pl.from_pandas(df)
            out_path = out_dir / f"{stem}.parquet"
            pl_df.write_parquet(str(out_path))
            print(f"  {rdata_path.name} -> {out_path.name}  {pl_df.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdata-dir", default="../mRchmadness/data")
    parser.add_argument("--out-dir", default="src/marchmadness/data")
    args = parser.parse_args()

    rdata_dir = Path(args.rdata_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(rdata_dir.glob("*.RData")):
        convert_rdata(f, out_dir)

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
