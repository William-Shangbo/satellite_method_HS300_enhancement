from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from index_platform.panel.cleaning import HS300RawCleaner


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Display raw vs cleaned HS300 sample data and write artifacts.",
    )
    parser.add_argument(
        "--zip-path",
        default="/Users/shangbo/study/P2025/ita/index_enforcement/src/沪深300成分股自2010以来的数据.zip",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_root / "artifacts" / "raw_clean_demo"),
    )
    parser.add_argument("--sample-rows", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaner = HS300RawCleaner(args.zip_path)
    raw_frame, cleaned_frame, summary = cleaner.clean_panel()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_sample = raw_frame.head(args.sample_rows).copy()
    cleaned_sample = cleaned_frame.head(args.sample_rows).copy()
    raw_sample.to_csv(output_dir / "raw_sample.csv", index=False)
    cleaned_sample.to_csv(output_dir / "cleaned_sample.csv", index=False)
    (output_dir / "cleaning_summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("CLEANING_SUMMARY")
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    print("\nRAW_SAMPLE")
    print(raw_sample.to_string(index=False))
    print("\nCLEANED_SAMPLE")
    print(cleaned_sample.to_string(index=False))


if __name__ == "__main__":
    main()
