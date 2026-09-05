"""Regenerate the simulated telemetry dataset.

    python scripts/generate_data.py --rows 840 --out data/simulated_energy_data_with_ai.csv

The sample data ships with the repo, so this is only needed to produce a
larger fleet, a longer window, or a fresh seed for demos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEVICE_TYPES = ["HVAC", "Pump", "Lighting", "Compressor", "Server Rack"]

#: Baseline draw and swing per device type, in kWh.
LOAD_PROFILE = {
    "HVAC": (62.0, 18.0),
    "Pump": (38.0, 12.0),
    "Lighting": (25.0, 9.0),
    "Compressor": (55.0, 16.0),
    "Server Rack": (70.0, 8.0),
}


def generate(hours: int, seed: int, start: str) -> pd.DataFrame:
    """Build an hourly fleet trace with a realistic daily load curve."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=hours, freq="h")

    rows = []
    for index, device_type in enumerate(DEVICE_TYPES, start=1):
        base, swing = LOAD_PROFILE[device_type]
        for timestamp in timestamps:
            # Load peaks mid-afternoon and troughs overnight.
            daily = np.sin((timestamp.hour - 4) / 24 * 2 * np.pi)
            usage = base + swing * daily + rng.normal(0, swing / 3)
            status = "ON" if rng.random() > 0.08 else "OFF"
            voltage = rng.normal(230, 4)

            # Inject the two fault modes the detectors are built to catch.
            if rng.random() < 0.012:
                usage, status = base + swing * 2.5, "OFF"  # phantom load
            if rng.random() < 0.008:
                voltage = rng.choice([rng.normal(212, 3), rng.normal(248, 3)])

            rows.append(
                {
                    "timestamp": timestamp,
                    "device_id": f"DEV-{index}",
                    "device_type": device_type,
                    "usage_kwh": round(max(usage, 0.5), 2),
                    "voltage": round(voltage, 2),
                    "status": status,
                    "hour": timestamp.hour,
                }
            )

    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    # Ship the rule-based baseline alongside, as the original export did.
    frame["anomaly_flag"] = ((frame["status"] == "OFF") & (frame["usage_kwh"] > 90)).astype(int)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, default=168, help="Hours per device (default: 168).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2025-03-20 00:00:00")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/simulated_energy_data_generated.csv"),
    )
    args = parser.parse_args()

    frame = generate(args.hours, args.seed, args.start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame):,} rows to {args.out}")


if __name__ == "__main__":
    main()
