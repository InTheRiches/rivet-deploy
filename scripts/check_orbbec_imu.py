#!/usr/bin/env python3
"""Stream Orbbec IMU values and policy-frame projected gravity for alignment checks."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bdx_api.deployment import load_deployment_config
from bdx_api.orbbec_imu import OrbbecIMU


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Rivet's Orbbec camera IMU")
    parser.add_argument("--config", default=None, help="Rivet deployment YAML")
    args = parser.parse_args()

    config_path, deployment = load_deployment_config(args.config)
    imu_config = deployment.get("imu", {})
    if not isinstance(imu_config, dict):
        raise ValueError("The deployment map's imu section must be a YAML mapping")
    imu = OrbbecIMU(
        sensor_to_base_rotation=imu_config.get("sensor_to_base_rotation"),
        stale_after_s=float(imu_config.get("stale_after_s", 0.25)),
        gravity_time_constant_s=float(imu_config.get("gravity_time_constant_s", 0.5)),
        gyro_bias_samples=int(imu_config.get("gyro_bias_samples", 50)),
    )

    print(f"Deployment map: {config_path}")
    print("Keep Rivet still for startup calibration, then gently tilt it to check the gravity axes.")
    if not imu.start():
        raise RuntimeError("Orbbec IMU initialization failed")
    try:
        while True:
            data = imu.get_latest_data()
            print(
                f"gyro rad/s {data['gyro']} | "
                f"projected gravity {data['projected_gravity']} | "
                f"fresh={imu.is_fresh()}",
                flush=True,
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nIMU check stopped.")
    finally:
        imu.stop()


if __name__ == "__main__":
    main()
