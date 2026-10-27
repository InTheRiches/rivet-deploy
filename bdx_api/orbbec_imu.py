"""Orbbec SDK v2 IMU input for the Rivet hardware backend."""

from __future__ import annotations

import math
import threading
import time

import numpy as np


class OrbbecIMU:
    """Read Gemini IMU frames and estimate policy-frame projected gravity.

    The SDK supplies angular velocity in rad/s and acceleration in m/s^2. This
    class estimates gravity direction with a complementary filter because the
    camera IMU reports accel and gyro frames rather than a fused orientation.
    """

    def __init__(
        self,
        sensor_to_base_rotation=None,
        *,
        stale_after_s: float = 0.25,
        gravity_time_constant_s: float = 0.5,
        gyro_bias_samples: int = 50,
        startup_timeout_s: float = 10.0,
    ):
        rotation = np.eye(3, dtype=np.float64) if sensor_to_base_rotation is None else np.asarray(
            sensor_to_base_rotation, dtype=np.float64
        )
        if rotation.shape != (3, 3):
            raise ValueError("sensor_to_base_rotation must be a 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-3
        ):
            raise ValueError("sensor_to_base_rotation must be a right-handed orthonormal rotation")
        if stale_after_s <= 0 or gravity_time_constant_s <= 0 or gyro_bias_samples < 1:
            raise ValueError("IMU timing and calibration settings must be positive")

        self.sensor_to_base_rotation = rotation
        self.stale_after_s = float(stale_after_s)
        self.gravity_time_constant_s = float(gravity_time_constant_s)
        self.gyro_bias_samples = int(gyro_bias_samples)
        self.startup_timeout_s = float(startup_timeout_s)

        self.latest_data = {
            "gyro": np.zeros(3, dtype=np.float32),
            "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "accel": np.zeros(3, dtype=np.float32),
        }
        self.lock = threading.Lock()
        self._initialized = threading.Event()
        self._running = False
        self._pipeline = None
        self._thread = None
        self._last_sample_time = 0.0
        self._gyro_bias = np.zeros(3, dtype=np.float64)
        self._gravity = None
        self._last_error = None

    def start(self) -> bool:
        try:
            from pyorbbecsdk import (
                Config,
                OBFrameAggregateOutputMode,
                OBSensorType,
                Pipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Orbbec SDK Python bindings are missing. Install the deployment's orbbec extra "
                "(pyorbbecsdk2) and Orbbec's Linux device rules."
            ) from exc

        try:
            pipeline = Pipeline()
            self._pipeline = pipeline
            device = pipeline.get_device()
            if device is None:
                raise RuntimeError("No Orbbec camera was found")
            device.get_sensor(OBSensorType.ACCEL_SENSOR)
            device.get_sensor(OBSensorType.GYRO_SENSOR)

            config = Config()
            config.enable_accel_stream()
            config.enable_gyro_stream()
            config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            pipeline.start(config)
            self._running = True
            print(
                f"[Orbbec IMU] Keep Rivet still while collecting {self.gyro_bias_samples} "
                "startup gyro samples."
            )
            self._thread = threading.Thread(target=self._read_loop, name="orbbec-imu", daemon=True)
            self._thread.start()
            if not self._initialized.wait(timeout=self.startup_timeout_s):
                detail = f" Latest SDK error: {self._last_error}." if self._last_error else ""
                raise RuntimeError(
                    f"Timed out waiting for Orbbec accel/gyro data and {self.gyro_bias_samples} "
                    f"stationary gyro samples.{detail}"
                )
            print(
                f"[Orbbec IMU] Streaming accel + gyro; calibrated gyro bias from "
                f"{self.gyro_bias_samples} startup samples."
            )
            return True
        except Exception:
            self.stop()
            raise

    @staticmethod
    def _frame_vector(frame) -> np.ndarray:
        value = frame.get_value()
        return np.array([value.x, value.y, value.z], dtype=np.float64)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm < 1e-6:
            raise ValueError("Orbbec IMU returned a zero or invalid vector")
        return vector / norm

    def _read_loop(self) -> None:
        gyro_samples: list[np.ndarray] = []
        last_update = None
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                accel_frame = frames.get_accel_frame()
                gyro_frame = frames.get_gyro_frame()
                if accel_frame is None or gyro_frame is None:
                    continue

                accel_sensor = self._frame_vector(accel_frame)
                gyro_sensor = self._frame_vector(gyro_frame)
                if not np.all(np.isfinite(accel_sensor)) or not np.all(np.isfinite(gyro_sensor)):
                    continue

                if len(gyro_samples) < self.gyro_bias_samples:
                    gyro_samples.append(gyro_sensor)
                    if len(gyro_samples) == self.gyro_bias_samples:
                        self._gyro_bias = np.mean(gyro_samples, axis=0)
                        print("[Orbbec IMU] Startup gyro calibration complete.")
                    continue

                accel_base = self.sensor_to_base_rotation @ accel_sensor
                gyro_base = self.sensor_to_base_rotation @ (gyro_sensor - self._gyro_bias)
                now = time.monotonic()
                dt = 0.0 if last_update is None else min(max(now - last_update, 0.0), 0.05)
                last_update = now

                accel_norm = float(np.linalg.norm(accel_base))
                gravity_from_accel = None
                if math.isfinite(accel_norm) and accel_norm > 1e-6:
                    # A stationary accelerometer measures specific force, opposite gravity.
                    gravity_from_accel = -accel_base / accel_norm

                if self._gravity is None:
                    if gravity_from_accel is None:
                        continue
                    self._gravity = gravity_from_accel
                else:
                    # For a body-frame gravity vector, body angular velocity changes
                    # the vector by -omega x gravity.
                    predicted = self._gravity - np.cross(gyro_base, self._gravity) * dt
                    self._gravity = self._normalize(predicted)
                    if gravity_from_accel is not None and 0.8 * 9.80665 <= accel_norm <= 1.2 * 9.80665:
                        alpha = math.exp(-dt / self.gravity_time_constant_s)
                        self._gravity = self._normalize(
                            alpha * self._gravity + (1.0 - alpha) * gravity_from_accel
                        )

                with self.lock:
                    self.latest_data["gyro"] = gyro_base.astype(np.float32)
                    self.latest_data["projected_gravity"] = self._gravity.astype(np.float32)
                    self.latest_data["accel"] = accel_base.astype(np.float32)
                    self._last_sample_time = now
                    self._last_error = None
                self._initialized.set()
            except Exception as exc:
                self._last_error = str(exc)
                time.sleep(0.01)

    def get_latest_data(self) -> dict[str, np.ndarray]:
        with self.lock:
            return {name: value.copy() for name, value in self.latest_data.items()}

    def is_fresh(self) -> bool:
        with self.lock:
            return self._last_sample_time > 0 and time.monotonic() - self._last_sample_time <= self.stale_after_s

    def stop(self) -> None:
        self._running = False
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
