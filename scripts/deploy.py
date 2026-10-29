import os
import csv
import argparse
import sys
import numpy as np
from pathlib import Path

# Make direct execution work from the checkout, even when the package has not
# been installed into the active Python environment (or the command is run
# from a different working directory).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- NEW: Import the config loader to print metadata ---
from bdx_api.config import load_policy_config, load_isaaclab_config, load_imitation_config
from bdx_api.deployment import PROJECT_ROOT, load_deployment_config, resolve_deployment_config
from bdx_api.can_bus import add_can_transport_arguments, can_transport_kwargs

def main():
    parser = argparse.ArgumentParser(description="BDX-R Deploy — sim or real")
    parser.add_argument("--backend", type=str, default="sim", choices=["sim", "real"],
                        help="Backend: 'sim' for MuJoCo, 'real' for hardware")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to ONNX policy model")
    parser.add_argument("--deployment-config", type=str, default=None,
                        help="Path to Rivet hardware map (defaults to config/rivet_deployment.yaml)")
    add_can_transport_arguments(parser)
    parser.add_argument("--authority-levels", type=str, default="0,0.1,0.25,0.5,0.75,1.0",
                        help="Operator-gated real-policy authority stages from 0 to 1")
    parser.add_argument("--isaaclab", action="store_true",
                        help="Use IsaacLab-trained policy (sets action scale, obs order, gains from env.yaml; defaults model to models/isaaclab.onnx)")

    # --- XML PATHS ---
    parser.add_argument("--xml", type=str, default="xml/bdxr.xml",
                        help="Path to MuJoCo XML (sim only, full robot)")
    parser.add_argument("--xml_headless", type=str, default="xml/BDX-R-Legs.xml",
                        help="Path to headless MuJoCo XML (used if --legs is set)")

    parser.add_argument("--sim_dt", type=float, default=0.02,
                        help="Simulation timestep (sim only)")
    parser.add_argument("--ctrl_hz", type=float, default=400.0,
                        help="Motor control rate in Hz (real hardware only, default 400 Hz). Policy still runs at 50 Hz.")
    parser.add_argument("--decimation", type=int, default=None,
                        help="Control decimation (policy runs every N steps). Defaults to ctrl_hz/50 for real, round(0.02/sim_dt) for sim.")
    parser.add_argument("--imu_backend", choices=["orbbec", "xsens"], default="orbbec",
                        help="IMU source for real hardware (default: Orbbec camera IMU)")
    parser.add_argument("--imu_port", type=str, default="/dev/ttyUSB0",
                        help="Serial port for the Xsens MTi-3 when --imu_backend xsens is selected")

    # --- LEGS ONLY FLAG ---
    parser.add_argument("--legs", action="store_true",
                        help="Hold the head joints fixed and only deploy the legs policy")

    # --- NO IMU FLAG ---
    parser.add_argument("--no_imu", action="store_true",
                        help="Disable IMU and use constant 0 for gyro and [0,0,-1] for gravity (debugging)")

    # --- IMITATION FLAG ---
    parser.add_argument("--imitation", action="store_true",
                        help="Deploy the imitation learning legs policy (uses imitate/imitate.onnx)")
    parser.add_argument("--pkl", type=str, default="imitate/polynomial_coefficients_legs.pkl",
                        help="Path to polynomial_coefficients_legs.pkl for imitation reference motion")

    args = parser.parse_args()
    deployment_config_path = resolve_deployment_config(args.deployment_config)

    if args.decimation is None:
        if args.backend == "real":
            args.decimation = max(1, round(args.ctrl_hz / 50))
        else:
            args.decimation = max(1, round(0.02 / args.sim_dt))

    # Resolve model path
    if args.model is not None:
        model_path = Path(args.model)
    elif args.imitation:
        model_path = Path("imitate/imitate.onnx")
    elif args.isaaclab:
        model_path = Path("models/isaaclab.onnx")
    else:
        _, deployment = load_deployment_config(deployment_config_path)
        model_path = Path(deployment["policy_model"])
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model_path = model_path.resolve()

    try:
        authority_levels = [float(value.strip()) for value in args.authority_levels.split(",") if value.strip()]
    except ValueError as exc:
        parser.error(f"Invalid --authority-levels: {exc}")
    if (not authority_levels or authority_levels[0] != 0.0 or authority_levels[-1] != 1.0
            or any(a >= b for a, b in zip(authority_levels, authority_levels[1:]))):
        parser.error("--authority-levels must increase from 0 to 1, for example 0,0.1,0.25,0.5,0.75,1")

    if args.imitation:
        cfg_override = load_imitation_config()
        args.legs = True  # imitation is always legs-only
        args.sim_dt = 0.0025
        args.decimation = 8
    elif args.isaaclab:
        cfg_override = load_isaaclab_config()
    else:
        cfg_override = None

    if args.isaaclab:
        if args.sim_dt == 0.005:
            args.sim_dt = 0.0025
        if args.decimation == 4:
            args.decimation = 8

    # ==========================================
    # PRINT POLICY CONFIG
    # ==========================================
    try:
        cfg = cfg_override if cfg_override is not None else load_policy_config(model_path)
        print("\n" + "=" * 60)
        print("  ONNX POLICY METADATA LOADED")
        print("=" * 60)
        print(f"  Mode       : {'IsaacLab' if args.isaaclab else 'MJLab/ONNX'}")
        print(f"  Model Path : {model_path}")
        print(f"  Obs dim    : {cfg.obs_dim}")
        print(f"  Action dim : {cfg.action_dim}")
        print(f"  Joints ({len(cfg.joint_names)}): {cfg.joint_names}")
        print(f"  Stiffness  : {cfg.joint_stiffness}")
        print(f"  Damping    : {cfg.joint_damping}")
        print(f"  Defaults   : {[round(p, 3) for p in cfg.default_joint_pos]}")
        print(f"  Action Scl : {cfg.action_scale}")
        print(f"  Obs names  : {cfg.observation_names}")
        print(f"  Cmd names  : {cfg.command_names}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n[WARNING] Could not print ONNX metadata: {e}\n")
        if args.backend == "real":
            raise

    # 1. Decide which XML to use based on the flag
    if args.legs:
        active_xml = args.xml_headless
        print(f"Deploying in LEGS ONLY mode. Holding head stiff.")
        print(f"Using XML: {active_xml}")
    else:
        active_xml = args.xml
        print(f"Deploying FULL 14-joint model.")
        print(f"Using XML: {active_xml}")

    if args.no_imu:
        print(f"[DEBUG] --no_imu flag is ACTIVE. Mocking IMU sensor data.")

    # 2. Initialize the correct backend
    if args.backend == "sim":
        from bdx_api.sim import MujocoBackend
        backend = MujocoBackend(
            xml_path=active_xml,
            model_path=model_path,
            sim_dt=args.sim_dt,
            legs_only=args.legs,
            cfg_override=cfg_override,
        )
    elif args.backend == "real":
        from bdx_api.hardware import HardwareBackend, HardwareInitializationCancelled
        try:
            backend = HardwareBackend(
                model_path=model_path,
                loop_dt=1.0 / args.ctrl_hz,
                imu_port=args.imu_port,
                imu_backend=args.imu_backend,
                legs_only=args.legs,
                use_imu=not args.no_imu,
                cfg_override=cfg_override,
                deployment_config_path=deployment_config_path,
                confirm_initial_positions=True,
                **can_transport_kwargs(args),
            )
        except HardwareInitializationCancelled:
            print("Deployment cancelled before any motor was enabled.")
            return
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # 3. Initialize the Runner
    if args.imitation:
        from bdx_api.interface import ImitationPolicyRunner
        runner = ImitationPolicyRunner(
            backend=backend,
            model_path=model_path,
            decimation=args.decimation,
            cfg_override=cfg_override,
            pkl_path=args.pkl,
            authority_levels=authority_levels if args.backend == "real" else None,
        )
    else:
        from bdx_api.interface import PolicyRunner
        runner = PolicyRunner(
            backend=backend,
            model_path=model_path,
            decimation=args.decimation,
            cfg_override=cfg_override,
            authority_levels=authority_levels if args.backend == "real" else None,
        )

    # 4. Run the control loop
    runner.run()

    # ==========================================
    # CSV EXPORT (Flight Recorder)
    # ==========================================
    entries = runner.logger.entries
    if len(entries) > 1:
        os.makedirs("logs", exist_ok=True)
        csv_path = "logs/latest_run.csv"
        print(f"\nSaving {len(entries)} frames to {csv_path}...")

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            num_joints = len(entries[0].actual_pos)
            header = ["tick"]
            header += [f"cmd_{i}" for i in range(3)]
            header += [f"ang_vel_{i}" for i in range(3)]
            header += [f"proj_grav_{i}" for i in range(3)]
            header += [f"pos_{i}" for i in range(num_joints)]
            header += [f"vel_{i}" for i in range(num_joints)]
            header += [f"action_{i}" for i in range(num_joints)]
            writer.writerow(header)

            for i, entry in enumerate(entries):
                row = [i]
                row.extend(entry.cmd)
                row.extend(entry.ang_vel)
                row.extend(entry.projected_gravity)
                row.extend(entry.actual_pos)
                row.extend(entry.actual_vel)
                row.extend(entry.actions)
                writer.writerow(row)
        print("CSV saved successfully.")

if __name__ == "__main__":
    main()
