#!/usr/bin/env python3
"""Interactive Rivet motor ID, direction, and standing-zero commissioning."""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bdx_api.config import load_policy_config
from bdx_api.deployment import load_deployment_config, save_deployment_config


def policy_config(model_arg: str | None, config_arg: str | None = None):
    _, deployment = load_deployment_config(config_arg)
    model_path = Path(model_arg).expanduser() if model_arg else Path(deployment["policy_model"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path.resolve(), load_policy_config(model_path.resolve())


def selected_names(cfg, requested: str | None = None, deployment: dict | None = None) -> list[str]:
    if deployment is None:
        _, deployment = load_deployment_config()
    policy_names = [name.strip() for name in cfg.joint_names]
    if requested:
        if requested not in deployment["joints"]:
            raise ValueError(f"{requested!r} is not present in the Rivet deployment map")
        return [requested]
    return policy_names + [name for name in deployment["joints"] if name not in policy_names]


def require_entry(config: dict, name: str) -> dict:
    try:
        return config["joints"][name]
    except KeyError as exc:
        raise ValueError(f"No wiring entry exists for policy joint {name!r}") from exc


def ensure_no_id_conflict(config: dict, joint: str, motor_id: int) -> None:
    for other_name, entry in config["joints"].items():
        if other_name != joint and entry["motor_id"] == motor_id:
            raise ValueError(f"Motor ID {motor_id} is already assigned to {other_name}")


def open_bus(channel: str):
    import can

    return can.interface.Bus(interface="socketcan", channel=channel)


def make_client(bus, timeout: float = 1.0):
    from robstride.client import Client

    return Client(bus, recv_timeout=timeout)


def cmd_status(args) -> None:
    model_path, cfg = policy_config(args.model, args.config)
    config_path, deployment = load_deployment_config(args.config)
    print(f"Policy: {model_path}")
    print(f"Hardware map: {config_path}")
    print(f"CAN interfaces: {', '.join(deployment['can_channels'])}")
    print("\nPolicy joint order and commissioning state:")
    policy_ready = True
    all_ids_ready = True
    all_directions_ready = True
    shown_names = [name.strip() for name in cfg.joint_names]
    shown_names += [name for name in deployment["joints"] if name not in shown_names]
    for index, name in enumerate(shown_names):
        entry = require_entry(deployment, name.strip())
        id_ok = bool(entry.get("id_verified", False))
        direction_ok = bool(entry.get("direction_verified", False))
        all_ids_ready &= id_ok
        all_directions_ready &= direction_ok
        if name.strip() in {joint.strip() for joint in cfg.joint_names}:
            policy_ready &= id_ok and direction_ok
        state = "ready" if id_ok and direction_ok else "pending"
        print(
            f"  {index:02d} {name.strip():20} {entry['can_interface']} "
            f"ID {entry['motor_id']:3} {entry['motor_type']:>2} "
            f"sign {entry.get('direction', 1):+d} | ID {'ok' if id_ok else 'pending'}; "
            f"direction {'ok' if direction_ok else 'pending'} | {state}"
        )
    print(f"\nPolicy deployment: {'READY' if policy_ready else 'BLOCKED until policy joints are verified'}")
    print(
        "All-motor zero stage: "
        + ("READY" if all_ids_ready else "BLOCKED until every mapped motor ID is verified")
    )
    print(f"All mapped directions verified: {'yes' if all_directions_ready else 'no'}")
    last_zero_save = deployment.get("last_zero_eeprom_status")
    if last_zero_save:
        print(f"Last zero EEPROM save: {last_zero_save}")


def cmd_verify_ids(args) -> None:
    _, cfg = policy_config(args.model, args.config)
    config_path, deployment = load_deployment_config(args.config)
    names = selected_names(cfg, args.joint, deployment)
    buses = {}
    clients = {}
    try:
        for name in names:
            entry = require_entry(deployment, name)
            channel = entry["can_interface"]
            if channel not in buses:
                buses[channel] = open_bus(channel)
                clients[channel] = make_client(buses[channel])

            default_id = entry["motor_id"]
            raw_id = input(f"{name} on {channel}; current motor ID [{default_id}] (or q to stop): ").strip()
            if raw_id.lower() == "q":
                break
            current_id = int(raw_id) if raw_id else default_id
            if not 1 <= current_id <= 127:
                raise ValueError("Motor ID must be between 1 and 127")

            entry["id_verified"] = False
            save_deployment_config(config_path, deployment)
            position = clients[channel].read_param(current_id, "mechpos")
            print(f"  Motor {current_id} answered; encoder position {float(position):+.4f} rad")
            expected_id = entry["motor_id"]
            if current_id != expected_id:
                ensure_no_id_conflict(deployment, name, expected_id)
                try:
                    clients[channel].read_param(expected_id, "mechpos")
                except Exception:
                    pass
                else:
                    raise RuntimeError(
                        f"Configured target ID {expected_id} also answers on {channel}; "
                        "resolve the duplicate ID before changing this motor"
                    )
                answer = input(
                    f"  Reassign this motor from ID {current_id} to policy slot ID {expected_id}? "
                    "Type CHANGE to send the RobStride ID update: "
                ).strip()
                if answer != "CHANGE":
                    print("  Left unchanged; this joint remains unverified.")
                    entry["id_verified"] = False
                    save_deployment_config(config_path, deployment)
                    continue
                clients[channel].update_id(current_id, expected_id)
                position = clients[channel].read_param(expected_id, "mechpos")
                print(f"  New ID {expected_id} answered; encoder position {float(position):+.4f} rad")

            answer = input(
                f"  Confirm the responding motor is physically installed at {name}? Type YES: "
            ).strip()
            entry["id_verified"] = answer == "YES"
            save_deployment_config(config_path, deployment)
            print("  ID verified and saved." if entry["id_verified"] else "  Not confirmed; left pending.")
    finally:
        for bus in buses.values():
            bus.shutdown()


def cmd_set_id(args) -> None:
    if not 1 <= args.current_id <= 127 or not 1 <= args.new_id <= 127:
        raise ValueError("Motor IDs must be between 1 and 127")
    config_path, deployment = load_deployment_config(args.config)
    entry = require_entry(deployment, args.joint)
    ensure_no_id_conflict(deployment, args.joint, args.new_id)
    channel = entry["can_interface"]
    bus = open_bus(channel)
    client = make_client(bus)
    try:
        position = client.read_param(args.current_id, "mechpos")
        print(
            f"{args.joint} is mapped to {channel}, ID {entry['motor_id']}; "
            f"ID {args.current_id} responded at {float(position):+.4f} rad."
        )
        if input(f"Confirm this physical motor is installed at {args.joint}? Type YES: ").strip() != "YES":
            print("ID update cancelled.")
            return
        confirmation = input(
            f"Change the motor ID now from {args.current_id} to {args.new_id}? Type CHANGE: "
        ).strip()
        if confirmation != "CHANGE":
            print("ID update cancelled.")
            return
        if args.current_id != args.new_id:
            try:
                client.read_param(args.new_id, "mechpos")
            except Exception:
                pass
            else:
                raise RuntimeError(
                    f"Target ID {args.new_id} already answers on {channel}; resolve the duplicate before reassignment"
                )
            entry["id_verified"] = False
            save_deployment_config(config_path, deployment)
            client.update_id(args.current_id, args.new_id)
        position = client.read_param(args.new_id, "mechpos")
        entry["motor_id"] = args.new_id
        entry["id_verified"] = True
        save_deployment_config(config_path, deployment)
        print(f"ID {args.new_id} answered at {float(position):+.4f} rad; map updated and saved.")
    finally:
        bus.shutdown()


def _send_small_move(bus, motor_id: int, motor_type: str, start: float, delta: float) -> None:
    from bdx_api.hardware import MOTOR_TYPE_PARAMS, _send_mit_command

    steps = 20
    for step in range(1, steps + 1):
        target = start + delta * step / steps
        _send_mit_command(
            bus, motor_id, target, 0.0, 5.0, 0.5, 0.0,
            MOTOR_TYPE_PARAMS[motor_type],
        )
        time.sleep(0.05)


def cmd_check_directions(args) -> None:
    _, cfg = policy_config(args.model, args.config)
    config_path, deployment = load_deployment_config(args.config)
    names = selected_names(cfg, args.joint, deployment)
    print(
        "Direction checks move one motor by at most 1 degree at low gains. "
        "Physically support the robot so it cannot fall or pinch. Keep the emergency stop ready."
    )
    if input("Type MOTOR TEST when the robot is secured and the area is clear: ").strip() != "MOTOR TEST":
        print("Direction checks cancelled.")
        return

    buses = {}
    try:
        for name in names:
            entry = require_entry(deployment, name)
            if not entry.get("id_verified", False):
                raise RuntimeError(f"Verify the ID for {name} before checking its direction")
            channel = entry["can_interface"]
            if channel not in buses:
                buses[channel] = open_bus(channel)
            bus = buses[channel]
            client = make_client(bus)
            motor_id = entry["motor_id"]
            choice = input(
                f"Ready to move {name} (CAN {channel}, motor ID {motor_id}) by 1 degree. "
                "Press ENTER to test this joint or type q to stop: "
            ).strip().lower()
            if choice == "q":
                break
            start = float(client.read_param(motor_id, "mechpos"))
            enable_attempted = True
            current_target = start
            try:
                client.enable(motor_id)
                entry["direction_verified"] = False
                save_deployment_config(config_path, deployment)
                direction = int(entry.get("direction", 1))
                current_target = start + math.radians(direction)
                _send_small_move(bus, motor_id, entry["motor_type"], start, current_target - start)
                measured = float(client.read_param(motor_id, "mechpos"))
                print(f"  Encoder change in policy coordinates: {math.degrees((measured - start) * direction):+.2f} degrees")
                answer = input(
                    f"Did exactly {name} move about 1 degree in the policy-positive direction? [y/n/w/m]: "
                ).strip().lower()
                if answer == "m":
                    raise RuntimeError(f"No clear motion was observed for {name}; direction remains unverified")
                if answer == "w":
                    entry["id_verified"] = False
                    entry["direction_verified"] = False
                    save_deployment_config(config_path, deployment)
                    raise RuntimeError(
                        f"A different motor moved while testing {name}; correct the ID mapping and rerun commissioning"
                    )
                if answer == "n":
                    _send_small_move(bus, motor_id, entry["motor_type"], current_target, start - current_target)
                    current_target = start
                    direction *= -1
                    entry["direction"] = direction
                    print(f"  Inverted {name} sign to {direction:+d}; checking the corrected direction once.")
                    current_target = start + math.radians(direction)
                    _send_small_move(bus, motor_id, entry["motor_type"], start, current_target - start)
                    measured = float(client.read_param(motor_id, "mechpos"))
                    print(f"  Encoder change in policy coordinates: {math.degrees((measured - start) * direction):+.2f} degrees")
                    answer = input(
                        f"Did exactly {name} now move in the policy-positive direction? [y/w/m]: "
                    ).strip().lower()
                    if answer == "m":
                        raise RuntimeError(f"No clear motion was observed for {name}; direction remains unverified")
                if answer == "w":
                    entry["id_verified"] = False
                    entry["direction_verified"] = False
                    save_deployment_config(config_path, deployment)
                    raise RuntimeError(f"Wrong motor moved for {name}; correct its ID mapping")
                entry["direction_verified"] = answer == "y"
                if not entry["direction_verified"]:
                    raise RuntimeError(f"Direction for {name} is still unverified")
                save_deployment_config(config_path, deployment)
                print(f"  {name}: direction {direction:+d} verified and saved.")
            finally:
                if current_target != start:
                    try:
                        _send_small_move(bus, motor_id, entry["motor_type"], current_target, start - current_target)
                    except Exception as exc:
                        print(f"  Could not return motor {motor_id} to its starting position: {exc}")
                if enable_attempted:
                    try:
                        client.disable(motor_id)
                    except Exception as exc:
                        print(f"  Could not confirm disable for motor {motor_id}: {exc}")
    finally:
        for bus in buses.values():
            bus.shutdown()


def cmd_stand_zero(args) -> None:
    config_path, deployment = load_deployment_config(args.config)
    missing = []
    for name, entry in deployment["joints"].items():
        if not entry.get("id_verified", False):
            missing.append(name)
    if missing:
        raise RuntimeError("Verify motor IDs first: " + ", ".join(missing))

    try:
        ready = input(
            "Manually place Rivet in its desired standing pose and support it securely. "
            "This resets encoder zero on every configured motor without moving the joints or saving to EEPROM. "
            "Type STAND ZERO to proceed: "
        )
    except (KeyboardInterrupt, EOFError):
        print("Standing/zeroing cancelled.")
        return
    if ready.strip() != "STAND ZERO":
        print("Standing/zeroing cancelled.")
        return

    buses = {}
    try:
        per_channel = {channel: [] for channel in deployment["can_channels"]}
        for entry in deployment["joints"].values():
            per_channel[entry["can_interface"]].append(entry["motor_id"])
        for channel, motor_ids in per_channel.items():
            if motor_ids:
                buses[channel] = open_bus(channel)
        errors: list[str] = []

        def zero_channel(channel: str, motor_ids: list[int]) -> None:
            try:
                responses = make_client(buses[channel]).zero_pos_many(motor_ids)
                for motor_id, response in responses.items():
                    if response.errors:
                        errors.append(f"{channel} motor {motor_id}: {response.errors}")
            except Exception as exc:
                errors.append(f"{channel}: {exc}")

        workers = [
            threading.Thread(target=zero_channel, args=(channel, ids), daemon=True)
            for channel, ids in per_channel.items() if ids
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if errors:
            raise RuntimeError("Zero batch did not complete cleanly: " + "; ".join(errors))

        print("All configured motors acknowledged zero_pos.")
        print(
            "Saving writes each motor's current configuration, including its new zero offset, "
            "to nonvolatile EEPROM so it persists across power cycles. The saved zero may "
            "require a motor power cycle to take effect as the persistent calibration. "
            "Skipping leaves the new zero active only for this powered session; the previously "
            "saved zero returns after a reset or power cycle."
        )
        try:
            save_choice = input("Save the configuration to EEPROM on all motors? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            save_choice = "n"
            print("\nEEPROM save skipped.")

        save_failures: list[str] = []
        if save_choice == "y":
            def save_channel(channel: str, motor_ids: list[int]) -> None:
                try:
                    client = make_client(buses[channel])
                except Exception as exc:
                    save_failures.append(f"{channel}: could not create RobStride client: {exc}")
                    return
                for motor_id in motor_ids:
                    try:
                        client.save_configuration(motor_id)
                    except Exception as exc:
                        save_failures.append(f"{channel} motor {motor_id}: {exc}")

            save_workers = [
                threading.Thread(target=save_channel, args=(channel, ids), daemon=True)
                for channel, ids in per_channel.items() if ids
            ]
            for worker in save_workers:
                worker.start()
            for worker in save_workers:
                worker.join()

            deployment["last_zero_eeprom_status"] = "saved" if not save_failures else "partial"
            if save_failures:
                deployment["last_zero_eeprom_failures"] = save_failures
                print("EEPROM save was incomplete; check these motor acknowledgments:")
                for failure in save_failures:
                    print(f"  - {failure}")
                print(
                    "Motors that acknowledged the save retain the new zero across power cycles; "
                    "motors that did not acknowledge it may revert to their previous saved zero."
                )
            else:
                deployment.pop("last_zero_eeprom_failures", None)
                print(
                    "All configured motors acknowledged the EEPROM save. Power-cycle the motors "
                    "for the saved zero calibration to take effect."
                )
        else:
            deployment["last_zero_eeprom_status"] = "skipped"
            deployment.pop("last_zero_eeprom_failures", None)
            print("EEPROM save skipped; the zero is temporary until the motors are reset or power-cycled.")

        deployment["last_zero_policy"] = deployment["policy_model"]
        deployment["last_zero_joint_count"] = sum(len(ids) for ids in per_channel.values())
        save_deployment_config(config_path, deployment)
    finally:
        for bus in buses.values():
            bus.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rivet hardware commissioning tools")
    parser.add_argument("--config", default=None, help="Hardware map YAML; defaults to config/rivet_deployment.yaml")
    parser.add_argument("--model", default=None, help="ONNX model; defaults to policy_model in the hardware map")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Show policy joint order and commissioning state").set_defaults(func=cmd_status)
    ids = commands.add_parser("verify-ids", help="Check every mapped joint and optionally change CAN IDs")
    ids.add_argument("--joint", help="Verify only this configured joint")
    ids.set_defaults(func=cmd_verify_ids)
    set_id = commands.add_parser("set-id", help="Change one motor ID immediately and update the hardware map")
    set_id.add_argument("joint")
    set_id.add_argument("--current-id", type=int, required=True)
    set_id.add_argument("--new-id", type=int, required=True)
    set_id.set_defaults(func=cmd_set_id)
    directions = commands.add_parser("check-directions", help="Move each policy joint by 1 degree and verify its sign")
    directions.add_argument("--joint", help="Check only this policy joint")
    directions.set_defaults(func=cmd_check_directions)
    zero = commands.add_parser(
        "stand-zero",
        help="Zero all configured motors while the robot is manually held in its standing pose",
    )
    zero.set_defaults(func=cmd_stand_zero)

    args = parser.parse_args()
    try:
        args.func(args)
    except (KeyboardInterrupt, EOFError):
        print("\nCommissioning interrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"Commissioning stopped: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
