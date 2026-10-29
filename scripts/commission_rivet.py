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
from bdx_api.can_bus import add_can_transport_arguments, open_can_bus
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


def ask_which_motor_enabled(joints: dict[str, dict]) -> tuple[str | None, bool]:
    names = list(joints)
    print("  Which physical motor enabled?")
    for index, name in enumerate(names, start=1):
        print(f"    {index}. {name}")
    while True:
        answer = input("  Enter its number or joint name, u if unknown, or q to stop: ").strip()
        if answer.lower() == "q":
            return None, True
        if answer.lower() in {"u", "unknown"}:
            return None, False
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1], False
        if answer in joints:
            return answer, False
        print("  Choose a listed number or exact joint name, or enter u/q.")


def record_verified_motor(
    config_path: Path,
    deployment: dict,
    joint: str,
    channel: str,
    motor_id: int,
) -> None:
    """Map a responding motor ID to its observed physical joint without changing the motor ID."""
    entry = require_entry(deployment, joint)
    old_channel, old_id = entry["can_interface"], entry["motor_id"]
    previous_owner = next(
        (
            other_name
            for other_name, other_entry in deployment["joints"].items()
            if other_name != joint and other_entry["motor_id"] == motor_id
        ),
        None,
    )
    if previous_owner is not None:
        displaced = require_entry(deployment, previous_owner)
        displaced["can_interface"] = old_channel
        displaced["motor_id"] = old_id
        displaced["id_verified"] = False
        displaced["direction_verified"] = False
        print(
            f"  ID {motor_id} was provisionally assigned to {previous_owner}; "
            f"that entry moved to {old_channel} ID {old_id} and needs verification."
        )

    changed = old_channel != channel or old_id != motor_id
    entry["can_interface"] = channel
    entry["motor_id"] = motor_id
    entry["id_verified"] = True
    if changed:
        entry["direction_verified"] = False
    save_deployment_config(config_path, deployment)
    print(f"  Saved {joint} as {channel} ID {motor_id}; the motor's own ID was not changed.")


def open_bus(channel: str, args, deployment: dict):
    return open_can_bus(
        channel,
        deployment["can_channels"],
        can_interface=args.can_interface,
        slcan_channel=args.slcan_channel,
        slcan_bitrate=args.slcan_bitrate,
        slcan_tty_baudrate=args.slcan_tty_baudrate,
    )


def make_client(bus, timeout: float = 1.0):
    # Use the repository's client extensions for batched zeroing and EEPROM saves.
    # The separately installed lowercase `robstride` package does not expose them.
    from Robstride.client import Client

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
    responding: list[tuple[str, int]] = []
    try:
        print(
            "Scanning configured CAN buses by reading mechpos only. "
            "This discovery step does not enable or move any motor."
        )
        for channel in deployment["can_channels"]:
            buses[channel] = open_bus(channel, args, deployment)
            clients[channel] = make_client(buses[channel])
            scan_client = make_client(buses[channel], timeout=0.1)
            transport_channel = args.slcan_channel if args.can_interface == "slcan" else channel
            print(f"  Scanning {channel} ({transport_channel}) motor IDs 1–127...")
            channel_ids = []
            for motor_id in range(1, 128):
                try:
                    scan_client.read_param(motor_id, "mechpos")
                except Exception:
                    continue
                channel_ids.append(motor_id)
                responding.append((channel, motor_id))
            ids_text = ", ".join(str(motor_id) for motor_id in channel_ids) or "none"
            print(f"  {channel}: responding motor IDs: {ids_text}")

        if not responding:
            print("No motors responded to the read-only scan; the hardware map was not changed.")
            return

        ready = input(
            "The next step can energize/hold one motor at a time, but sends no motion target. "
            "Keep Rivet supported and the area clear. Type ENABLE CHECK to continue: "
        ).strip()
        if ready != "ENABLE CHECK":
            print("Motor identification cancelled; the hardware map was not changed.")
            return

        print(
            "\nNext, each responding ID will be enabled by itself so you can identify its physical joint. "
            "For each joint, its configured ID is tried first, followed by the other untested responders. "
            "Only an enable command is sent: no position, velocity, or torque target is issued. "
            "The motor is disabled as soon as you answer."
        )
        remaining_candidates = list(responding)
        verified_this_run: set[str] = set()
        stop = False

        for name in names:
            if name in verified_this_run:
                print(f"\n{name}: already identified during this session.")
                continue

            entry = require_entry(deployment, name)
            entry["id_verified"] = False
            entry["direction_verified"] = False
            save_deployment_config(config_path, deployment)
            print(f"\nLooking for physical joint: {name}")
            found_expected = False

            while remaining_candidates:
                preferred = (entry["can_interface"], entry["motor_id"])
                candidate_index = (
                    remaining_candidates.index(preferred)
                    if preferred in remaining_candidates
                    else 0
                )
                channel, motor_id = remaining_candidates.pop(candidate_index)
                ready = input(
                    f"  Press ENTER to enable {channel} ID {motor_id} while watching {name}; "
                    "type q to stop: "
                ).strip().lower()
                if ready == "q":
                    stop = True
                    break

                try:
                    response = clients[channel].enable(motor_id)
                except Exception as exc:
                    print(f"  ID {motor_id} did not acknowledge enable: {exc}")
                    try:
                        clients[channel].disable(motor_id)
                    except Exception as disable_exc:
                        print(
                            f"  Could not confirm disable for {channel} ID {motor_id}: "
                            f"{disable_exc}. Disable it manually before continuing."
                        )
                    continue

                try:
                    print(
                        f"  {channel} ID {motor_id} acknowledged enable "
                        f"(mode={response.mode.name}, errors={response.errors or 'none'})."
                    )
                    while True:
                        answer = input(
                            f"  Is the motor at {name} the one enabled? [y/n/q]: "
                        ).strip().lower()
                        if answer in {"y", "n", "q"}:
                            break
                        print("  Enter y, n, or q.")

                    if answer == "q":
                        stop = True
                    elif answer == "y":
                        record_verified_motor(config_path, deployment, name, channel, motor_id)
                        verified_this_run.add(name)
                        found_expected = True
                    else:
                        actual_joint, should_stop = ask_which_motor_enabled(deployment["joints"])
                        if should_stop:
                            stop = True
                        elif actual_joint is None:
                            print(f"  ID {motor_id} left unidentified; it will not be assigned.")
                        else:
                            record_verified_motor(
                                config_path, deployment, actual_joint, channel, motor_id
                            )
                            verified_this_run.add(actual_joint)
                            found_expected = actual_joint == name
                finally:
                    try:
                        clients[channel].disable(motor_id)
                        print(f"  Disabled {channel} ID {motor_id}.")
                    except Exception as exc:
                        print(
                            f"  Could not confirm disable for {channel} ID {motor_id}: {exc}. "
                            "Disable it manually before continuing."
                        )

                if stop or found_expected:
                    break

            if stop:
                break
            if not found_expected and name not in verified_this_run:
                print(f"  {name} remains unverified; no responding ID was confirmed for it.")

        print("\nID verification results:")
        for name, entry in deployment["joints"].items():
            state = "verified" if entry.get("id_verified", False) else "pending"
            print(f"  {name:20} {entry['can_interface']} ID {entry['motor_id']:3}: {state}")
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
    bus = open_bus(channel, args, deployment)
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
                buses[channel] = open_bus(channel, args, deployment)
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
                buses[channel] = open_bus(channel, args, deployment)
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
    ids = commands.add_parser(
        "verify-ids",
        help="Discover responding IDs and identify motors one at a time without motion commands",
    )
    add_can_transport_arguments(ids)
    ids.add_argument("--joint", help="Verify only this configured joint")
    ids.set_defaults(func=cmd_verify_ids)
    set_id = commands.add_parser("set-id", help="Change one motor ID immediately and update the hardware map")
    add_can_transport_arguments(set_id)
    set_id.add_argument("joint")
    set_id.add_argument("--current-id", type=int, required=True)
    set_id.add_argument("--new-id", type=int, required=True)
    set_id.set_defaults(func=cmd_set_id)
    directions = commands.add_parser("check-directions", help="Move each policy joint by 1 degree and verify its sign")
    add_can_transport_arguments(directions)
    directions.add_argument("--joint", help="Check only this policy joint")
    directions.set_defaults(func=cmd_check_directions)
    zero = commands.add_parser(
        "stand-zero",
        help="Zero all configured motors while the robot is manually held in its standing pose",
    )
    add_can_transport_arguments(zero)
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
