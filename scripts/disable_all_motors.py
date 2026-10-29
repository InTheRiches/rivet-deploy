#!/usr/bin/env python3
"""Send disable commands to every motor ID in Rivet's deployment map."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bdx_api.can_bus import add_can_transport_arguments, open_can_bus
from bdx_api.deployment import load_deployment_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send RobStride disable commands to the motors listed in Rivet's deployment map."
    )
    parser.add_argument(
        "--deployment-config",
        help="Path to Rivet hardware map (defaults to config/rivet_deployment.yaml)",
    )
    add_can_transport_arguments(parser)
    args = parser.parse_args()

    config_path, deployment = load_deployment_config(args.deployment_config)
    channels = deployment["can_channels"]
    motors = [
        (name, entry["can_interface"], entry["motor_id"])
        for name, entry in deployment["joints"].items()
    ]

    print(f"Deployment map: {config_path}")
    print("Disable targets:")
    for name, channel, motor_id in motors:
        print(f"  {name:<22} {channel} ID {motor_id}")
    print("\nStop any active deploy process first and support the robot securely.")
    print("Disabling can remove motor holding torque and let an unsupported robot fall.")
    print("This sends disable commands only to IDs listed above; it does not move joints.")
    try:
        confirmation = input("Type DISABLE to proceed: ")
    except (KeyboardInterrupt, EOFError):
        print("Cancelled; no CAN commands sent.")
        return 1
    if confirmation.strip() != "DISABLE":
        print("Cancelled; no CAN commands sent.")
        return 1

    from Robstride.client import Client, MotorMsg

    buses = {}
    clients = {}
    pending = {(channel, motor_id): name for name, channel, motor_id in motors}
    last_errors = {}
    try:
        for channel in channels:
            buses[channel] = open_can_bus(
                channel,
                channels,
                can_interface=args.can_interface,
                slcan_channel=args.slcan_channel,
                slcan_bitrate=args.slcan_bitrate,
                slcan_tty_baudrate=args.slcan_tty_baudrate,
            )
            clients[channel] = Client(buses[channel], retry_count=0, recv_timeout=1.0)

        # Clear queued replies from earlier traffic so they cannot be mistaken
        # for acknowledgements to this disable sweep.
        for bus in buses.values():
            while bus.recv(timeout=0) is not None:
                pass

        for attempt in range(2):
            if not pending:
                break
            label = "first" if attempt == 0 else "retry"
            print(f"\nSending {label} disable request to all pending motors...")

            # Fan out to all pending motors before waiting on any one motor.
            for (channel, motor_id), _name in list(pending.items()):
                client = clients[channel]
                try:
                    client.bus.send(client._rs_msg(
                        MotorMsg.Disable,
                        client.host_can_id,
                        motor_id,
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ))
                except Exception as exc:
                    last_errors[(channel, motor_id)] = f"send error: {exc}"

            # Collect replies independently on each configured bus.
            for channel in channels:
                channel_pending = {
                    motor_id
                    for pending_channel, motor_id in pending
                    if pending_channel == channel
                }
                if not channel_pending:
                    continue

                client = clients[channel]
                deadline = time.monotonic() + 1.0
                while channel_pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        message = buses[channel].recv(timeout=remaining)
                    except Exception as exc:
                        for pending_id in channel_pending:
                            last_errors[(channel, pending_id)] = f"receive error: {exc}"
                        break
                    if message is None:
                        break
                    if message.is_error_frame:
                        continue
                    try:
                        message_type, motor_id, host_id = client._parse_resp_abitration_id(
                            message.arbitration_id
                        )
                        if (message_type != MotorMsg.Feedback.value
                                or host_id != client.host_can_id
                                or motor_id not in channel_pending):
                            continue
                    except Exception as exc:
                        # Ignore malformed or unrelated traffic and keep
                        # waiting for the remaining motor acknowledgements.
                        continue

                    try:
                        response = client._parse_feedback_resp(message, motor_id)
                    except Exception as exc:
                        last_errors[(channel, motor_id)] = f"acknowledgement error: {exc}"
                        continue

                    name = pending.pop((channel, motor_id))
                    channel_pending.remove(motor_id)
                    errors = ", ".join(error.name for error in response.errors) or "none"
                    print(
                        f"  ACK {name} ({channel} ID {motor_id}): "
                        f"mode={response.mode.name}, errors={errors}"
                    )
    finally:
        for bus in buses.values():
            try:
                bus.shutdown()
            except Exception:
                pass

    for (channel, motor_id), name in pending.items():
        detail = last_errors.get((channel, motor_id), "no acknowledgement received")
        print(
            f"  UNCONFIRMED {name} ({channel} ID {motor_id}): {detail}. "
            "The command may have been sent; check this motor manually."
        )

    if pending:
        print(f"\nDisable acknowledgement missing for {len(pending)} motor(s).")
        return 2
    print(f"\nAll {len(motors)} configured motors acknowledged disable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
