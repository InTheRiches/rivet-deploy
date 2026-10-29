"""CAN transport setup shared by deployment and commissioning tools."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence


def add_can_transport_arguments(parser: ArgumentParser) -> None:
    """Add SocketCAN (default) and serial SLCAN command-line options."""
    parser.add_argument(
        "--can-interface",
        choices=("socketcan", "slcan"),
        default="socketcan",
        help="CAN transport (default: socketcan)",
    )
    parser.add_argument(
        "--slcan-channel",
        help="Serial device for SLCAN, for example /dev/cu.usbmodem1234 on macOS",
    )
    parser.add_argument(
        "--slcan-bitrate",
        type=int,
        default=1_000_000,
        help="CAN bus bitrate for SLCAN (default: 1000000)",
    )
    parser.add_argument(
        "--slcan-tty-baudrate",
        type=int,
        default=115_200,
        help="Serial baud rate for the SLCAN adapter (default: 115200)",
    )


def can_transport_kwargs(args: Namespace) -> dict[str, object]:
    """Return the transport options accepted by HardwareBackend."""
    return {
        "can_interface": args.can_interface,
        "slcan_channel": args.slcan_channel,
        "slcan_bitrate": args.slcan_bitrate,
        "slcan_tty_baudrate": args.slcan_tty_baudrate,
    }


def validate_can_transport(
    configured_channels: Sequence[str],
    *,
    can_interface: str = "socketcan",
    slcan_channel: str | None = None,
) -> None:
    """Validate how configured logical buses map to the selected transport."""
    if can_interface not in {"socketcan", "slcan"}:
        raise ValueError(f"Unsupported CAN interface {can_interface!r}")
    if can_interface == "socketcan":
        if slcan_channel:
            raise ValueError("--slcan-channel can only be used with --can-interface slcan")
        return
    if not slcan_channel:
        raise ValueError("SLCAN requires --slcan-channel with the serial device path")
    if len(configured_channels) != 1:
        raise ValueError(
            "SLCAN uses one serial adapter for one CAN bus; configure exactly one logical "
            "channel in can_channels"
        )


def open_can_bus(
    logical_channel: str,
    configured_channels: Sequence[str],
    *,
    can_interface: str = "socketcan",
    slcan_channel: str | None = None,
    slcan_bitrate: int = 1_000_000,
    slcan_tty_baudrate: int = 115_200,
):
    """Open one configured logical CAN bus over SocketCAN or an SLCAN serial port."""
    validate_can_transport(
        configured_channels,
        can_interface=can_interface,
        slcan_channel=slcan_channel,
    )
    if logical_channel not in configured_channels:
        raise ValueError(f"Unknown configured CAN channel {logical_channel!r}")

    import can

    if can_interface == "socketcan":
        return can.interface.Bus(interface="socketcan", channel=logical_channel)
    return can.interface.Bus(
        interface="slcan",
        channel=slcan_channel,
        bitrate=slcan_bitrate,
        tty_baudrate=slcan_tty_baudrate,
    )
