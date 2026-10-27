# bdx_api/config.py

import os
import onnx
import yaml  # <--- NEW IMPORT
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict  # <--- ADDED Dict

from bdx_api.actuator_specs import ROBSTRIDE_06_GAINS
from bdx_api.deployment import PROJECT_ROOT, load_deployment_config


# ==========================================
# Model Path Resolution
# ==========================================

def resolve_model_path() -> Path:
    env_path = os.getenv("BDX_MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"BDX_MODEL_PATH={env_path} does not exist")
        return p.resolve()

    _, deployment = load_deployment_config()
    configured_model = Path(deployment.get("policy_model", "models/walk.onnx"))
    default_path = configured_model if configured_model.is_absolute() else PROJECT_ROOT / configured_model

    if not default_path.exists():
        raise FileNotFoundError(
            f"No model found at {default_path}. "
            "Update policy_model in config/rivet_deployment.yaml or set BDX_MODEL_PATH."
        )

    return default_path.resolve()


MODEL_PATH = resolve_model_path()


# ==========================================
# Dataclass Definition
# ==========================================

@dataclass(frozen=True)
class RobotPolicyConfig:
    obs_dim: int
    action_dim: int
    joint_names: List[str]
    joint_stiffness: List[float]
    joint_damping: List[float]
    default_joint_pos: List[float]
    action_scale: List[float]
    observation_names: List[str]
    command_names: List[str]
    obs_scales: Dict[str, float]


# ==========================================
# Utility Parsers
# ==========================================

def _parse_float_list(value: str) -> List[float]:
    cleaned = value.replace("[", "").replace("]", "").strip()
    return [float(x.strip()) for x in cleaned.split(",") if x.strip()]


def _parse_str_list(value: str) -> List[str]:
    cleaned = value.replace("[", "").replace("]", "").strip()
    return [x.strip().strip("'\"") for x in cleaned.split(",") if x.strip()]


# ==========================================
# ONNX Metadata Loader
# ==========================================

def load_policy_config(model_path: Path = None) -> RobotPolicyConfig:
    if model_path is None:
        from bdx_api.config import resolve_model_path
        model_path = resolve_model_path()

    # ==========================================
    # 1. PYTORCH / ISAACLAB PATH (Reads YAML)
    # ==========================================
    if model_path.suffix == '.pt':
        yaml_path = model_path.with_suffix('.yaml')
        if not yaml_path.exists():
            raise FileNotFoundError(f"PyTorch models require a sidecar config. Missing: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        return RobotPolicyConfig(
            obs_dim=data["obs_dim"],
            action_dim=data["action_dim"],
            joint_names=data["joint_names"],
            joint_stiffness=data["joint_stiffness"],
            joint_damping=data["joint_damping"],
            default_joint_pos=data["default_joint_pos"],
            action_scale=data["action_scale"],
            observation_names=data["observation_names"],
            command_names=data.get("command_names", ["lin_vel_x", "lin_vel_y", "ang_vel_z"]),
            obs_scales=data.get("obs_scales", {})  # Load scales if they exist
        )

    # ==========================================
    # 2. ONNX / MJLAB PATH (Reads Metadata)
    # ==========================================
    elif model_path.suffix == '.onnx':
        model = onnx.load(str(model_path))

        if len(model.graph.input) != 1:
            raise ValueError(f"Expected 1 input, got {len(model.graph.input)}")
        if len(model.graph.output) != 1:
            raise ValueError(f"Expected 1 output, got {len(model.graph.output)}")

        input_tensor = model.graph.input[0]
        output_tensor = model.graph.output[0]
        obs_dim = input_tensor.type.tensor_type.shape.dim[1].dim_value
        action_dim = output_tensor.type.tensor_type.shape.dim[1].dim_value

        metadata = {prop.key: prop.value for prop in model.metadata_props}

        required_keys =[
            "joint_names", "joint_stiffness", "joint_damping",
            "default_joint_pos", "action_scale", "observation_names",
        ]
        for key in required_keys:
            if key not in metadata:
                raise KeyError(f"Missing required ONNX metadata key: '{key}'")

        joint_names = _parse_str_list(metadata["joint_names"])
        joint_stiffness = _parse_float_list(metadata["joint_stiffness"])
        joint_damping = _parse_float_list(metadata["joint_damping"])

        # Detect broken export (Kp all ~1.0, Kd all ~0.0) and fall back to known-good gains
        if all(abs(kp - 1.0) < 0.01 for kp in joint_stiffness):
            print("[WARNING] ONNX gains look broken (Kp=1.0, Kd≈0). Using POLICY_GAINS_FALLBACK.")
            joint_stiffness = [POLICY_GAINS_FALLBACK[n][0] for n in joint_names]
            joint_damping   = [POLICY_GAINS_FALLBACK[n][1] for n in joint_names]

        return RobotPolicyConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            joint_names=joint_names,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            default_joint_pos=_parse_float_list(metadata["default_joint_pos"]),
            action_scale=_parse_float_list(metadata["action_scale"]),
            observation_names=_parse_str_list(metadata["observation_names"]),
            command_names=_parse_str_list(metadata.get("command_names", "lin_vel_x,lin_vel_y,ang_vel_z")),
            obs_scales={}
        )
    else:
        raise ValueError(f"Unsupported model extension: {model_path.suffix}. Use .onnx or .pt")


# ==========================================
# Safe Standing Gains
# ==========================================

# Higher than policy gains — used to hold the robot stiffly upright
# before the learned policy takes over.
STANDUP_GAINS = {
    "Left_Hip_Yaw":    (200, 5.027),
    "Left_Hip_Roll":   (200, 5.027),
    "Left_Hip_Pitch":  (200, 5.027),
    "Left_Knee":       (200, 5.027),
    "Left_Ankle":      (60, 1.056),
    "Right_Hip_Yaw":   (200, 5.027),
    "Right_Hip_Roll":  (200, 5.027),
    "Right_Hip_Pitch": (200, 5.027),
    "Right_Knee":      (200, 5.027),
    "Right_Ankle":     (60, 1.056),
    "Neck_Pitch":      (15, 2),
    "Head_Pitch":      (5.0, 3.0),
    "Head_Yaw":        (5.0, 3.0),
    "Head_Roll":       (5.0, 3.0),
}

# ==========================================
# Fallback Policy Gains
# ==========================================
# Used if the ONNX exporter accidentally writes Kp=1.0 and Kd=0.0
POLICY_GAINS_FALLBACK = {
    "Left_Hip_Yaw":    ROBSTRIDE_06_GAINS,
    "Left_Hip_Roll":   ROBSTRIDE_06_GAINS,
    "Left_Hip_Pitch":  (78.957, 5.027),
    "Left_Knee":       (78.957, 5.027),
    "Left_Ankle":      (16.581, 1.056),
    "Right_Hip_Yaw":   ROBSTRIDE_06_GAINS,
    "Right_Hip_Roll":  ROBSTRIDE_06_GAINS,
    "Right_Hip_Pitch": (78.957, 5.027),
    "Right_Knee":      (78.957, 5.027),
    "Right_Ankle":     (16.581, 1.056),
    "Neck_Pitch":      (15, 2),
    "Head_Pitch":      (5.0, 3.0),
    "Head_Yaw":        (5.0, 3.0),
    "Head_Roll":       (5.0, 3.0),
}

# ==========================================
# IsaacLab Hardcoded Config (from env.yaml)
# ==========================================
# Joint order follows URDF articulation order (preserve_order=False resolves to this)
_ISAACLAB_JOINT_NAMES = [
    "Left_Hip_Yaw",   "Right_Hip_Yaw",
    "Left_Hip_Roll",  "Right_Hip_Roll",
    "Left_Hip_Pitch", "Right_Hip_Pitch",
    "Left_Knee",      "Right_Knee",
    "Left_Ankle",     "Right_Ankle",
]
# Gains from env.yaml actuators: hips_and_knees kp=78.957, kd=5.027; ankles kp=16.581, kd=1.056
_IL_KP = [78.957, 78.957, 78.957, 78.957, 78.957, 78.957, 78.957, 78.957, 16.581, 16.581]
_IL_KD = [5.027,  5.027,  5.027,  5.027,  5.027,  5.027,  5.027,  5.027,  1.056,  1.056]

def load_isaaclab_config() -> RobotPolicyConfig:
    """
    Returns the hardcoded config derived from the IsaacLab env.yaml.
    Use this when deploying a policy trained in IsaacLab (--isaaclab flag).
    Obs order: imu_ang_vel(3), imu_projected_gravity(3), joint_pos(10), joint_vel(10), actions(10), velocity_commands(3) = 39
    Joint axes in the URDF match the MuJoCo XML exactly — no sign correction needed.
    """
    return RobotPolicyConfig(
        obs_dim=39,
        action_dim=10,
        joint_names=_ISAACLAB_JOINT_NAMES,
        joint_stiffness=_IL_KP,
        joint_damping=_IL_KD,
        default_joint_pos=[0.0] * 10,  # init_state.joint_pos: .*: 0.0
        action_scale=[0.5] * 10,        # actions.joint_pos.scale: 0.5
        observation_names=["imu_ang_vel", "imu_projected_gravity", "joint_pos", "joint_vel", "actions", "velocity_commands"],
        command_names=["lin_vel_x", "lin_vel_y", "ang_vel_z"],
        obs_scales={"imu_ang_vel": 0.2, "joint_vel": 0.05},  # from policy obs scales in env.yaml
    )


# ==========================================
# Imitation Legs Config
# ==========================================
# Joint order: Left side first (Yaw, Roll, Pitch, Knee, Ankle), then Right side
_IMITATION_JOINT_NAMES = [
    "Left_Hip_Yaw",  "Left_Hip_Roll",  "Left_Hip_Pitch",  "Left_Knee",  "Left_Ankle",
    "Right_Hip_Yaw", "Right_Hip_Roll", "Right_Hip_Pitch", "Right_Knee", "Right_Ankle",
]
_IMITATION_KP = [78.957, 78.957, 78.957, 78.957, 16.581,
                 78.957, 78.957, 78.957, 78.957, 16.581]
_IMITATION_KD = [5.027, 5.027, 5.027, 5.027, 1.056,
                 5.027, 5.027, 5.027, 5.027, 1.056]
# Per-joint action scales from training spec
_IMITATION_ACTION_SCALE = [0.1330, 0.1330, 0.1330, 0.1330, 0.1643,
                            0.1330, 0.1330, 0.1330, 0.1330, 0.1643]

def load_imitation_config() -> RobotPolicyConfig:
    """
    Hardcoded config for the BDX-R imitation legs policy.
    Obs (51): ang_vel(3), proj_grav(3), joint_pos(10), joint_vel(10),
              actions(10), command(3), imitation_phase(2), ref_joint_pos(10)
    """
    return RobotPolicyConfig(
        obs_dim=51,
        action_dim=10,
        joint_names=_IMITATION_JOINT_NAMES,
        joint_stiffness=_IMITATION_KP,
        joint_damping=_IMITATION_KD,
        default_joint_pos=[0.0] * 10,
        action_scale=_IMITATION_ACTION_SCALE,
        observation_names=["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
                           "actions", "command", "imitation_phase", "ref_joint_pos"],
        command_names=["lin_vel_x", "lin_vel_y", "ang_vel_z"],
        obs_scales={},
    )


# ==========================================
# Debug
# ==========================================

if __name__ == "__main__":
    cfg = load_policy_config()
    print(f"Obs dim:    {cfg.obs_dim}")
    print(f"Action dim: {cfg.action_dim}")
    print(f"Joints:     {cfg.joint_names}")
    print(f"Kp:         {cfg.joint_stiffness}")
    print(f"Kd:         {cfg.joint_damping}")
    print(f"Defaults:   {cfg.default_joint_pos}")
    print(f"Scale:      {cfg.action_scale}")
    print(f"Obs names:  {cfg.observation_names}")
    print(f"Cmd names:  {cfg.command_names}")
