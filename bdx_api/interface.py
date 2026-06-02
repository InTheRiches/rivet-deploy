import os
import sys
import time
import signal
import atexit
import threading
import subprocess
import numpy as np
import onnxruntime as ort
from abc import ABC, abstractmethod
from pathlib import Path

from bdx_api.config import load_policy_config, load_isaaclab_config, RobotPolicyConfig
from bdx_api.logger import RuntimeLogger


# ==========================================
# Utils & Keyboard ... (Kept the same)
# ==========================================

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

_HEAD_STEP = 0.1
_HEAD_LIMITS = {
    "Neck_Pitch": (-0.35, 0.87),
    "Head_Pitch": (-0.87, 0.87),
    "Head_Yaw":   (-1.05, 1.05),
    "Head_Roll":  (-0.87, 0.87),
}

class KeyboardController:
    def __init__(self, lin_vel_step=0.1, ang_vel_step=0.1,
                 lin_vel_x_range=(-1.0, 1.0),
                 lin_vel_y_range=(-0.4, 0.4),
                 ang_vel_z_range=(-1.0, 1.0)):
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_yaw = 0.0
        self.lin_vel_step = lin_vel_step
        self.ang_vel_step = ang_vel_step
        self.lin_vel_x_range = lin_vel_x_range
        self.lin_vel_y_range = lin_vel_y_range
        self.ang_vel_z_range = ang_vel_z_range
        self.head_offsets = {k: 0.0 for k in _HEAD_LIMITS}
        self._quit = False
        self._sound_request = None
        self._lock = threading.Lock()
        self._old_term_settings = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        import tty
        import termios
        import select
        fd = sys.stdin.fileno()
        self._old_term_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._quit:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':
                        if select.select([sys.stdin], [], [], 0.01)[0]:
                            ch2 = sys.stdin.read(1)
                            if ch2 == '[':
                                ch3 = sys.stdin.read(1)
                                if ch3 == 'A': ch = 'w'
                                elif ch3 == 'B': ch = 's'
                                elif ch3 == 'D': ch = 'a'
                                elif ch3 == 'C': ch = 'd'
                                else: continue
                            else: continue
                        else:
                            self._quit = True
                            break
                    elif ch == '\x03':
                        self._quit = True
                        break
                    self._handle_key(ch.lower())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_term_settings)

    def _handle_key(self, ch):
        with self._lock:
            # --- Body locomotion ---
            if ch == 'w': self.lin_vel_x = min(self.lin_vel_x + self.lin_vel_step, self.lin_vel_x_range[1])
            elif ch == 's': self.lin_vel_x = max(self.lin_vel_x - self.lin_vel_step, self.lin_vel_x_range[0])
            elif ch == 'a': self.lin_vel_y = min(self.lin_vel_y + self.lin_vel_step, self.lin_vel_y_range[1])
            elif ch == 'd': self.lin_vel_y = max(self.lin_vel_y - self.lin_vel_step, self.lin_vel_y_range[0])
            elif ch == 'q': self.ang_vel_yaw = min(self.ang_vel_yaw + self.ang_vel_step, self.ang_vel_z_range[1])
            elif ch == 'e': self.ang_vel_yaw = max(self.ang_vel_yaw - self.ang_vel_step, self.ang_vel_z_range[0])
            elif ch == ' ':
                self.lin_vel_x = 0.0; self.lin_vel_y = 0.0; self.ang_vel_yaw = 0.0
            # --- Head controls ---
            elif ch == 'i':
                lo, hi = _HEAD_LIMITS["Neck_Pitch"]
                self.head_offsets["Neck_Pitch"] = min(self.head_offsets["Neck_Pitch"] + _HEAD_STEP, hi)
            elif ch == 'k':
                lo, hi = _HEAD_LIMITS["Neck_Pitch"]
                self.head_offsets["Neck_Pitch"] = max(self.head_offsets["Neck_Pitch"] - _HEAD_STEP, lo)
            elif ch == 'l':
                lo, hi = _HEAD_LIMITS["Head_Pitch"]
                self.head_offsets["Head_Pitch"] = min(self.head_offsets["Head_Pitch"] + _HEAD_STEP, hi)
            elif ch == 'j':
                lo, hi = _HEAD_LIMITS["Head_Pitch"]
                self.head_offsets["Head_Pitch"] = max(self.head_offsets["Head_Pitch"] - _HEAD_STEP, lo)
            elif ch == '[':
                lo, hi = _HEAD_LIMITS["Head_Yaw"]
                self.head_offsets["Head_Yaw"] = min(self.head_offsets["Head_Yaw"] + _HEAD_STEP, hi)
            elif ch == ']':
                lo, hi = _HEAD_LIMITS["Head_Yaw"]
                self.head_offsets["Head_Yaw"] = max(self.head_offsets["Head_Yaw"] - _HEAD_STEP, lo)
            # --- Sound controls ---
            elif ch in '0123456789':
                self._sound_request = int(ch)

    def restore_terminal(self):
        if self._old_term_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old_term_settings)
            except Exception: pass

    @property
    def should_quit(self): return self._quit

    def get_head_offsets(self):
        with self._lock:
            return dict(self.head_offsets)

    def get_sound_request(self):
        with self._lock:
            req = self._sound_request
            self._sound_request = None
            return req

    def get_command(self):
        with self._lock:
            return np.array([round(self.lin_vel_x, 2), round(self.lin_vel_y, 2), round(self.ang_vel_yaw, 2)], dtype=np.float32)

class RobotBackend(ABC):
    @abstractmethod
    def get_imu_angular_velocity(self) -> np.ndarray: ...
    @abstractmethod
    def get_projected_gravity(self) -> np.ndarray: ...
    @abstractmethod
    def get_joint_positions(self, joint_ids: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def get_joint_velocities(self, joint_ids: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def set_joint_targets(self, targets: np.ndarray) -> None: ...
    @abstractmethod
    def step(self) -> None: ...
    @abstractmethod
    def is_running(self) -> bool: ...
    @abstractmethod
    def get_joint_map(self, joint_names: list[str]) -> np.ndarray: ...
    @abstractmethod
    def get_default_pos_array(self, num_actuators: int) -> np.ndarray: ...
    @abstractmethod
    def standup(self, duration: float = 2.0) -> None: ...
    @abstractmethod
    def hold_standing_tick(self) -> None: ...
    @abstractmethod
    def activate_policy_gains(self) -> None: ...
    def shutdown(self) -> None: pass  # Override in hardware backend to disable motors

# ==========================================
# Policy Inference Adapters
# ==========================================
class BasePolicyBackend(ABC):
    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        pass

class OnnxPolicyBackend(BasePolicyBackend):
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: obs.reshape(1, -1)})[0][0]

class PyTorchPolicyBackend(BasePolicyBackend):
    def __init__(self, model_path: Path):
        import torch
        # Load the PyTorch JIT model
        self.model = torch.jit.load(str(model_path))
        self.model.eval()

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            obs_tensor = torch.tensor(obs.reshape(1, -1), dtype=torch.float32)
            action = self.model(obs_tensor)
            return action.numpy()[0]
        
# ==========================================
# Policy Runner
# ==========================================
class PolicyRunner:
    def __init__(self, backend: RobotBackend, model_path: Path, decimation: int = 4,
                 cfg_override: RobotPolicyConfig = None):
        self.backend = backend
        self.cfg: RobotPolicyConfig = cfg_override if cfg_override is not None else load_policy_config(model_path)
        
        # --- NEW: Automatically pick ONNX or PyTorch adapter ---
        if model_path.suffix == '.onnx':
            self.model = OnnxPolicyBackend(model_path)
        elif model_path.suffix == '.pt':
            self.model = PyTorchPolicyBackend(model_path)
        else:
            raise ValueError("Model must be .onnx or .pt")

        self.decimation = decimation

        self.joint_map = backend.get_joint_map(self.cfg.joint_names)
        self.num_actuators = len(self.joint_map)

        self.default_pos = backend.get_default_pos_array(self.num_actuators)
        for i, name in enumerate(self.cfg.joint_names):
            mj_id = self.joint_map[i]
            self.default_pos[mj_id] = self.cfg.default_joint_pos[i]

        self.action_scale = np.array(self.cfg.action_scale, dtype=np.float32)

        self.new_actions = np.zeros(len(self.cfg.joint_names), dtype=np.float32)
        self.prev_actions = np.zeros(len(self.cfg.joint_names), dtype=np.float32)
        self.current_interpolated_actions = np.zeros(len(self.cfg.joint_names), dtype=np.float32)
        
        self.dof_targets = self.default_pos.copy()
        self._filtered_targets = self.default_pos.copy()

        self.kb = KeyboardController()
        self.logger = RuntimeLogger(joint_names=self.cfg.joint_names)
        self._interrupted = False
        self._sigint_count = 0

        # Map head/neck joint names → indices in dof_targets (only on hardware backend)
        self._head_joint_indices = {}
        self._head_smooth = {}
        if hasattr(backend, 'all_joint_names') and hasattr(backend, 'num_policy_joints'):
            for i, name in enumerate(backend.all_joint_names[backend.num_policy_joints:],
                                     start=backend.num_policy_joints):
                self._head_joint_indices[name] = i
                self._head_smooth[name] = 0.0


    def _handle_sigint(self, signum, frame):
        self._sigint_count += 1
        if self._sigint_count >= 2:
            print("\n\n  Double Ctrl+C — force quitting!")
            self._cleanup()
            self.backend.shutdown()
            os._exit(1)
        print("\n\n  Ctrl+C received — stopping...")
        self._interrupted = True

    def _cleanup(self):
        try: self.kb.restore_terminal()
        except Exception: pass

    def run(self):
        old_handler = signal.signal(signal.SIGINT, self._handle_sigint)
        atexit.register(self._cleanup)

        print("\n" + "=" * 50)
        print("  READY TO STAND")
        print("=" * 50)
        try:
            input(">>> Press ENTER to stand up (safe standing gains)...")
        except (KeyboardInterrupt, EOFError):
            self._interrupted = True

        if self._interrupted:
            self._cleanup()
            self.backend.shutdown()
            signal.signal(signal.SIGINT, old_handler)
            print("\nShutdown complete.")
            return

        self.backend.standup(duration=2.0)

        print("\nRobot is standing with safe gains. Place it on the floor.")
        print(">>> Press ENTER to deploy the policy (switches to trained Kp/Kd)...")

        enter_pressed = threading.Event()
        def _wait_for_enter():
            input()
            enter_pressed.set()
        waiter = threading.Thread(target=_wait_for_enter, daemon=True)
        waiter.start()

        while not enter_pressed.is_set() and self.backend.is_running() and not self._interrupted:
            self.backend.hold_standing_tick()

        if self._interrupted:
            self._cleanup()
            self.backend.shutdown()
            signal.signal(signal.SIGINT, old_handler)
            print("\nShutdown complete.")
            return

        self.backend.activate_policy_gains()

        # --- Audio setup ---
        _audio_proc = None
        _all = sorted(Path("assets").glob("*.wav")) if Path("assets").exists() else []
        _test = [f for f in _all if f.name == "test.wav"]
        _others = [f for f in _all if f.name != "test.wav"]
        _wav_files = _others[:8] + _test  # test.wav pinned to key 9
        vol = int(151 * 65 / 100)  # 65% volume
        subprocess.run(["amixer", "-c", "2", "cset", "numid=4", f"{vol},{vol}"], capture_output=True)
        subprocess.run(["amixer", "-c", "2", "sset", "PCM Playback Switch", "on"], capture_output=True)
        _startup_sound = Path("assets/test.wav")
        if _startup_sound.exists():
            print(f"[Audio] Playing: {_startup_sound.name}")
            _audio_proc = subprocess.Popen(
                ["aplay", "-D", "plughw:2,0", "-q", str(_startup_sound)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        print("\n" + "=" * 50)
        print("  DEPLOYING POLICY (LIVE MODE)")
        print("=" * 50)
        print("  Locomotion:  W/S=fwd/back  A/D=strafe  Q/E=turn  SPACE=stop")
        if self._head_joint_indices:
            print("  Head:        I/K=neck  J/L=head pitch  [/]=head yaw")
        if _wav_files:
            print(f"  Sounds:      1-9=play  0=stop  ({len(_wav_files)} files in assets/)")
        print("=" * 50)

        it = 0
        loop_times =[]
        
        # --- NEW: Frequency Tracking ---
        freq_start_time = time.time()
        ctrl_ticks = 0
        policy_ticks = 0

        while self.backend.is_running() and not self.kb.should_quit and not self._interrupted:
            loop_start = time.perf_counter()

            # --- SOUND HANDLING ---
            sound_req = self.kb.get_sound_request()
            if sound_req is not None:
                if _audio_proc and _audio_proc.poll() is None:
                    _audio_proc.terminate()
                    _audio_proc = None
                if sound_req > 0:
                    idx = sound_req - 1
                    if idx < len(_wav_files):
                        _audio_proc = subprocess.Popen(
                            ["aplay", "-D", "plughw:2,0", "-q", str(_wav_files[idx])],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )

            # --- POLICY UPDATE STEP (e.g. 50Hz) ---
            if it % self.decimation == 0:
                cmd = self.kb.get_command()

                ang_vel = self.backend.get_imu_angular_velocity()
                projected_gravity = self.backend.get_projected_gravity()
                dof_pos = self.backend.get_joint_positions(self.joint_map)
                dof_vel = self.backend.get_joint_velocities(self.joint_map)

                obs_parts =[]
                for term in self.cfg.observation_names:
                    term = term.strip()
                    # --- NEW: Fetch the scale for this observation, default to 1.0 ---
                    scale = self.cfg.obs_scales.get(term, 1.0)

                    # --- NEW: Apply the scale to each append! ---
                    if term in ("projected_gravity", "imu_projected_gravity"): obs_parts.append(projected_gravity * scale)
                    elif term in ("base_ang_vel", "imu_ang_vel"): obs_parts.append(ang_vel * scale)
                    elif term in ("joint_pos", "dof_pos"): obs_parts.append((dof_pos - self.default_pos[self.joint_map]) * scale)
                    elif term in ("joint_vel", "dof_vel"): obs_parts.append(dof_vel * scale)
                    elif term == "actions": obs_parts.append(self.prev_actions * scale)
                    elif term in ("command", "commands", "velocity_commands"): obs_parts.append(cmd * scale)

                obs = np.concatenate(obs_parts).astype(np.float32)

                # Store old actions and predict new ones
                self.prev_actions = self.new_actions.copy()
                
                # --- NEW: Use the adapter for inference ---
                self.new_actions = self.model.get_action(obs)
                
                policy_ticks += 1

                self.logger.log(
                    cmd=cmd,
                    target_pos=self.dof_targets[self.joint_map],
                    actual_pos=dof_pos,
                    actual_vel=dof_vel,
                    ang_vel=ang_vel,
                    projected_gravity=projected_gravity,
                    actions=self.new_actions, 
                )

            # --- CONTROL INTERPOLATION STEP (e.g. 200Hz) ---
            #if self.decimation > 1:
            #    alpha = (it % self.decimation) / float(self.decimation - 1)
            #    self.current_interpolated_actions = (1.0 - alpha) * self.prev_actions + (alpha * self.new_actions)
            #else:
            #    self.current_interpolated_actions = self.new_actions
            # --- CONTROL STEP (Zero-Order Hold) ---
            # Apply the exact action the policy requested immediately, just like the simulator does.
            self.current_interpolated_actions = self.new_actions

            # Calculate final targets using the smoothed action
            action_delta = self.current_interpolated_actions * self.action_scale
            self.dof_targets[self.joint_map] = self.default_pos[self.joint_map] + action_delta

            # Apply head/neck offsets — smoothed so motion isn't jerky
            if self._head_joint_indices:
                head_offsets = self.kb.get_head_offsets()
                for name, idx in self._head_joint_indices.items():
                    if name in head_offsets:
                        target = head_offsets[name]
                        self._head_smooth[name] += 0.05 * (target - self._head_smooth[name])
                        self.dof_targets[idx] = self._head_smooth[name]
            
            # =================================================================
            # [SAFETY OVERRIDE] 
            # If you want to test hardware WITHOUT the legs moving, uncomment this:
            #self.dof_targets[self.joint_map] = self.default_pos[self.joint_map].copy()
            # =================================================================

            self._filtered_targets = 0.0 * self._filtered_targets + 1 * self.dof_targets
            self.backend.set_joint_targets(self._filtered_targets)
            self.backend.step()
            ctrl_ticks += 1
            
            loop_end = time.perf_counter()
            loop_times.append(loop_end - loop_start)

            # --- DEBUG CONSOLE OUTPUT (Runs every ~1 second at 400Hz ctrl) ---
            if it % 400 == 0 and it > 0:
                l_arr = np.array(loop_times) * 1000 
                loop_times =[] 
                
                # Calculate actual frequencies
                now = time.time()
                dt = now - freq_start_time
                actual_ctrl_hz = ctrl_ticks / dt
                actual_pol_hz = policy_ticks / dt
                
                # Reset tracking variables
                freq_start_time = now
                ctrl_ticks = 0
                policy_ticks = 0

                print("\n" + "-"*55)
                print(f"[DEBUG TICK {it}]")
                print(f"  Frequencies:   Control = {actual_ctrl_hz:5.1f} Hz | Policy = {actual_pol_hz:5.1f} Hz")
                print(f"  Loop Timing:   Avg {l_arr.mean():.2f}ms | Max {l_arr.max():.2f}ms | Std {l_arr.std():.2f}ms")
                
                if hasattr(self.backend, 'get_latency_stats'):
                    lats = self.backend.get_latency_stats()
                    if len(lats) > 0:
                        worst_max = max([stats['max'] for stats in lats.values()])
                        worst_avg = max([stats['avg'] for stats in lats.values()])
                        print(f"  CAN Latency:   Worst Avg: {worst_avg:.2f}ms | Worst Max: {worst_max:.2f}ms")
                
                print(f"  Proj Gravity:  {projected_gravity}")
                print(f"  Interpolated Actions[0:3]: {self.current_interpolated_actions[0:14]}")
                print("-" * 55)
                
            it += 1

        if _audio_proc and _audio_proc.poll() is None:
            _audio_proc.terminate()

        self._cleanup()
        signal.signal(signal.SIGINT, old_handler)
        print(f"\n\nRun complete. {len(self.logger.entries)} samples logged.")

        if self._interrupted and len(self.logger.entries) > 1:
            print("\nGenerating plots...")
            self.logger.plot(save_dir="logs")

        print("Shutting down.")


# ==========================================
# Imitation Policy Runner
# ==========================================
class ImitationPolicyRunner(PolicyRunner):
    """PolicyRunner extended with gait-phase + reference-motion observations."""

    POLICY_DT = 0.02  # 50 Hz control rate

    def __init__(self, backend: RobotBackend, model_path: Path, decimation: int,
                 cfg_override, pkl_path: str):
        super().__init__(backend, model_path, decimation, cfg_override)

        import torch
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location(
            "reference_motion_legs",
            Path(__file__).resolve().parents[1] / "imitate" / "reference_motion_legs.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ReferenceMotionLibraryLegs = mod.ReferenceMotionLibraryLegs
        self._torch = torch
        self.lib = ReferenceMotionLibraryLegs(pkl_path, device="cpu")
        self.phase = 0.0
        self._clip_idx = None

    def _update_clip(self, cmd: np.ndarray):
        torch = self._torch
        cmd_t = torch.tensor([[cmd[0], cmd[1], cmd[2]]], dtype=torch.float32)
        self._clip_idx = self.lib.nearest_clip(cmd_t)  # shape (1,)

    def _step_phase(self):
        torch = self._torch
        period = self.lib.period(self._clip_idx).item()
        self.phase = (self.phase + self.POLICY_DT / period) % 1.0

    def _get_ref_jpos(self) -> np.ndarray:
        torch = self._torch
        phase_t = torch.tensor([self.phase], dtype=torch.float32)
        _, _, ref_jpos, _, _ = self.lib.query(self._clip_idx, phase_t)
        return ref_jpos[0].numpy()

    def run(self):
        import math
        old_handler = signal.signal(signal.SIGINT, self._handle_sigint)
        atexit.register(self._cleanup)

        print("\n" + "=" * 50)
        print("  READY TO STAND  [IMITATION MODE]")
        print("=" * 50)
        input(">>> Press ENTER to stand up (safe standing gains)...")
        self.backend.standup(duration=2.0)

        print("\nRobot is standing. Place it on the floor.")
        print(">>> Press ENTER to deploy the imitation policy...")

        enter_pressed = threading.Event()
        def _wait_for_enter():
            input()
            enter_pressed.set()
        waiter = threading.Thread(target=_wait_for_enter, daemon=True)
        waiter.start()

        while not enter_pressed.is_set() and self.backend.is_running():
            self.backend.hold_standing_tick()

        self.backend.activate_policy_gains()

        print("\n" + "=" * 50)
        print("  DEPLOYING IMITATION POLICY (LIVE MODE)")
        print("=" * 50)

        it = 0
        loop_times = []
        freq_start_time = time.time()
        ctrl_ticks = 0
        policy_ticks = 0

        # Seed phase and clip with the initial command
        cmd = self.kb.get_command()
        self._update_clip(cmd)

        while self.backend.is_running() and not self.kb.should_quit and not self._interrupted:
            loop_start = time.perf_counter()

            if it % self.decimation == 0:
                cmd = self.kb.get_command()
                self._update_clip(cmd)
                self._step_phase()
                ref_jpos = self._get_ref_jpos()

                ang_vel = self.backend.get_imu_angular_velocity()
                projected_gravity = self.backend.get_projected_gravity()
                dof_pos = self.backend.get_joint_positions(self.joint_map)
                dof_vel = self.backend.get_joint_velocities(self.joint_map)

                sin_phase = math.sin(self.phase * 2.0 * math.pi)
                cos_phase = math.cos(self.phase * 2.0 * math.pi)

                obs = np.concatenate([
                    ang_vel,
                    projected_gravity,
                    dof_pos - self.default_pos[self.joint_map],
                    dof_vel,
                    self.prev_actions,
                    cmd,
                    np.array([sin_phase, cos_phase], dtype=np.float32),
                    ref_jpos,
                ]).astype(np.float32)

                self.prev_actions = self.new_actions.copy()
                self.new_actions = self.model.get_action(obs)
                policy_ticks += 1

                self.logger.log(
                    cmd=cmd,
                    target_pos=self.dof_targets[self.joint_map],
                    actual_pos=dof_pos,
                    actual_vel=dof_vel,
                    ang_vel=ang_vel,
                    projected_gravity=projected_gravity,
                    actions=self.new_actions,
                )

            if self.decimation > 1:
                alpha = (it % self.decimation) / float(self.decimation - 1)
                self.current_interpolated_actions = (1.0 - alpha) * self.prev_actions + alpha * self.new_actions
            else:
                self.current_interpolated_actions = self.new_actions

            action_delta = self.current_interpolated_actions * self.action_scale
            self.dof_targets[self.joint_map] = self.default_pos[self.joint_map] + action_delta

            self.backend.set_joint_targets(self.dof_targets)
            self.backend.step()
            ctrl_ticks += 1

            loop_end = time.perf_counter()
            loop_times.append(loop_end - loop_start)

            if it % 100 == 0 and it > 0:
                l_arr = np.array(loop_times) * 1000
                loop_times = []
                now = time.time()
                dt = now - freq_start_time
                actual_ctrl_hz = ctrl_ticks / dt
                actual_pol_hz = policy_ticks / dt
                freq_start_time = now
                ctrl_ticks = 0
                policy_ticks = 0

                print("\n" + "-" * 55)
                print(f"[IMITATION TICK {it}]")
                print(f"  Frequencies:   Control = {actual_ctrl_hz:5.1f} Hz | Policy = {actual_pol_hz:5.1f} Hz")
                print(f"  Loop Timing:   Avg {l_arr.mean():.2f}ms | Max {l_arr.max():.2f}ms")
                print(f"  Phase:         {self.phase:.3f}  cmd: {cmd}")
                print(f"  Proj Gravity:  {projected_gravity}")
                print("-" * 55)

            it += 1

        self._cleanup()
        signal.signal(signal.SIGINT, old_handler)
        print(f"\n\nRun complete. {len(self.logger.entries)} samples logged.")

        if self._interrupted and len(self.logger.entries) > 1:
            print("\nGenerating plots...")
            self.logger.plot(save_dir="logs")

        print("Shutting down.")