# Rivet deployment

This repository contains the policy runner and the staged tools for commissioning Rivet's RobStride motors. The deployment map is [config/rivet_deployment.yaml](config/rivet_deployment.yaml). It binds named joints to CAN interfaces, motor IDs, model types, and encoder signs. The ONNX file's `joint_names` metadata defines policy order; the mapping connects those names to physical motors.

RobStride 06 is supported for the hip yaw and roll joints. The deployment map selects the RS06 command and feedback ranges for those motors; the RS06 armature and chirp-tuned gains are also reflected in the MuJoCo models and chirp utility. When a policy exports valid gains, those ONNX gains remain in use during policy control; the RS06 gains here supply the fallback and chirp settings.

## Choose the policy

Set `policy_model` in `config/rivet_deployment.yaml` to the MuJoCo-exported `.onnx` file. You can also pass `--model` to either command below or set `BDX_MODEL_PATH`. The default currently points to `models/walk.onnx`.

Before moving hardware, check the policy names and each mapping:

```sh
python scripts/commission_rivet.py status
```

## Commission in order

1. **Verify motor IDs.** The tool reads each motor's `mechpos` through the local RobStride client, asks you to confirm the physical joint, and offers to update a mismatched ID immediately. An ID update is sent with `Client.update_id` and then checked at the new ID. The mapping records the verified ID.

   ```sh
   python scripts/commission_rivet.py verify-ids
   ```

   To change a single ID directly, provide the joint name and both IDs:

   ```sh
   python scripts/commission_rivet.py set-id Left_Knee --current-id 4 --new-id 17
   ```

   The command asks you to type `CHANGE`, applies the change on that motor's CAN bus, checks the new ID, and saves the map. IDs must be unique in the configured robot map. Use an unused temporary ID when resolving an ID swap.

2. **Put Rivet in its standing pose and zero all motors.** Manually place and securely support the robot in the desired standing pose, then run the tool. It sends `zero_pos` to every configured motor in a batch across the CAN buses. After all motors acknowledge, the tool asks whether to save their current configurations, including the new zero offsets, to EEPROM. Choosing **yes** makes the offsets persist across power cycles; power-cycle the motors for the saved zero calibration to take effect. Choosing **no** leaves the new offsets active for the current powered session, and the previously saved offsets return after a reset or power cycle. This step does not move motors; direction signs are still unverified.

   ```sh
   python scripts/commission_rivet.py stand-zero
   ```

3. **Check motor and direction one joint at a time.** The tool enables only the selected motor and commands a one-degree low-gain movement from the newly zeroed position. Confirm that exactly the named joint moved in its policy-positive direction. If it moved the other way, the tool flips that joint's sign and repeats the check. If a different joint moved, it invalidates that ID confirmation so the map can be corrected. The default sequence covers the policy joints and additional mapped motors.

   ```sh
   python scripts/commission_rivet.py check-directions
   ```

4. **Ramp into the real policy.** Real deployment requires verified IDs, verified directions for every active joint, fresh feedback from every active motor, and a typed hardware-ready confirmation. The Orbbec camera IMU is the default real-hardware IMU. The first policy stage is 0% action authority with standing gains. Press Enter at each prompt only after inspecting the robot; the default stages are 0%, 10%, 25%, 50%, 75%, and 100%. Ctrl+C stops the run.

   ```sh
   python scripts/deploy.py --backend real --legs
   ```

   Use `--authority-levels 0,0.05,0.1,0.2,0.35,0.5,0.75,1` to add smaller stages. Pass `--model` when selecting a policy other than the configured default.

   Install the deployment and Orbbec bindings with `pip install -e '.[real,orbbec]'`, then run Orbbec's Linux environment setup once to install camera permissions/udev rules. Before policy deployment, verify the live signals with:

   ```sh
   python scripts/check_orbbec_imu.py
   ```

   The tool calibrates gyro bias from the first 50 samples, so keep Rivet still while it starts. It prints angular velocity and projected gravity in policy/body coordinates. In the deployment map, `imu.sensor_to_base_rotation` rotates the camera's raw sensor axes into the robot's policy axes; the default identity matrix assumes those axes already align. Confirm that the projected gravity is approximately `[0, 0, -1]` when Rivet is upright and changes on the expected axes when tilted. If the camera is mounted at an angle, set this 3×3 rotation before deployment. `--imu_backend xsens --imu_port /dev/ttyUSB0` selects the existing serial Xsens path instead.

The commissioning commands are interactive because motor ID changes, joint motion, and encoder zeroing act on physical hardware. Keep Rivet supported, clear the area, and keep the emergency stop ready during these steps. `zero_pos` sets the current encoder position as zero; it does not find a mechanical home.

## Notes

- `--no_imu` uses zero angular velocity and upright gravity values for bench debugging; tilt detection is inactive in this mode. Real policy deployment should use the Orbbec IMU unless explicitly selecting another sensor.
- `direction` in the deployment map is applied to both measured positions/velocities and outgoing position targets.
- `id_verified` and `direction_verified` default to false. Real hardware deployment stays blocked until the relevant joints have been commissioned.
- `robstride/client.py` and `setup/robstride/client.py` expose the existing ID change method, batched `zero_pos_many` helper, and `save_configuration` EEPROM command. `stand-zero` uses the EEPROM command only after you explicitly choose to save.
