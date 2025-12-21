import pybullet as p
import pybullet_data
import time
import math
import os


def main():
    # Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load the plane
    p.loadURDF("plane.urdf")

    # Load the Franka Panda robot
    # Calculate the path to the URDF file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path: drives/rsbullet/rsbullet/scripts/pybullet_robot.py
    # URDF: drives/asserts/franka_panda/panda.urdf
    urdf_path = os.path.join(script_dir, "../../../asserts/franka_panda/panda.urdf")

    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        return

    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    robotId = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True)

    # Target joint angles
    # [0., -FRAC_PI_4, 0., -3. * FRAC_PI_4, 0., FRAC_PI_2, FRAC_PI_4]
    target_joints = [
        0.0,
        -math.pi / 4,
        0.0,
        -3.0 * math.pi / 4,
        0.0,
        math.pi / 2,
        math.pi / 4,
    ]

    # Identify controllable joints (revolute joints)
    num_joints = p.getNumJoints(robotId)
    controllable_joints = []
    for i in range(num_joints):
        info = p.getJointInfo(robotId, i)
        joint_type = info[2]
        # qIndex > -1 implies it has a state (is not fixed)
        if joint_type == p.JOINT_REVOLUTE:
            controllable_joints.append(i)

    print(f"Found {len(controllable_joints)} revolute joints.")

    # Set joint positions
    if len(controllable_joints) >= len(target_joints):
        for i, target_pos in enumerate(target_joints):
            joint_index = controllable_joints[i]
            # Reset joint state sets the position instantly (teleport)
            p.resetJointState(robotId, joint_index, target_pos)
            # Set motor control to maintain the position
            p.setJointMotorControl2(
                robotId, joint_index, p.POSITION_CONTROL, targetPosition=target_pos
            )
            print(f"Set joint {joint_index} to {target_pos}")
    else:
        print(
            f"Warning: Robot has {len(controllable_joints)} revolute joints, but {len(target_joints)} target angles provided."
        )

    # Simulation loop
    print("Starting simulation loop...")
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()
