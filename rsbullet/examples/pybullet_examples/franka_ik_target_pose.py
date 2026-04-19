import math
import time

import pybullet as p
import pybullet_data


def main() -> None:
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", [0.0, 0.0, 0.0], useFixedBase=True)

    arm_joint_indices = []
    for joint_index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_index)
        if info[2] == p.JOINT_REVOLUTE:
            arm_joint_indices.append(joint_index)
            if len(arm_joint_indices) == 7:
                break

    if len(arm_joint_indices) != 7:
        raise RuntimeError("failed to find 7 Franka arm revolute joints")

    default_arm_pose = [0.0, -0.6, 0.0, -2.1, 0.0, 1.6, 0.8]
    for i, joint_index in enumerate(arm_joint_indices):
        p.resetJointState(robot_id, joint_index, default_arm_pose[i])

    end_effector_link_index = 11
    target_pos = [0.45, 0.00, 0.45]
    target_orn = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])

    lower_limits = [
        -2.8973,
        -1.7628,
        -2.8973,
        -3.0718,
        -2.8973,
        -0.0175,
        -2.8973,
        0.0,
        0.0,
    ]
    upper_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04]
    joint_ranges = [u - l for l, u in zip(lower_limits, upper_limits)]
    rest_poses = default_arm_pose + [0.02, 0.02]

    tick = 0
    try:
        while p.isConnected(client_id):
            joint_poses = p.calculateInverseKinematics(
                robot_id,
                end_effector_link_index,
                target_pos,
                target_orn,
                lowerLimits=lower_limits,
                upperLimits=upper_limits,
                jointRanges=joint_ranges,
                restPoses=rest_poses,
                maxNumIterations=200,
                residualThreshold=1e-4,
            )

            for i, joint_index in enumerate(arm_joint_indices):
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=200.0,
                )

            p.stepSimulation()
            time.sleep(1.0 / 240.0)

            tick += 1
            if tick % 240 == 0:
                ee_state = p.getLinkState(
                    robot_id,
                    end_effector_link_index,
                    computeLinkVelocity=0,
                    computeForwardKinematics=1,
                )
                ee_pos = ee_state[4]
                pos_err = math.sqrt(
                    (ee_pos[0] - target_pos[0]) ** 2
                    + (ee_pos[1] - target_pos[1]) ** 2
                    + (ee_pos[2] - target_pos[2]) ** 2
                )
                print(
                    "ee=({:.3f}, {:.3f}, {:.3f}), target=({:.3f}, {:.3f}, {:.3f}), pos_err={:.4f}".format(
                        ee_pos[0],
                        ee_pos[1],
                        ee_pos[2],
                        target_pos[0],
                        target_pos[1],
                        target_pos[2],
                        pos_err,
                    )
                )
    finally:
        if p.isConnected(client_id):
            p.disconnect(client_id)


if __name__ == "__main__":
    main()
