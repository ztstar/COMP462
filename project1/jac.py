import copy
import numpy as np
import pybullet as p
import pybullet_data as pd
import pybullet_utils.bullet_client as bc


class JacSolver(object):
    """
    The Jacobian solver for the 7-DoF Franka Panda robot.
    """

    def __init__(self):
        self.bullet_client = bc.BulletClient(connection_mode=p.DIRECT)
        self.bullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    def forward_kinematics(self, joint_values):
        """
        Calculate the Forward Kinematics of the robot given joint angle values.
        args: joint_values: The joint angle values of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:       pos: The position of the end-effector.
                            Type: numpy.ndarray [x, y, z]
                      quat: The orientation of the end-effector represented by quaternion.
                            Type: numpy.ndarray [x, y, z, w]
        """
        for j in range(7):
            self.bullet_client.resetJointState(self.panda, j, joint_values[j])
        ee_state = self.bullet_client.getLinkState(self.panda, linkIndex=11)
        pos, quat = np.array(ee_state[4]), np.array(ee_state[5])
        return pos, quat

    def get_jacobian_matrix(self, joint_values):
        """
        Numerically calculate the Jacobian matrix based on joint angles.
        args: joint_values: The joint angles of the query configuration.
                            Type: numpy.ndarray of shape (7,)
        returns:         J: The calculated Jacobian matrix.
                            Type: numpy.ndarray of shape (6, 7)
        """
        ########## TODO ##########
        J = np.zeros(shape=(6, 7))

        h = 1e-6

        for i in range(7):
            q_plus = np.array(joint_values, dtype=float)
            q_minus = np.array(joint_values, dtype=float)

            q_plus[i] += h
            q_minus[i] -= h

            pos_plus, quat_plus = self.forward_kinematics(q_plus)
            pos_minus, quat_minus = self.forward_kinematics(q_minus)

            # Linear part
            J[:3, i] = (pos_plus - pos_minus) / (2*h)

            # Angular part
            # R(R-) = (R+)， R = (R+)(R-)^T
            quat_minus_inv = self.bullet_client.invertTransform(
                [0, 0, 0], quat_minus.tolist()
            )[1]

            quat_rel = self.bullet_client.multiplyTransforms(
                [0, 0, 0], quat_plus.tolist(),
                [0, 0, 0], quat_minus_inv
            )[1]

            axis, angle = self.bullet_client.getAxisAngleFromQuaternion(quat_rel)

            J[3:, i] = (np.array(axis) * angle) / (2*h)

        ##########################
        return J