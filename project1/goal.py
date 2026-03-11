import numpy as np
import jac


class Goal(object):
    """
    A trivial goal that is always satisfied.
    """

    def __init__(self):
        pass

    def is_satisfied(self, state):
        """
        Determine if the query state satisfies this goal or not.
        """
        return True


class RelocateGoal(Goal):
    """
    The goal for relocating tasks.
    (i.e., pushing the target object into a circular goal region.)
    """

    def __init__(self, x_g=0.2, y_g=-0.2, r_g=0.1):
        """
        args: x_g: The x-coordinate of the center of the goal region.
              y_g: The y-coordinate of the center of the goal region.
              r_g: The radius of the goal region.
        """
        super(RelocateGoal, self).__init__()
        self.x_g, self.y_g, self.r_g = x_g, y_g, r_g

    def is_satisfied(self, state):
        """
        Check if the state satisfies the RelocateGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        """
        stateVec = state["stateVec"]
        x_tgt, y_tgt = stateVec[7], stateVec[8] # position of the target object
        if np.linalg.norm([x_tgt - self.x_g, y_tgt - self.y_g]) < self.r_g:
            return True
        else:
            return False


class GraspGoal(Goal):
    """
    The goal for grasping tasks.
    (i.e., approaching the end-effector to a pose that can grasp the target object.)
    """

    def __init__(self):
        super(GraspGoal, self).__init__()
        self.jac_solver = jac.JacSolver() # the jacobian solver

    def is_satisfied(self, state):
        """
        Check if the state satisfies the GraspGoal or not.
        args: state: The state to check.
                     Type: dict, {"stateID", int, "stateVec", numpy.ndarray}
        returns: True or False.
        """
        ########## TODO ##########
        
        state_vec = state["stateVec"]

        joint_values = state_vec[:7]

        # target cube pose
        x_target, y_target, theta_target = state_vec[7:10]
        target_xy = np.array([x_target, y_target])

        # end-effector pose
        ee_pos, ee_quat = self.jac_solver.forward_kinematics(joint_values)
        ee_xy = np.array([ee_pos[0] - 0.4, ee_pos[1] - 0.2])
        
        # quaternion -> yaw -> planar direction
        x, y, z, w = ee_quat
        ee_theta = np.arctan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))
        ee_dir = np.array([np.cos(ee_theta), np.sin(ee_theta)])

        diff = target_xy - ee_xy

        d2 = abs(np.dot(diff, ee_dir))
        d1 = abs(diff[0]*ee_dir[1] - diff[1]*ee_dir[0])

        def wrap_to_pi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        gamma = min(
            abs(wrap_to_pi(ee_theta - (theta_target + k * np.pi / 2)))
            for k in range(4)
        )

        return (d1 < 0.01) and (d2 < 0.02) and (gamma < 0.2)

        ##########################
        