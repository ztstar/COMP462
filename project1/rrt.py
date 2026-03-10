import numpy as np
import time
import samplers
import utils


class Tree(object):
    """
    The tree class for Kinodynamic RRT.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.dim_state = self.pdef.get_state_dimension()
        self.nodes = []
        self.stateVecs = np.empty(shape=(0, self.dim_state))

    def add(self, node):
        """
        Add a new node into the tree.
        """
        self.nodes.append(node)
        self.stateVecs = np.vstack((self.stateVecs, node.state['stateVec']))
        assert len(self.nodes) == self.stateVecs.shape[0]

    def nearest(self, rstateVec):
        """
        Find the node in the tree whose state vector is nearest to "rstateVec".
        """
        dists = self.pdef.distance_func(rstateVec, self.stateVecs)
        nnode_id = np.argmin(dists)
        return self.nodes[nnode_id]

    def size(self):
        """
        Query the size (number of nodes) of the tree. 
        """
        return len(self.nodes)
    

class Node(object):
    """
    The node class for Tree.
    """

    def __init__(self, state):
        """
        args: state: The state associated with this node.
                     Type: dict, {"stateID": int, 
                                  "stateVec": numpy.ndarray}
        """
        self.state = state
        self.control = None # the control asscoiated with this node
        self.parent = None # the parent node of this node

    def get_control(self):
        return self.control

    def get_parent(self):
        return self.parent

    def set_control(self, control):
        self.control = control

    def set_parent(self, pnode):
        self.parent = pnode
    

class KinodynamicRRT(object):
    """
    The Kinodynamic Rapidly-exploring Random Tree (RRT) motion planner.
    """

    def __init__(self, pdef):
        """
        args: pdef: The problem definition instance for the problem to be solved.
                    Type: pdef.ProblemDefinition
        """
        self.pdef = pdef
        self.tree = Tree(pdef)
        self.state_sampler = samplers.StateSampler(self.pdef) # state sampler
        self.control_sampler = samplers.ControlSampler(self.pdef) # control sampler

    def solve(self, time_budget):
        """
        The main algorithm of Kinodynamic RRT.
        args:  time_budget: The planning time budget (in seconds).
        returns: is_solved: True or False.
                      plan: The motion plan found by the planner,
                            represented by a sequence of tree nodes.
                            Type: a list of rrt.Node
        """
        ########## TODO ##########
        solved = False
        plan = None        
        
        # Initialize tree with start state
        start_state = self.pdef.get_start_state()
        start_node = Node(start_state)
        start_node.set_parent(None)
        self.tree.add(start_node)

        t_start = time.time()

        while time.time() - t_start < time_budget:
            # Sample a random state vector
            rstateVec = self.state_sampler.sample()

            # Find nearest node in the tree
            nnode = self.tree.nearest(rstateVec)

            # sample controls from nearest node toward sample state
            bctrl, ostate = self.control_sampler.sample_to(nnode, rstateVec, k=1)

            # If no valid propagated state was found, skip this iteration
            if bctrl is None or ostate is None:
                continue

            # Create and add the new node
            new_node = Node(ostate)
            new_node.set_parent(nnode)
            new_node.set_control(bctrl)
            self.tree.add(new_node)

            # Check goal
            if self.pdef.get_goal().is_satisfied(ostate):
                solved = True
                plan = []

                curr = new_node
                while curr is not None:
                    plan.append(curr)
                    curr = curr.get_parent()
                
                plan.reverse()
                
                return solved, plan

        ##########################

        return solved, plan
