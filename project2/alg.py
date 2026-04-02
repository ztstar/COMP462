import numpy as np
import itertools as it
import scipy.spatial
import utils
import time


########## Task 1: Primitive Wrenches ##########

def primitive_wrenches(mesh, grasp, mu=0.2, n_edges=8):
    """
    Find the primitive wrenches for each contact of a grasp.
    args:   mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
           grasp: The indices of the mesh triangles being contacted.
                  Type: list of int
              mu: The friction coefficient of the mesh surface.
                  (default: 0.2)
         n_edges: The number of edges of the friction polyhedral cone.
                  Type: int (default: 8)
    returns:   W: The primitive wrenches.
                  Type: numpy.ndarray of shape (len(grasp) * n_edges, 6)
    """
    ########## TODO ##########
    contact_points = utils.get_centroid_of_triangles(mesh, grasp)

    W = []

    for i, face_id in enumerate(grasp):
        p = contact_points[i]
        p_rel = p - mesh.center_mass
        n = mesh.face_normals[face_id].astype(float)
        n = n / np.linalg.norm(n)

        # Build two orthonormal tangent directions t1, t2
        if abs(n[0]) < 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        else:
            helper = np.array([0.0, 1.0, 0.0])
        
        t1 = np.cross(n, helper)
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        t2 = t2 / np.linalg.norm(t2)

        # Discretize the friction cone into n_edges primitive forces
        for k in range(n_edges):
            theta = 2.0 * np.pi * k / n_edges
            f = n + mu * (np.cos(theta) * t1 + np.sin(theta) * t2)
            tau = np.cross(p_rel, f)
            w = np.hstack((f, tau))
            W.append(w)

    W = np.array(W)

    ##########################
    return W


########## Task 2: Grasp Quality Evaluation ##########

def eval_Q(mesh, grasp, mu=0.2, n_edges=8, lmbd=1.0):
    """
    Evaluate the L1 quality of a grasp.
    args:   mesh: The object mesh model.
                  Type: trimesh.base.Trimesh
           grasp: The indices of the mesh triangles being contacted.
                  Type: list of int
              mu: The friction coefficient of the mesh surface.
                  (default: 0.2)
         n_edges: The number of edges of the friction polyhedral cone.
                  Type: int (default: 8)
            lmbd: The scale of torque magnitude.
                  (default: 1.0)
    returns:   Q: The L1 quality score of the given grasp.
    """
    ########## TODO ##########
    # Get the primitive wrenches from part1.
    W = primitive_wrenches(mesh, grasp, mu=mu, n_edges=n_edges).copy()
    
    # Scale torque. 
    W[:, 3:] *= np.sqrt(lmbd)

    # Convex hull in wrench space
    hull = scipy.spatial.ConvexHull(W)

    Q = np.min(-hull.equations[:, -1])

    ##########################
    return Q


########## Task 3: Stable Grasp Sampling ##########

def sample_stable_grasp(mesh, thresh=0.0):
    """
    Sample a stable grasp such that its L1 quality is larger than a threshold.
    args:     mesh: The object mesh model.
                    Type: trimesh.base.Trimesh
            thresh: The threshold for stable grasp.
                    (default: 0.0)
    returns: grasp: The stable grasp represented by the indices of triangles.
                    Type: list of int
                 Q: The L1 quality score of the sampled grasp, 
                    expected to be larger than thresh.
    """
    ########## TODO ##########
    n_faces = len(mesh.faces)

    while True:
        grasp = np.random.choice(n_faces, size=3, replace=False).tolist()
        Q = eval_Q(mesh, grasp)

        if Q > thresh:
            return grasp, Q

    ##########################
    return grasp, Q


########## Task 4: Grasp Optimization ##########

def find_neighbors(mesh, tr_id, eta=1):
    """
    Find the eta-order neighbor faces (triangles) of tr_id on the mesh model.
    args:       mesh: The object mesh model.
                      Type: trimesh.base.Trimesh
               tr_id: The index of the query face (triangle).
                      Type: int
                 eta: The maximum order of the neighbor faces:
                      Type: int
    returns: nbr_ids: The list of the indices of the neighbor faces.
                      Type: list of int
    """
    ########## TODO ##########
    
    # Build adjacency list from pairs of faces that share a vertex
    if not hasattr(mesh, "_face_nbr_adj"):
        adj = [[] for _ in range(len(mesh.faces))]
        for a, b in mesh.face_neighborhood:
            adj[a].append(b)
            adj[b].append(a)
        mesh._face_nbr_adj = adj
    
    adj = mesh._face_nbr_adj
    
    visited = set([tr_id])
    frontier = set([tr_id])

    for _ in range(eta):
        next_frontier = set()
        for f in frontier:
            for nbr in adj[f]:
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)
        frontier = next_frontier
        if not frontier:
            break
    
    visited.remove(tr_id)
    nbr_ids = list(visited)
    ##########################
    return nbr_ids

def local_optimal(mesh, grasp):
    """
    Find the optimal neighbor grasp of the given grasp.
    args:     mesh: The object mesh model.
                    Type: trimesh.base.Trimesh
             grasp: The indices of the mesh triangles being contacted.
                    Type: list of int
    returns: G_opt: The optimal neighbor grasp with the highest quality.
                    Type: list of int
             Q_max: The L1 quality score of G_opt.
    """
    ########## TODO ##########
    candidate_lists = []
    for tr_id in grasp:
        nbrs = find_neighbors(mesh, tr_id, eta=1)
        candidate_lists.append([tr_id] + nbrs)
    
    G_opt = grasp[:]
    Q_max = eval_Q(mesh, grasp)

    for candidate in it.product(*candidate_lists):
        candidate = list(candidate)

        # Reject grasps that reuse the same face
        if (len(set(candidate)) < len(candidate)):
            continue
            
        Q = eval_Q(mesh, candidate)
        if Q > Q_max:
            Q_max = Q
            G_opt = candidate

    ##########################
    return G_opt, Q_max

def optimize_grasp(mesh, grasp):
    """
    Optimize the given grasp and return the trajectory.
    args:    mesh: The object mesh model.
                   Type: trimesh.base.Trimesh
            grasp: The indices of the mesh triangles being contacted.
                   Type: list of int
    returns: traj: The trajectory of the grasp optimization.
                   Type: list of grasp (each grasp is a list of int)
    """
    traj = []
    ########## TODO ##########
    traj = [grasp[:]]

    current_grasp = grasp[:]
    current_Q = eval_Q(mesh, current_grasp)

    while True:
        next_grasp, next_Q = local_optimal(mesh, current_grasp)

        # stop if no strict improvement
        if next_Q <= current_Q:
            break

        traj.append(next_grasp[:])
        current_grasp = next_grasp
        current_Q = next_Q

    ##########################
    return traj


########## Task 5: Grasp Optimization with Reachability ##########

def optimize_reachable_grasp(mesh, r=0.5):
    """
    Sample a reachable grasp and optimize it.
    args:    mesh: The object mesh model.
                   Type: trimesh.base.Trimesh
                r: The reachability measure. (default: 0.5)
    returns: traj: The trajectory of the grasp optimization.
                   Type: list of grasp (each grasp is a list of int) 
    """
    traj = []
    ########## TODO ##########
    n_faces = len(mesh.faces)

    # Step 1: Sample an initial reachable grasp
    while True:
        grasp = np.random.choice(n_faces, size=3, replace=False).tolist()
        pts = utils.get_centroid_of_triangles(mesh, grasp)
        psi = np.mean(pts, axis=0)
        avg_dist = np.mean(np.linalg.norm(pts - psi, axis=1))
        if avg_dist < r:
            break
    
    traj.append(grasp[:])
    current_grasp = grasp[:]
    current_Q = eval_Q(mesh, current_grasp)

    # Step 2: Optimize while enforcing reachability on neighbors
    while True:
        candidate_lists = []
        for tr_id in current_grasp:
            nbrs = find_neighbors(mesh, tr_id, eta=1)
            candidate_lists.append([tr_id] + nbrs)
        
        best_grasp = current_grasp[:]
        best_Q = current_Q

        for candidate in it.product(*candidate_lists):
            candidate = list(candidate)

            # avoid duplicate contact faces
            if len(set(candidate)) < len(candidate):
                continue
                
            # reachability constraint
            pts = utils.get_centroid_of_triangles(mesh, candidate)
            psi = np.mean(pts, axis=0)
            avg_dist = np.mean(np.linalg.norm(pts - psi, axis=1))
            if avg_dist >= r:
                continue

            Q = eval_Q(mesh, candidate)
            if Q > best_Q:
                best_Q = Q
                best_grasp = candidate
            
        # stop if no improvement
        if best_Q <= current_Q:
            break

        traj.append(best_grasp[:])
        current_grasp = best_grasp
        current_Q = best_Q

    ##########################
    return traj
