import numpy as np
import matplotlib.pyplot as plt
from cam_utils import Camera,triangulate
import pickle
from collections import defaultdict
from scipy.spatial import Delaunay
from collections import defaultdict
from meshutils import writeply
from reconstruct import reconstruct


def generate_mesh(scan_dir, output_pickle, threshold=0.05, diff_thr=0.05, bbox=None, trithresh=None, smoothing=0):
    '''
    Generate a 3D mesh from a set of images images captured by two cameras and save it in a pickle file.

    Parameters:
    scan_dir : str
        Directory containing the images for the scan.
    output_pickle : str
        Output file path for the generated mesh in pickle format.
    threshold : float
        Threshold to determine if a bit is decodeable.
    diff_thr : float
        Threshold for the difference between object and background images to create a mask.
    bbox : tuple or None
        Bounding box to prune points outside of the specified bounds. Should be a tuple of the form (x_min, x_max, y_min, y_max, z_min, z_max).
    trithresh : float or None
        Threshold for triangle edge lengths to prune triangles that are too long. If None, no pruning is done.
    smoothing : int
        Number of smoothing iterations to apply to the mesh. Default is 0.

    '''
    # Image prefixes for left and right cameras
    imprefixL = f'{scan_dir}/frame_C0_'
    imprefixR = f'{scan_dir}/frame_C1_'

    # Background image (from grab_0)
    bkgL = 'david/grab_0/color_C0_00_u.png'

    # Object image from left camera
    objL = f'{scan_dir}/color_C0_01_u.png'

    # Only left image is used becaue we can just use the points that align with the left image

    # Load calibration pickle files for left and right cameras
    with open('calibration_C0.pickle', 'rb') as f: calib0 = pickle.load(f)
    with open('calibration_C1.pickle', 'rb') as f: calib1 = pickle.load(f)

    # Camera objects for left and right cameras
    camL = Camera(np.array([[calib0['fx']], [calib0['fy']]]), np.array([[calib0['cx']], [calib0['cy']]]), calib0['extrinsics'][0]['R'], calib0['extrinsics'][0]['t'])
    camR = Camera(np.array([[calib1['fx']], [calib1['fy']]]), np.array([[calib1['cx']], [calib1['cy']]]), calib1['extrinsics'][0]['R'], calib1['extrinsics'][0]['t'])

    # Call reconstruct function to get 2D and 3D points
    pts2L, pts2R, pts3, colors = reconstruct(imprefixL, imprefixR, threshold, diff_thr, camL, camR, objL, bkgL)

    # Print bounds before pruning
    print("Bounds before bounding box pruning:")
    print("X:", pts3[0].min(), pts3[0].max())
    print("Y:", pts3[1].min(), pts3[1].max())
    print("Z:", pts3[2].min(), pts3[2].max())

    # Bounding box pruning to get rid of points outside of bounds
    if bbox is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        mask = (
            (pts3[0] >= x_min) & (pts3[0] <= x_max) &
            (pts3[1] >= y_min) & (pts3[1] <= y_max) &
            (pts3[2] >= z_min) & (pts3[2] <= z_max)
        )
        pts3 = pts3[:, mask]
        pts2L = pts2L[:, mask]
        pts2R = pts2R[:, mask]
        colors = colors[:, mask]

    # Triangle pruning
    tri = Delaunay(pts2L.T)
    vertices = pts3.T
    faces = tri.simplices

    if trithresh is not None:
        # Corners of the triangle
        c1 = vertices[faces[:, 0]]
        c2 = vertices[faces[:, 1]]
        c3 = vertices[faces[:, 2]]

        # Calculate the lengths of the edges using corners
        e1 = np.linalg.norm(c1 - c2, axis=1)
        e2 = np.linalg.norm(c2 - c3, axis=1)
        e3 = np.linalg.norm(c3 - c1, axis=1)
        longest = np.maximum.reduce([e1, e2, e3]) #longest edge
        faces = faces[longest < trithresh]

        # Remove all unreferenced points
        used = np.unique(faces)
        idx_map = -np.ones(vertices.shape[0], dtype=int)
        idx_map[used] = np.arange(len(used))
        vertices = vertices[used]
        colors = colors[:, used]
        faces = idx_map[faces]
        pts3 = vertices.T

    # Mesh smoothing
    if smoothing > 0:
        # Store neighbors for each vertex
        neighbors = {}

        # Fill neighbors from faces
        for tri in faces:
            a, b, c = tri
            for u, v in [(a, b), (b, c), (c, a)]:
                if u not in neighbors:
                    neighbors[u] = set()
                if v not in neighbors:
                    neighbors[v] = set()
                neighbors[u].add(v)
                neighbors[v].add(u)
        
        # Smooth the mesh by averaging the positions of each vertex with its neighbors
        for i in range(smoothing):
            new_vertices = vertices.copy()
            for j in range(len(vertices)):
                if j in neighbors:
                    nbrs = list(neighbors[j])
                else: 
                    nbrs = []

                if nbrs:
                    new_vertices[j] = np.mean(vertices[nbrs], axis=0)
            vertices = new_vertices
        
        pts3 = vertices.T

    # Save the smooth mesh to a pickle file
    with open(output_pickle, 'wb') as f:
        pickle.dump({'pts3': pts3, 'pts2L': pts2L, 'pts2R': pts2R, 'colors': colors,'faces': faces}, f)


def svd_alignment(X1, X2):
    '''
    Aligns two sets of points from different views using SVD
    Parameters:
    X1 : numpy.array
        Points from the first view.
    X2 : numpy.array
        Points from the second view.
    Returns:
    R : Rotation matrix aligning X1 to X2.
    t : Translation vector aligning X1 to X2.
    '''

    m1 = np.mean(X1, axis=1, keepdims=True)
    m2 = np.mean(X2, axis=1, keepdims=True)
    X1c = X1 - m1
    X2c = X2 - m2
    H = X2c @ X1c.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = m1 - R @ m2
    return R, t