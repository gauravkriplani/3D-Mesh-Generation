import numpy as np
import torch
import cv2
from scipy.spatial import cKDTree


def load_midas(model_type="MiDaS_small", device="cpu"):
    """Load a MiDaS model from torch.hub. Returns (model, transform)."""
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "MiDaS_small":
        transform = transforms.small_transform
    else:
        transform = transforms.default_transform
    model.to(device)
    model.eval()
    return model, transform


def predict_depth_numpy(img, model, transform, device="cpu"):
    """Predict depth for a numpy image (H x W x C) in range [0,1] or [0,255].

    Returns a float32 depth map (H x W). Note: MiDaS outputs relative depth.
    """
    # convert to uint8 PIL-like image expected by transform
    im = img
    if im.dtype != np.uint8:
        im = (np.clip(im, 0.0, 1.0) * 255).astype(np.uint8)

    # transform expects HxWxC in RGB
    if im.shape[2] == 4:
        im = im[:, :, :3]
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.shape[2] == 3 else im

    input_batch = transform(im).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=im.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    depth = prediction.cpu().numpy()
    # normalize to positive
    depth = depth - depth.min()
    depth = depth / (depth.max() + 1e-8)
    return depth.astype(np.float32)


def depth_to_point_cloud(depth, cam, mask=None):
    """Convert a depth map (H x W) in camera-forward units to 3D points in world frame.

    `cam` may be a Camera-like object with attributes `fx, fy, cx, cy, R, t` or a dict with those keys.
    Returns points as (3,N) numpy array.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # read intrinsics
    try:
        fx = float(np.squeeze(cam.fx))
        fy = float(np.squeeze(cam.fy))
        cx = float(np.squeeze(cam.cx))
        cy = float(np.squeeze(cam.cy))
        R = np.array(cam.R)
        t = np.array(cam.t).reshape((3, 1))
    except Exception:
        # assume dict-style
        fx = float(np.squeeze(cam["fx"]))
        fy = float(np.squeeze(cam["fy"]))
        cx = float(np.squeeze(cam["cx"]))
        cy = float(np.squeeze(cam["cy"]))
        R = np.array(cam["R"])
        t = np.array(cam["t"]).reshape((3, 1))

    z = depth.reshape(-1)
    if mask is not None:
        mask_flat = mask.reshape(-1)
    else:
        mask_flat = np.ones_like(z, dtype=bool)

    u_flat = u.reshape(-1)
    v_flat = v.reshape(-1)

    x_cam = (u_flat - cx) * z / fx
    y_cam = (v_flat - cy) * z / fy
    pts_cam = np.vstack((x_cam, y_cam, z))

    pts_cam = pts_cam[:, mask_flat]
    pts_world = (R @ pts_cam) + t
    return pts_world


def svd_alignment(X_src, X_tgt):
    """Return R,t that aligns X_src to X_tgt (3xN arrays) using SVD."""
    m1 = np.mean(X_src, axis=1, keepdims=True)
    m2 = np.mean(X_tgt, axis=1, keepdims=True)
    X1c = X_src - m1
    X2c = X_tgt - m2
    H = X2c @ X1c.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = m2 - R @ m1
    return R, t


def compare_point_clouds(pc_geom, pc_pred):
    """Align pc_pred to pc_geom and compute RMSE and MAE between nearest neighbors.

    Both inputs are (3,N) numpy arrays.
    Prints stats and returns (rmse, mae, median_error).
    """
    # center and align
    R, t = svd_alignment(pc_pred, pc_geom)
    pc_pred_aligned = R @ pc_pred + t

    # NN distances from geom -> pred
    tree = cKDTree(pc_pred_aligned.T)
    dists, _ = tree.query(pc_geom.T, k=1)
    rmse = np.sqrt(np.mean(dists ** 2))
    mae = np.mean(np.abs(dists))
    med = np.median(dists)
    print(f"Point cloud compare: RMSE={rmse:.6f}, MAE={mae:.6f}, MED={med:.6f}, N_geom={pc_geom.shape[1]}, N_pred={pc_pred.shape[1]}")
    return rmse, mae, med

