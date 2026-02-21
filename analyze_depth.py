#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
from cam_utils import Camera
from reconstruct import reconstruct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scan_dir", help="Directory containing scan images")
    p.add_argument("--calib0", default="calibration_C0.pickle")
    p.add_argument("--calib1", default="calibration_C1.pickle")
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--diff_thr", type=float, default=0.05)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    with open(args.calib0, "rb") as f:
        calib0 = pickle.load(f)
    with open(args.calib1, "rb") as f:
        calib1 = pickle.load(f)

    camL = Camera(np.array([[calib0['fx']], [calib0['fy']]]), np.array([[calib0['cx']], [calib0['cy']]]), calib0['extrinsics'][0]['R'], calib0['extrinsics'][0]['t'])
    camR = Camera(np.array([[calib1['fx']], [calib1['fy']]]), np.array([[calib1['cx']], [calib1['cy']]]), calib1['extrinsics'][0]['R'], calib1['extrinsics'][0]['t'])

    imprefixL = f"{args.scan_dir}/frame_C0_"
    imprefixR = f"{args.scan_dir}/frame_C1_"
    objL = f"{args.scan_dir}/color_C0_01_u.png"
    bkgL = f"{args.scan_dir}/grab_0/color_C0_00_u.png"

    print("Running geometric reconstruction and learned-depth comparison...")
    pts2L, pts2R, pts3, colors = reconstruct(imprefixL, imprefixR, args.threshold, args.diff_thr, camL, camR, objL, bkgL, use_depth_model=True, depth_device=args.device)

    print("Geometric reconstruction produced", pts3.shape[1], "points")


if __name__ == '__main__':
    main()
