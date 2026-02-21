# 3D-Mesh-Generation

This is a 3D reconstruction project that calibrates image datasets into clean, watertight meshes and high-fidelity point clouds.

This is achieved by blending classical multi-view geometry with learned depth estimation to produce reconstructions that are both accurate and robust to noise and occlusions.

Key features

- Camera calibration and structured-light decoding for precise correspondences.
- Geometric triangulation and Poisson-style mesh generation from matched pixels.
- Learned depth estimation (MiDaS) fused into metric point clouds for improved surface coverage.
- Quantitative comparison utilities to evaluate learned depth vs geometric reconstruction.
- Lightweight CLI (analyze_depth.py) for end-to-end evaluation and stats output.

Technology highlights: Python, NumPy, OpenCV, Matplotlib, PyTorch (MiDaS), SciPy.
