import sys
import os
import argparse
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply_file")
    parser.add_argument("--no-voxel", action="store_true")
    parser.add_argument("--voxel-size", type=float, default=0.03)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


args = parse_args()

PLY_FILE = args.ply_file
VOXEL_SIZE = args.voxel_size
USE_VOXEL = not args.no_voxel

ROAD_HALF_WIDTH = 2.0
INNER_ROAD_RATIO = 0.70

ALONG_MIN = 1.0
ALONG_MAX = 25.0

PLANE_DIST_THR = 0.02
POTHOLE_DEPTH_THR = -0.015
EXPAND_DEPTH_THR = -0.005
LOCAL_DIFF_THR = 0.007

K = 20
EXPAND_NEIGHBOR_RATIO = 0.40

DBSCAN_EPS = 0.12
DBSCAN_MIN_SAMPLES = 25
MIN_CLUSTER_POINTS = 25

MIN_DEPTH = None
MAX_WIDTH = None

ROAD_POINTS = {
    "552.ply": {
        "road_points": (
            np.array([1.71, 0.413, -0.044], dtype=np.float64),
            np.array([21.080, 1.98, 1.403], dtype=np.float64),
        ),
        "along_min": 1.0,
        "along_max": 25.0,
        "inner_road_ratio": 0.70,
    },
    "559.ply": {
        "road_points": (
            np.array([-0.414754, -0.191220, -0.942218], dtype=np.float64),
            np.array([4.387130, 0.654194, -0.673998], dtype=np.float64),
        ),
        "along_min": -0.2,
        "along_max": 5.2,
        "inner_road_ratio": 0.85,

        "pothole_depth_thr": -0.012,
        "expand_depth_thr": -0.004,

        "dbscan_eps": 0.14,
        "dbscan_min_samples": 20,
        "min_cluster_points": 60,

        "min_depth": 0.010,
        "max_width": 0.5,
    },
    "847.ply": {
        "road_points": (
            np.array([4.661179, 0.506775, -0.960172], dtype=np.float64),
            np.array([-0.401863, 0.419310, -1.135084], dtype=np.float64),
        ),
        "along_min": -0.3,
        "along_max": 5.4,
        "inner_road_ratio": 0.85,
        "pothole_depth_thr": -0.035,
        "expand_depth_thr": -0.015,
        "dbscan_eps": 0.16,
        "dbscan_min_samples": 25,
        "min_cluster_points": 80,
    },
}


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero vector cannot be normalized.")
    return v / n


base_name = os.path.basename(PLY_FILE)

if base_name not in ROAD_POINTS:
    print(f"Error: No configuration found for {base_name}")
    print(f"Available files: {', '.join(ROAD_POINTS.keys())}")
    sys.exit(1)

cfg = ROAD_POINTS[base_name]

ROAD_POINT_1, ROAD_POINT_2 = cfg["road_points"]

ALONG_MIN = cfg.get("along_min", ALONG_MIN)
ALONG_MAX = cfg.get("along_max", ALONG_MAX)
INNER_ROAD_RATIO = cfg.get("inner_road_ratio", INNER_ROAD_RATIO)

PLANE_DIST_THR = cfg.get("plane_dist_thr", PLANE_DIST_THR)
POTHOLE_DEPTH_THR = cfg.get("pothole_depth_thr", POTHOLE_DEPTH_THR)
EXPAND_DEPTH_THR = cfg.get("expand_depth_thr", EXPAND_DEPTH_THR)
LOCAL_DIFF_THR = cfg.get("local_diff_thr", LOCAL_DIFF_THR)

DBSCAN_EPS = cfg.get("dbscan_eps", DBSCAN_EPS)
DBSCAN_MIN_SAMPLES = cfg.get("dbscan_min_samples", DBSCAN_MIN_SAMPLES)
MIN_CLUSTER_POINTS = cfg.get("min_cluster_points", MIN_CLUSTER_POINTS)

MIN_DEPTH = cfg.get("min_depth", MIN_DEPTH)
MAX_WIDTH = cfg.get("max_width", MAX_WIDTH)

pcd = o3d.io.read_point_cloud(PLY_FILE)

if USE_VOXEL:
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

points = np.asarray(pcd.points)
points = points[np.isfinite(points).all(axis=1)]

road_dir = normalize(ROAD_POINT_2 - ROAD_POINT_1)

v = points - ROAD_POINT_1
along_dist = v @ road_dir
proj = ROAD_POINT_1 + np.outer(along_dist, road_dir)
perp_dist = np.linalg.norm(points - proj, axis=1)

roi_mask = (
    (perp_dist < ROAD_HALF_WIDTH) &
    (along_dist > ALONG_MIN) &
    (along_dist < ALONG_MAX)
)

roi_points = points[roi_mask]
roi_perp_dist = perp_dist[roi_mask]

if len(roi_points) < K:
    print("Error: Not enough points in region of interest.")
    sys.exit(1)

pcd_roi = o3d.geometry.PointCloud()
pcd_roi.points = o3d.utility.Vector3dVector(roi_points)

plane_model, _ = pcd_roi.segment_plane(
    distance_threshold=PLANE_DIST_THR,
    ransac_n=3,
    num_iterations=1000
)

a, b, c, d = plane_model

normal = np.array([a, b, c], dtype=np.float64)
norm = np.linalg.norm(normal)

if norm < 1e-12:
    print("Error: Invalid plane normal.")
    sys.exit(1)

normal_unit = normal / norm
signed_dist = (roi_points @ normal + d) / norm

road_mask = np.abs(signed_dist) < PLANE_DIST_THR
seed_mask = signed_dist < POTHOLE_DEPTH_THR

nbrs = NearestNeighbors(n_neighbors=K).fit(roi_points)
_, indices = nbrs.kneighbors(roi_points)

local_mask = np.zeros(len(roi_points), dtype=bool)

for i in range(len(roi_points)):
    neigh_mean = np.mean(signed_dist[indices[i]])
    if signed_dist[i] < neigh_mean - LOCAL_DIFF_THR:
        local_mask[i] = True

expanded_mask = seed_mask.copy()

for i in range(len(roi_points)):
    if seed_mask[i]:
        continue

    neigh = indices[i]
    seed_neighbor_ratio = np.sum(seed_mask[neigh]) / K

    if (
        seed_neighbor_ratio > EXPAND_NEIGHBOR_RATIO
        and signed_dist[i] < EXPAND_DEPTH_THR
        and local_mask[i]
    ):
        expanded_mask[i] = True

candidate_mask = seed_mask | expanded_mask

inner_mask = roi_perp_dist < (ROAD_HALF_WIDTH * INNER_ROAD_RATIO)
candidate_mask = candidate_mask & inner_mask

candidate_points = roi_points[candidate_mask]
candidate_signed = signed_dist[candidate_mask]
candidate_idx = np.where(candidate_mask)[0]

colors = np.tile([0.6, 0.6, 0.6], (len(roi_points), 1))
colors[road_mask] = [0.0, 1.0, 0.0]

valid_count = 0

if len(candidate_points) >= DBSCAN_MIN_SAMPLES:
    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES
    ).fit(candidate_points)

    labels = clustering.labels_
    unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]

    for lbl in unique_labels:
        cluster_sel = labels == lbl
        cluster_pts = candidate_points[cluster_sel]
        cluster_signed = candidate_signed[cluster_sel]

        if len(cluster_pts) < MIN_CLUSTER_POINTS:
            continue

        center = cluster_pts.mean(axis=0)

        center_v = center - ROAD_POINT_1
        center_along = center_v @ road_dir
        center_proj = ROAD_POINT_1 + center_along * road_dir
        center_perp = np.linalg.norm(center - center_proj)

        if center_perp >= (ROAD_HALF_WIDTH * INNER_ROAD_RATIO):
            continue

        depths = np.abs(cluster_signed)

        cluster_plane_pts = cluster_pts - np.outer(cluster_signed, normal_unit)
        cluster_centered = cluster_plane_pts - cluster_plane_pts.mean(axis=0)

        cov = np.cov(cluster_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        side_axis = eigvecs[:, 1]
        side_proj = cluster_centered @ side_axis

        width = float(np.percentile(side_proj, 95) - np.percentile(side_proj, 5))
        depth = float(np.percentile(depths, 95))

        if MIN_DEPTH is not None and depth < MIN_DEPTH:
            continue

        if MAX_WIDTH is not None and width > MAX_WIDTH:
            continue

        valid_count += 1

        cluster_global_idx = candidate_idx[cluster_sel]
        colors[cluster_global_idx] = [1.0, 0.0, 0.0]

        print(
            f"Pothole #{valid_count:02d} | "
            f"Points: {len(cluster_pts)} | "
            f"Depth: {depth:.3f} m | "
            f"Width: {width:.3f} m"
        )

    print(f"\nTotal potholes detected: {valid_count}")

else:
    print("No potholes detected.")

pcd_roi.colors = o3d.utility.Vector3dVector(colors)

if args.output is None:
    root, ext = os.path.splitext(PLY_FILE)
    out_name = root + "_detected.ply"
else:
    out_name = args.output

o3d.io.write_point_cloud(out_name, pcd_roi)

o3d.visualization.draw_geometries([pcd_roi])
