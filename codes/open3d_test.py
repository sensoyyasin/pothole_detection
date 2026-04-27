'''
(x,y,z)
plane fitting + local comparison + DBSCAN
sensoyyasin - 20 april 2026
'''

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

PLY_FILE = "256_road_clean.ply"

VOXEL_SIZE = 0.03
ROAD_HALF_WIDTH = 2.0
INNER_ROAD_RATIO = 0.60
ALONG_MIN = 0.0
ALONG_MAX = 25.0

PLANE_DIST_THR = 0.02
POTHOLE_DEPTH_THR = -0.02
EXPAND_DEPTH_THR = -0.005
LOCAL_DIFF_THR = 0.01

K = 20
EXPAND_NEIGHBOR_RATIO = 0.30

DBSCAN_EPS = 0.15
DBSCAN_MIN_SAMPLES = 30
MIN_CLUSTER_POINTS = 40

ROAD_POINT_1 = np.array([1.71, 0.413, -0.044], dtype=np.float64)
ROAD_POINT_2 = np.array([21.080, 1.98, 1.403], dtype=np.float64)

pcd = o3d.io.read_point_cloud(PLY_FILE)
print(f"[INFO] Raw points: {len(pcd.points):,}")

pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
print(f"[INFO] After voxel downsample: {len(pcd.points):,}")

pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
print(f"[INFO] After outlier removal: {len(pcd.points):,}")

points = np.asarray(pcd.points)

road_dir = ROAD_POINT_2 - ROAD_POINT_1
road_dir = road_dir / np.linalg.norm(road_dir)

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
roi_along_dist = along_dist[roi_mask]

print(f"[INFO] ROI points: {len(roi_points):,}")

pcd_roi = o3d.geometry.PointCloud()
pcd_roi.points = o3d.utility.Vector3dVector(roi_points)

plane_model, _ = pcd_roi.segment_plane(
    distance_threshold=PLANE_DIST_THR,
    ransac_n=3,
    num_iterations=1000
)

a, b, c, d = plane_model
print(f"[INFO] Plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

normal = np.array([a, b, c], dtype=np.float64)
norm = np.linalg.norm(normal)

signed_dist = (roi_points @ normal + d) / norm
print(f"[DEBUG] signed min/max: {signed_dist.min():.4f}, {signed_dist.max():.4f}")

road_mask = np.abs(signed_dist) < PLANE_DIST_THR
seed_mask = signed_dist < POTHOLE_DEPTH_THR

nbrs = NearestNeighbors(n_neighbors=K).fit(roi_points)
_, indices = nbrs.kneighbors(roi_points)

expanded_mask = seed_mask.copy()
for i in range(len(roi_points)):
    if seed_mask[i]:
        continue
    neigh = indices[i]
    if np.sum(seed_mask[neigh]) > EXPAND_NEIGHBOR_RATIO * K and signed_dist[i] < EXPAND_DEPTH_THR:
        expanded_mask[i] = True

local_mask = np.zeros(len(roi_points), dtype=bool)
for i in range(len(roi_points)):
    neigh = indices[i]
    neigh_mean = np.mean(signed_dist[neigh])
    if signed_dist[i] < neigh_mean - LOCAL_DIFF_THR:
        local_mask[i] = True

candidate_mask = seed_mask | (expanded_mask & local_mask)

inner_mask = roi_perp_dist < (ROAD_HALF_WIDTH * INNER_ROAD_RATIO)
candidate_mask = candidate_mask & inner_mask

candidate_points = roi_points[candidate_mask]
candidate_signed = signed_dist[candidate_mask]
candidate_idx = np.where(candidate_mask)[0]

print(f"[INFO] Candidate pothole points: {len(candidate_points):,}")

colors = np.tile([0.6, 0.6, 0.6], (len(roi_points), 1))
colors[road_mask] = [0.0, 1.0, 0.0]

boxes = []

if len(candidate_points) >= DBSCAN_MIN_SAMPLES:
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(candidate_points)
    labels = clustering.labels_
    unique_labels = [lbl for lbl in np.unique(labels) if lbl != -1]

    valid_count = 0

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

        valid_count += 1

        cluster_global_idx = candidate_idx[cluster_sel]
        colors[cluster_global_idx] = [1.0, 0.0, 0.0]

        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(cluster_pts)
        )
        bbox.color = (1.0, 0.0, 0.0)
        boxes.append(bbox)

        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()

        length = max_bound[0] - min_bound[0]
        width = max_bound[1] - min_bound[1]
        height = max_bound[2] - min_bound[2]

        max_depth = float(np.abs(cluster_signed.min()))
        mean_depth = float(np.abs(cluster_signed.mean()))

        print(
            f"Pothole #{valid_count:02d} | "
            f"Points: {len(cluster_pts)} | "
            f"Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | "
            f"PerpDist: {center_perp:.3f} | "
            f"MaxDepth: {max_depth:.3f} m | "
            f"MeanDepth: {mean_depth:.3f} m | "
            f"BBox: ({length:.3f} x {width:.3f} x {height:.3f}) m"
        )

    print(f"[INFO] Valid potholes: {valid_count}")
else:
    print("[INFO] Not enough candidate points for DBSCAN")
    print("[INFO] Valid potholes: 0")

pcd_roi.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud("final_result.ply", pcd_roi)
print("[INFO] Saved: final_result.ply")

o3d.visualization.draw_geometries([pcd_roi, *boxes])
