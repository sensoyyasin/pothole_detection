import cv2
import numpy as np
import json
import os

with open("/Users/yasinsensoy/Desktop/File/20260407052256/calibration/calib.json") as f:
    calib = json.load(f)

front = calib["camera_info"]["front"]

K_flat = front["K"]
K = np.array([
    [K_flat[0], K_flat[1], K_flat[2]],
    [K_flat[3], K_flat[4], K_flat[5]],
    [K_flat[6], K_flat[7], K_flat[8]]
], dtype=np.float64)

D = np.array(front["coeff"], dtype=np.float64)

frames_dir = "/Users/yasinsensoy/Desktop/File/frames"
output_dir = "/Users/yasinsensoy/Desktop/File/frames_undistorted"
os.makedirs(output_dir, exist_ok=True)

filenames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

# Build maps once — same for all frames
sample = cv2.imread(os.path.join(frames_dir, filenames[0]))
h, w = sample.shape[:2]

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=0.0
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
)

# Crop params — remove blurry edges
cx1, cx2 = int(w * 0.20), int(w * 0.80)
cy1, cy2 = int(h * 0.10), int(h * 0.90)

total = len(filenames)
for i, filename in enumerate(filenames):
    img = cv2.imread(os.path.join(frames_dir, filename))

    # Undistort
    undistorted = cv2.remap(img, map1, map2,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)

    # Crop center
    cropped = undistorted[cy1:cy2, cx1:cx2]

    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, cropped)
    print(f"[{i+1}/{total}] {filename}")

print(f"\nDone! {total} frames saved to {output_dir}")
