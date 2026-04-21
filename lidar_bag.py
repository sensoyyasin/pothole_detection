from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import numpy as np
import os

BAG_PATH = "/Users/yasinsensoy/Desktop/File/20260407052256/LIDAR_20260407052256_0.bag"
IMAGE_BAG = "/Users/yasinsensoy/Desktop/File/20260407052256/IMAGE_20260407052256_0.bag"

typestore = get_typestore(Stores.ROS1_NOETIC)

custom_point_msg = """
uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
"""

custom_msg = """
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
livox_ros_driver2/CustomPoint[] points
"""

point_types = get_types_from_msg(custom_point_msg, "livox_ros_driver2/CustomPoint")
msg_types = get_types_from_msg(custom_msg, "livox_ros_driver2/CustomMsg")
typestore.register({**point_types, **msg_types})

# --- Read all LiDAR scans with timestamps ---
print("Reading LiDAR scans...")
lidar_scans = []  # list of (timestamp_sec, points_array)

with Reader(BAG_PATH) as reader:
    connections = [c for c in reader.connections
                   if c.topic == "/livox/lidar"]

    for conn, timestamp, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
        ts = timestamp / 1e9

        pts = np.array([[p.x, p.y, p.z, p.reflectivity]
                        for p in msg.points], dtype=np.float32)

        lidar_scans.append((ts, pts))

print(f"Total LiDAR scans: {len(lidar_scans)}")
print(f"Time range: {lidar_scans[0][0]:.3f}s → {lidar_scans[-1][0]:.3f}s")

# --- Read all image timestamps ---
print("\nReading image timestamps...")
image_timestamps = []  # list of (timestamp_sec, filename)

frames_dir = "/Users/yasinsensoy/Desktop/File/frames_undistorted"
filenames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

with Reader(IMAGE_BAG) as reader:
    connections = [c for c in reader.connections
                   if c.topic == "/camera_front/image_raw"]

    for frame_idx, (conn, timestamp, rawdata) in enumerate(
        reader.messages(connections=connections)
    ):
        ts = timestamp / 1e9
        filename = filenames[frame_idx]
        image_timestamps.append((ts, filename))

print(f"Total image frames: {len(image_timestamps)}")
print(f"Time range: {image_timestamps[0][0]:.3f}s → {image_timestamps[-1][0]:.3f}s")

# --- Match each image to closest LiDAR scan ---
print("\nMatching image frames to LiDAR scans...")
lidar_times = np.array([s[0] for s in lidar_scans])

matched = []
for img_ts, img_file in image_timestamps:
    idx = np.argmin(np.abs(lidar_times - img_ts))
    diff = abs(lidar_times[idx] - img_ts)

    if diff < 0.5:  # within 500ms
        matched.append({
            "image": img_file,
            "image_ts": img_ts,
            "lidar_ts": lidar_times[idx],
            "lidar_pts": lidar_scans[idx][1],
            "time_diff": diff
        })
        print(f"  {img_file} ↔ LiDAR scan {idx} (diff={diff*1000:.1f}ms, pts={len(lidar_scans[idx][1])})")
    else:
        print(f"  {img_file}: no matching LiDAR scan (min diff={diff*1000:.1f}ms)")

print(f"\nTotal matched pairs: {len(matched)}/{len(image_timestamps)}")
