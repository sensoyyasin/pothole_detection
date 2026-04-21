# IMAGE_20260407052256_0.bag

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
import cv2
import numpy as np
import os

typestore = get_typestore(Stores.ROS1_NOETIC)

BAG_PATH = "/Users/yasinsensoy/Desktop/File/20260407052256/IMAGE_20260407052256_0.bag"
output_dir = "/Users/yasinsensoy/Desktop/File/frames"
os.makedirs(output_dir, exist_ok=True)

with Reader(BAG_PATH) as reader:
    connections = [c for c in reader.connections
                   if c.topic == "/camera_front/image_raw"]

    for frame_idx, (conn, timestamp, rawdata) in enumerate(
        reader.messages(connections=connections)
    ):
        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)

        # CompressedImage: decode raw bytes directly into an OpenCV image
        img_data = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Frame {frame_idx + 1}: decode failed, skipping")
            continue

        # Name files 1, 2, 3, 4 ... in chronological order
        filename = f"{output_dir}/{frame_idx + 1}.jpg"
        cv2.imwrite(filename, img)

        print(f"[{frame_idx + 1}/125] → {filename}")

print("Done!")
