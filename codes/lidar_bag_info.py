from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
import numpy as np

BAG_PATH = "your_bag_path"

typestore = get_typestore(Stores.ROS1_NOETIC)

# Define Livox CustomPoint first
custom_point_msg = """
uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
"""

# Define Livox CustomMsg
custom_msg = """
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
livox_ros_driver2/CustomPoint[] points
"""

# Register types
point_types = get_types_from_msg(custom_point_msg, "livox_ros_driver2/CustomPoint")
msg_types = get_types_from_msg(custom_msg, "livox_ros_driver2/CustomMsg")

typestore.register({**point_types, **msg_types})

with Reader(BAG_PATH) as reader:
    connections = [c for c in reader.connections
                   if c.topic == "/livox/lidar"]

    for conn, timestamp, rawdata in reader.messages(connections=connections):
        msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
        print(f"Point count: {msg.point_num}")
        print(f"First point: x={msg.points[0].x:.3f}, y={msg.points[0].y:.3f}, z={msg.points[0].z:.3f}")
        break
