from rosbags.rosbag1 import Reader

BAG_PATH = "your_bag_path"

with Reader(BAG_PATH) as reader:
    connections = [c for c in reader.connections
                   if c.topic == "/camera_front/image_raw"]
    
    total = sum(c.msgcount for c in connections)
    duration = reader.duration / 1e9

    print(f"Total frames : {total}")
    print(f"Duration     : {duration:.1f} seconds")
    print(f"FPS          : {total / duration:.1f}")
