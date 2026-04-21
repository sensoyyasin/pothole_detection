# Pothole detection with Lidar + Camera
YOLOv8 and point cloud fusion for enhanced road pothole detection and quantification

Manual:
image_info.py -> Total frames : 125, Duration     : 37.2 seconds, FPS          : 3.4

calibration.py -> Fisheye to normal image 

image_coder.py -> convert to the normal image (! problematic, we're using blur it should be changed)

lidar_bag_info.py -> info function Point count , First point

lidar_bag.py -> Reading LiDAR scans... through lidar_bag file. and convert 

open3d_test.py -> Detect Potholes (Red) through the 256_road_clean.ply - plane fitting + local comparison + DBSCAN

train.py -> uses pretrained model - cazzz307/Pothole-Finetuned-YoloV8- (https://huggingface.co/cazzz307/Pothole-Finetuned-YoloV8/blob/main/Yolov8-fintuned-on-potholes.pt)
