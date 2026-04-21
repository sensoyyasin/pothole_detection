# Pothole detection with Lidar + Camera
YOLOv8 and point cloud fusion for enhanced road pothole detection and quantification

Dataset -> 20260407052256

Manual:
image_info.py -> Total frames : 125, Duration     : 37.2 seconds, FPS          : 3.4

calibration.py -> Fisheye to normal image 

image_coder.py -> convert to the normal image (! problematic, we're using blur it should be changed)

lidar_bag_info.py -> info function Point count , First point

lidar_bag.py -> Reading LiDAR scans... through lidar_bag file. and convert 

open3d_test.py -> Detect Potholes (Red) through the 256_road_clean.ply - plane fitting + local comparison + DBSCAN

train.py -> uses pretrained model - cazzz307/Pothole-Finetuned-YoloV8- (https://huggingface.co/cazzz307/Pothole-Finetuned-YoloV8/blob/main/Yolov8-fintuned-on-potholes.pt)


Frames : <img width="726" height="549" alt="Ekran Resmi 2026-04-20 18 13 07" src="https://github.com/user-attachments/assets/8bd49da8-7a69-475c-b911-dd8cdf5ac725" />

Frames Undistorted : <img width="642" height="644" alt="Ekran Resmi 2026-04-20 18 13 35" src="https://github.com/user-attachments/assets/70ba05ba-ea96-4f11-8f78-bc298be22a34" />

detections(with fisheye) : <img width="729" height="547" alt="Ekran Resmi 2026-04-20 18 14 30" src="https://github.com/user-attachments/assets/afad8645-ea49-46b8-b596-32d2d56545b3" />

detections_clean(without fisheye)  : <img width="654" height="650" alt="Ekran Resmi 2026-04-20 18 14 12" src="https://github.com/user-attachments/assets/57a8f835-cdda-4b65-a7be-809a6c2d5063" />
