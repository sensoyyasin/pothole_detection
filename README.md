# Pothole detection with LiDAR + Camera

YOLOv8 and point cloud fusion for road pothole detection and depth estimation.

Dataset → `20260407052256`

---

## Manual

- **image_info.py**  
  Total frames: 125  
  Duration: 37.2 seconds  
  FPS: 3.4  

- **calibration.py**  
  Converts fisheye images to normal images  

- **image_coder.py**  
  Converts images to normal format  
  (currently uses blur → needs improvement)

- **lidar_bag_info.py**  
  Prints basic LiDAR info (point count, first point, etc.)

- **lidar_bag.py**  
  Reads LiDAR `.bag` file and converts it into usable data  

- **open3d_test.py**  
  Detects potholes from `256_road_clean.ply`  
  Uses plane fitting + local comparison + DBSCAN  
  Outputs pothole regions as red points  

- **train.py**  
  Runs YOLOv8 with a pretrained pothole model  
  Model: https://huggingface.co/cazzz307/Pothole-Finetuned-YoloV8/blob/main/Yolov8-fintuned-on-potholes.pt  

---

## Idea

LiDAR → finds geometric pothole candidates (red points)  
YOLO → finds potholes in image  
Final → overlap of both = more reliable detection


Final_result from cloud compare : <img width="1259" height="539" alt="Ekran Resmi 2026-04-20 18 18 40" src="https://github.com/user-attachments/assets/c6c4632d-2597-4feb-bb65-d44192ad93b6" />

[INFO] Raw points: 2,929,160
[INFO] After voxel downsample: 334,130
[INFO] After outlier removal: 330,950
[INFO] ROI points: 183,629
[INFO] Plane: -0.0729x + 0.0285y + 0.9969z + 0.6716 = 0
[DEBUG] signed min/max: -0.1060, 0.6323
[INFO] Candidate pothole points: 1,493
Pothole #01 | Points: 100 | Center: (14.789, 0.891, 0.361) | PerpDist: 0.810 | MaxDepth: 0.037 m | MeanDepth: 0.022 m | BBox: (0.616 x 0.336 x 0.037) m
Pothole #02 | Points: 84 | Center: (10.979, 1.182, 0.066) | PerpDist: 0.581 | MaxDepth: 0.038 m | MeanDepth: 0.029 m | BBox: (0.332 x 0.196 x 0.029) m
Pothole #03 | Points: 155 | Center: (3.480, 0.610, -0.467) | PerpDist: 0.556 | MaxDepth: 0.053 m | MeanDepth: 0.030 m | BBox: (0.412 x 0.381 x 0.037) m
Pothole #04 | Points: 673 | Center: (4.838, 0.233, -0.365) | PerpDist: 0.700 | MaxDepth: 0.106 m | MeanDepth: 0.039 m | BBox: (1.135 x 0.653 x 0.163) m
Pothole #05 | Points: 133 | Center: (23.533, 1.727, 0.972) | PerpDist: 0.758 | MaxDepth: 0.040 m | MeanDepth: 0.026 m | BBox: (0.303 x 0.379 x 0.027) m
[INFO] Valid potholes: 5
[INFO] Saved: final_result.ply

Frames : <img width="726" height="549" alt="Ekran Resmi 2026-04-20 18 13 07" src="https://github.com/user-attachments/assets/8bd49da8-7a69-475c-b911-dd8cdf5ac725" />

Frames Undistorted : <img width="642" height="644" alt="Ekran Resmi 2026-04-20 18 13 35" src="https://github.com/user-attachments/assets/70ba05ba-ea96-4f11-8f78-bc298be22a34" />

detections(with fisheye) : <img width="729" height="547" alt="Ekran Resmi 2026-04-20 18 14 30" src="https://github.com/user-attachments/assets/afad8645-ea49-46b8-b596-32d2d56545b3" />

detections_clean(without fisheye)  : <img width="654" height="650" alt="Ekran Resmi 2026-04-20 18 14 12" src="https://github.com/user-attachments/assets/57a8f835-cdda-4b65-a7be-809a6c2d5063" />
