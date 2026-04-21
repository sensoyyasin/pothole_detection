from ultralytics import YOLO
import cv2
import os

model = YOLO("/Users/yasinsensoy/Desktop/File/Yolov8-fintuned-on-potholes.pt")

frames_dir = "/Users/yasinsensoy/Desktop/File/frames_undistorted"
output_dir = "/Users/yasinsensoy/Desktop/File/detections_clean"
os.makedirs(output_dir, exist_ok=True)

filenames = sorted(
    [f for f in os.listdir(frames_dir) if f.endswith(".jpg")],
    key=lambda x: int(x.replace(".jpg", ""))
)

total = len(filenames)

for i, filename in enumerate(filenames):
    img_path = os.path.join(frames_dir, filename)
    img = cv2.imread(img_path)

    h, w = img.shape[:2]

    roi_y1 = int(h * 0.45)
    roi_y2 = int(h * 0.9)
    roi = img[roi_y1:roi_y2, :]

    results = model(
        roi,
        conf=0.12,
        imgsz=1280,
        verbose=False
    )

    boxes = results[0].boxes

    annotated = img.copy()
    valid_count = 0

    if boxes is not None:

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            y1 += roi_y1
            y2 += roi_y1

            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            if area > 0.25 * w * h:
                continue

            if area < 2000:
                continue

            shrink = 0.6
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bw = int(bw * shrink)
            bh = int(bh * shrink)

            x1 = cx - bw // 2
            x2 = cx + bw // 2
            y1 = cy - bh // 2
            y2 = cy + bh // 2

            valid_count += 1

            label = f"pothole {conf:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(f"[{i+1}/{total}] {filename}: {valid_count} filtered pothole(s)")

    cv2.imwrite(os.path.join(output_dir, filename), annotated)

print("\nDone!")
