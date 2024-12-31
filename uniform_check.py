from ultralytics import YOLO
import cv2
import matplotlib.colors as colors
import json

class UniformCheck:
    def __init__(self):
        self.model = YOLO('./models/best.pt')  # Load your YOLO model
        self.classes = ["uniformTop", "uniformBottom", "id", "logo", "black shoes"]
        self.colors = ['#57cfff', '#60ff7a', '#f2ff4f', '#25a855', '#1787b4', '#98a500', '#e8d636']
        self.expected_classes = set(self.classes)  # Expected uniform components
        self.video = cv2.VideoCapture(0)  # Camera feed

    def get_video_with_anomaly_detection(self):
        """Process video, perform YOLO detection, and detect anomalies."""
        ret, frame = self.video.read()
        if not ret:
            return None, json.dumps({"error": "Video feed not available"})

        results = self.model(frame)

        detected_classes = set()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                class_id = int(box.cls)
                class_name = self.classes[class_id]
                detected_classes.add(class_name)

                # Draw bounding box and label
                hex_color = self.colors[class_id]
                rgb_color = colors.to_rgb(hex_color)
                color_tuple = tuple([int(255 * c) for c in rgb_color])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_tuple, 2)
                text = f"{class_name} {box.conf.item() * 100:.2f}%"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_tuple, 1)

        # Detect anomalies
        missing_items = list(self.expected_classes - detected_classes)
        detection_data = {
            "total_detected": len(detected_classes),
            "detected": list(detected_classes),
            "missing": missing_items,
            "status": "compliant" if not missing_items else "anomaly"
        }

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), json.dumps(detection_data)
