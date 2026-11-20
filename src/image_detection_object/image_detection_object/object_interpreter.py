import cv2
import mediapipe as mp
import time

# MediaPipe Object Detector Task
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Objects that the detector is allowed to return
OBJECTS_DETECTION_LIST = [
    "cell phone",
    "laptop",
    "tv",
    "backpack"
]


class ObjectDetectorWrapper:

    def __init__(
        self,
        model_path="test_mediapipe/efficientdet_lite0.tflite",
        score_threshold=0.65
    ):

        # Configure MediaPipe Object Detector
        options = ObjectDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=model_path
            ),
            running_mode=VisionRunningMode.VIDEO,
            score_threshold=score_threshold,
            max_results=10
        )

        self.detector = ObjectDetector.create_from_options(options)

        # Track timestamps for VIDEO mode
        self.last_timestamp = 0

        print("Object detector initialized.")

    def run_detection(self, frame) -> tuple:

        """
        Takes an OpenCV BGR image.
        Detects objects using MediaPipe.
        Filters unwanted objects.
        Draws bounding boxes.
        Returns the highest-confidence matching object and annotated frame.
        """

        if frame is None or frame.size == 0:
            print("Received empty frame.")
            return "None", frame

        # Mirror webcam image
        frame = cv2.flip(frame, 1)



        # Get image dimensions
        height, width = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        # Preserve original BGR frame for OpenCV drawing
        annotated_frame = rgb_frame.copy()

        # Ensure image is contiguous
        rgb_frame = rgb_frame.copy()

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Generate monotonically increasing timestamp
        timestamp_ms = max(
            time.monotonic_ns() // 1_000_000,
            self.last_timestamp + 1
        )

        self.last_timestamp = timestamp_ms

        # Run synchronous object detection
        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # No detections
        if not result.detections:
            return "None", annotated_frame

        # Store matching detections
        detected_objects = []

        # Process each detection
        for detection in result.detections:

            if not detection.categories:
                continue

            category = detection.categories[0]

            # Get category name
            category_name = category.category_name

            if not category_name:
                continue

            category_name = category_name.lower()

            # Ignore objects outside our allowed list
            if category_name not in OBJECTS_DETECTION_LIST:
                continue

            # Get confidence
            confidence = category.score

            # Get bounding box
            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            w = bbox.width
            h = bbox.height

            # Clamp coordinates to image boundaries
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            x2 = max(0, min(width - 1, x + w))
            y2 = max(0, min(height - 1, y + h))

            # Ignore invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Store valid detection
            detected_objects.append(
                (category_name, confidence, x1, y1, x2, y2)
            )

        # No matching objects found
        if not detected_objects:
            return "None", annotated_frame

        # Sort by confidence (highest first)
        detected_objects.sort(
            key=lambda detection: detection[1],
            reverse=True
        )

        # Draw all matching detections
        for detection in detected_objects:

            name, confidence, x1, y1, x2, y2 = detection

            # Bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label
            label = f"{name}: {confidence:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            label_y = max(y1, text_h + 10)

            # Label background
            cv2.rectangle(
                annotated_frame,
                (x1, label_y - text_h - 10),
                (x1 + text_w, label_y + baseline - 5),
                (0, 255, 0),
                -1
            )

            # Label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        # Return highest-confidence matching detection
        best_detection = detected_objects[0]

        detected_name = best_detection[0]
        detected_confidence = best_detection[1]

        print(
            f"Object detected: {detected_name} "
            f"({detected_confidence:.2f})"
        )

        return detected_name, annotated_frame

    def close(self):
        self.detector.close()