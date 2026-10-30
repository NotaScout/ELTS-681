import cv2
import mediapipe as mp
import time
import threading

# MediaPipe Object Detector Task
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class ObjectDetectorWrapper:
    def __init__(self, model_path="test_mediapipe/efficientdet_lite0.tflite", score_threshold=0.5):
        # Store latest detection results thread-safely
        self.latest_result = None
        self.result_lock = threading.Lock()

        # Configure Object Detector options for live video streaming
        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._print_result,
            score_threshold=score_threshold,
            max_results=5
        )
        self.detector = ObjectDetector.create_from_options(options)

    def _print_result(self, result, output_image, timestamp_ms):
        with self.result_lock:
            self.latest_result = result

    def run_detection(self, frame) -> str:
        """
        Takes a raw OpenCV/NumPy image frame, submits it to the object detector,
        draws the bounding box for the first found object, and returns its label string.
        """
        if frame is None:
            print("Received empty frame.")
            return "None"

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image container
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Submit frame asynchronously with monotonic timestamp in milliseconds
        timestamp_ms = time.monotonic_ns() // 1_000_000
        self.detector.detect_async(mp_image, timestamp_ms)

        # Retrieve the latest detection results
        with self.result_lock:
            result = self.latest_result

        # Process and return the first detected object only
        if result is not None and result.detections:
            height, width= frame.shape
            
            # Grab only the first detection
            detection = result.detections[0]
            if detection.categories:
                category = detection.categories[0]
                category_name = category.category_name

                return category_name

        return "None"