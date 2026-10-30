import cv2
import mediapipe as mp
import time
import threading

# MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class HandDetector:
    def __init__(self, model_path="test_mediapipe/hand_detector.task"):
        # Store the latest detection results
        self.latest_result = None
        self.result_lock = threading.Lock()

        # Configure and initialize the hand landmarker ONCE to prevent segmentation faults
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,  # Must be LIVE_STREAM for async + callback
            result_callback=self._print_result,
            num_hands=2
        )
        self.landmarker = HandLandmarker.create_from_options(options)


    def _print_result(self, result, output_image, timestamp_ms):
        # MediaPipe live stream callback signature receives (result, output_image, timestamp_ms)
        with self.result_lock:
            self.latest_result = result

    # Simple gesture classifier
    def _detect_gesture(self, landmarks):
        fingers = [
            (8, 6),    # Index finger
            (12, 10),  # Middle finger
            (16, 14),  # Ring finger
            (20, 18)   # Pinky finger
        ]

        extended = []
        for tip, joint in fingers:
            extended.append(
                landmarks[tip].y < landmarks[joint].y
            )

        index, middle, ring, pinky = extended

        if all(extended):
            return "Open Palm"
        elif not any(extended):
            return "Fist"
        elif index and middle and not ring and not pinky:
            return "Peace Sign"
        elif index and not middle and not ring and not pinky:
            return "Pointing"
        else:
            return "None"

    def run_detection(self, frame) -> str:
        """
        Takes a raw OpenCV/NumPy image frame directly 
        (e.g., converted from cv_bridge in your ROS node).
        """
        if frame is None:
            print("Received empty frame.")
            return "None"

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        # Convert BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Submit the frame for asynchronous detection
        timestamp_ms = time.monotonic_ns() // 1_000_000
        self.landmarker.detect_async(mp_image, timestamp_ms)

        # Get the latest detection results thread-safely
        with self.result_lock:
            result = self.latest_result

        # Display detected hands and gestures if available
        if result is not None and result.hand_landmarks:
            height, width = frame.shape

            for hand_landmarks in result.hand_landmarks:
                # Draw hand connections
                for start, end in HAND_CONNECTIONS:
                    x1 = int(hand_landmarks[start].x * width)
                    y1 = int(hand_landmarks[start].y * height)
                    x2 = int(hand_landmarks[end].x * width)
                    y2 = int(hand_landmarks[end].y * height)

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw landmark points
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                # Classify and return the gesture of the first detected hand
                return self._detect_gesture(hand_landmarks)

        return "None"