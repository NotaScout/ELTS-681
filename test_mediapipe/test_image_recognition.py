import cv2
import mediapipe as mp
import time
import threading

# MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Store the latest detection results
latest_result = None
result_lock = threading.Lock()


# Callback function for hand detection
def print_result(result, output_image, timestamp_ms):
    global latest_result

    with result_lock:
        latest_result = result


# Configure the hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="test_mediapipe/hand_detector.task"
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    num_hands=2
)


# Hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# Simple gesture classifier
def detect_gesture(landmarks):
    # Landmark indices for fingertips and finger joints
    fingers = [
        (8, 6),    # Index finger
        (12, 10),  # Middle finger
        (16, 14),  # Ring finger
        (20, 18)   # Pinky finger
    ]

    # Determine whether each finger is extended
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
        return "Unknown Gesture"


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit


# Start hand landmarker
with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read webcam frame.")
            break

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

        landmarker.detect_async(mp_image, timestamp_ms)

        # Get the latest detection results
        with result_lock:
            result = latest_result

        # Display detected hands and gestures
        if result is not None and result.hand_landmarks:

            height, width, _ = frame.shape

            for hand_landmarks in result.hand_landmarks:

                # Draw hand connections
                for start, end in HAND_CONNECTIONS:
                    x1 = int(hand_landmarks[start].x * width)
                    y1 = int(hand_landmarks[start].y * height)

                    x2 = int(hand_landmarks[end].x * width)
                    y2 = int(hand_landmarks[end].y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                # Draw landmark points
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 0, 255),
                        -1
                    )

                # Classify the gesture
                gesture = detect_gesture(hand_landmarks)

                # Display gesture name
                cv2.putText(
                    frame,
                    "Gesture: " + gesture,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # Print gesture in terminal
                print("Detected Gesture:", gesture)

        else:
            cv2.putText(
                frame,
                "No hand detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        # Display webcam window
        cv2.imshow("Hand Gesture Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Release resources
cap.release()
cv2.destroyAllWindows()