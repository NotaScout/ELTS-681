import cv2
import mediapipe as mp
import time
import threading

# MediaPipe Object Detector
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Store the latest detection results
latest_result = None
result_lock = threading.Lock()


# Callback function for object detection
def print_result(result, output_image, timestamp_ms):
    global latest_result

    with result_lock:
        latest_result = result


# Configure the object detector
options = ObjectDetectorOptions(
    base_options=BaseOptions(
        model_asset_path="test_mediapipe/efficientdet_lite0.tflite"
    ),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=10,
    score_threshold=0.5,
    result_callback=print_result
)


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit


# Start object detector
with ObjectDetector.create_from_options(options) as detector:

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read webcam frame.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Submit frame for asynchronous object detection
        timestamp_ms = time.monotonic_ns() // 1_000_000

        detector.detect_async(mp_image, timestamp_ms)

        # Get the latest detection results
        with result_lock:
            result = latest_result

        # Display detected objects
        if result is not None and result.detections:

            for detection in result.detections:

                # Get bounding box
                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                w = bbox.width
                h = bbox.height

                # Keep coordinates within the image
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + w)
                y2 = min(height, y + h)

                # Get object name and confidence
                category = detection.categories[0]

                object_name = (
                    category.display_name
                    or category.category_name
                    or "Unknown"
                )

                confidence = category.score

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Create label
                label = f"{object_name}: {confidence:.2f}"

                # Draw label background
                (text_w, text_h), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                label_y = max(y1, text_h + 10)

                cv2.rectangle(
                    frame,
                    (x1, label_y - text_h - 10),
                    (x1 + text_w, label_y + baseline - 5),
                    (0, 255, 0),
                    -1
                )

                # Display object name
                cv2.putText(
                    frame,
                    label,
                    (x1, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA
                )

                # Print detected object in terminal
                print(
                    f"Detected Object: {object_name}, "
                    f"Confidence: {confidence:.2f}"
                )

        else:
            # No objects detected
            cv2.putText(
                frame,
                "No objects detected",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        # Display webcam window
        cv2.imshow("Object Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Release resources
cap.release()
cv2.destroyAllWindows()