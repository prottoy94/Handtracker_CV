import csv
import os

import cv2

import config
from src.hand_tracker import HandTracker
from src.features import extract_features


# Replace this with the video address shown by your phone camera app.
PHONE_CAMERA_URL = "http://192.168.68.104:8080/video"


def save_sample(file_path, gesture_label, feature_list):
    file_has_data = os.path.exists(file_path) and os.path.getsize(file_path) > 0

    data_file = open(file_path, "a", newline="")
    csv_writer = csv.writer(data_file)

    if file_has_data == False:
        header = ["label"]
        for number in range(1, 43):
            header.append("feature_" + str(number))
        csv_writer.writerow(header)

    row = [gesture_label]
    for feature in feature_list:
        row.append(feature)
    csv_writer.writerow(row)

    data_file.close()


def draw_text(frame, text, x, y, color):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def main():
    gesture_label = input("Enter the name of the gesture to record: ").strip().lower()

    if gesture_label == "":
        print("Error: A gesture name is required.")
        return

    project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(project_folder, "data")
    csv_file_path = os.path.join(data_folder, "gestures.csv")

    if os.path.exists(data_folder) == False:
        os.makedirs(data_folder)

    tracker = HandTracker(
        camera_index=PHONE_CAMERA_URL,
        max_hands=config.MAX_HANDS,
        detection_confidence=config.DETECTION_CONFIDENCE,
        tracking_confidence=config.TRACKING_CONFIDENCE,
    )

    if tracker.is_camera_open() == False:
        print("Error: The phone camera stream could not be opened.")
        print("Check the phone IP address, URL, and Wi-Fi connection.")
        tracker.release()
        return

    print("Recording gesture: " + gesture_label)
    print("Show your hand and press 's' to save a sample.")
    print("Press 'q' to stop recording.")

    saved_sample_count = 0

    while True:
        frame = tracker.read_frame()

        if frame is None:
            print("Error: A frame could not be read from the phone camera.")
            print("Check the phone IP address, URL, and Wi-Fi connection.")
            break

        landmarks = tracker.find_landmarks(frame)

        if landmarks is None:
            draw_text(frame, "No hand detected", 10, 30, (0, 0, 255))
        else:
            tracker.draw_hand(frame)
            draw_text(frame, "Press s to save this sample", 10, 30, (0, 255, 0))
            draw_text(frame, "Saved samples: " + str(saved_sample_count), 10, 60, (0, 255, 255))

        cv2.imshow("Collect Gesture Data From Phone", frame)

        pressed_key = cv2.waitKey(1)

        if pressed_key == ord("q"):
            break

        if pressed_key == ord("s") and landmarks is not None:
            feature_list = extract_features(landmarks)
            save_sample(csv_file_path, gesture_label, feature_list)
            saved_sample_count = saved_sample_count + 1
            print("Saved sample " + str(saved_sample_count))

    tracker.release()
    print("Data collection stopped.")
    print("Total samples saved: " + str(saved_sample_count))


if __name__ == "__main__":
    main()
