import cv2

from src.hand_tracker import HandTracker


def main():
    tracker = HandTracker()

    if tracker.is_camera_open() == False:
        print("Error: The webcam could not be opened.")
        return

    print("Gesture controller started. Press 'q' in the video window to quit.")

    while True:
        frame = tracker.read_frame()

        if frame is None:
            print("Error: A frame could not be read from the webcam.")
            break

        landmarks = tracker.find_landmarks(frame)

        if landmarks is None:
            status_text = "No hand detected"
        else:
            status_text = "Hand detected: " + str(len(landmarks)) + " landmarks"
            tracker.draw_hand(frame)

        cv2.putText(
            frame,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Gesture Controller", frame) #This line displays the video frame with the hand landmarks and status text in a window titled "Gesture Controller".

        pressed_key = cv2.waitKey(1)
        if pressed_key == ord("q"):
            break

    tracker.release()
    print("Gesture controller stopped.")


if __name__ == "__main__":
    main()