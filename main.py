import cv2

import config
from src.hand_tracker import HandTracker
from src.features import extract_features
from src.classifier import GestureClassifier
from src.smoothing import Debouncer
from src.actions import ActionRunner


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
    tracker = HandTracker(
        camera_index=config.CAMERA_INDEX,
        max_hands=config.MAX_HANDS,
        detection_confidence=config.DETECTION_CONFIDENCE,
        tracking_confidence=config.TRACKING_CONFIDENCE,
    )

    classifier = GestureClassifier(
        config.PINCH_DISTANCE_THRESHOLD,
        config.OPEN_DISTANCE_THRESHOLD,
        config.FIST_DISTANCE_THRESHOLD,
    )

    debouncer = Debouncer(config.HOLD_FRAMES, config.COOLDOWN_SECONDS)
    runner = ActionRunner()

    if config.START_WITH_ACTIONS_ENABLED == False:
        runner.toggle_enabled()

    if tracker.is_camera_open() == False:
        print("Error: The webcam could not be opened.")
        return

    print("Gesture controller started.")
    print("Press '" + config.TOGGLE_ACTIONS_KEY + "' in the video window to switch actions on or off.")
    print("Press '" + config.QUIT_KEY + "' in the video window to quit.")

    last_confirmed_text = "none"

    while True:
        frame = tracker.read_frame()

        if frame is None:
            print("Error: A frame could not be read from the webcam.")
            break

        landmarks = tracker.find_landmarks(frame)

        gesture_label = "idle"
        confidence_score = 0.0

        if landmarks is not None:
            tracker.draw_hand(frame)
            feature_list = extract_features(landmarks)
            gesture_label, confidence_score = classifier.predict(feature_list)

            if confidence_score < config.MIN_CONFIDENCE:
                gesture_label = "idle"

        confirmed_gesture = debouncer.update(gesture_label)

        if confirmed_gesture is not None:
            if confirmed_gesture in config.ACTION_MAP:
                action_name = config.ACTION_MAP[confirmed_gesture]
                runner.run(action_name)
                last_confirmed_text = confirmed_gesture + " -> " + action_name
            else:
                last_confirmed_text = confirmed_gesture + " (no action assigned)"

        if landmarks is None:
            gesture_text = "Gesture: no hand detected"
        else:
            gesture_text = "Gesture: " + gesture_label + " (" + str(round(confidence_score, 2)) + ")"

        if debouncer.candidate_label is None:
            holding_text = "Holding: -"
        else:
            holding_text = "Holding: " + debouncer.candidate_label + " " + str(debouncer.frame_count) + "/" + str(config.HOLD_FRAMES)

        if runner.is_enabled() == True:
            actions_text = "Actions: ON  (press " + config.TOGGLE_ACTIONS_KEY + " to switch)"
            actions_color = (0, 255, 0)
        else:
            actions_text = "Actions: OFF (press " + config.TOGGLE_ACTIONS_KEY + " to switch)"
            actions_color = (0, 0, 255)

        draw_text(frame, gesture_text, 10, 30, (0, 255, 0))
        draw_text(frame, holding_text, 10, 60, (0, 255, 255))
        draw_text(frame, "Last confirmed: " + last_confirmed_text, 10, 90, (255, 255, 255))
        draw_text(frame, actions_text, 10, 120, actions_color)

        cv2.imshow(config.WINDOW_NAME, frame)

        pressed_key = cv2.waitKey(1)
        if pressed_key == ord(config.QUIT_KEY):
            break
        if pressed_key == ord(config.TOGGLE_ACTIONS_KEY):
            runner.toggle_enabled()

    tracker.release()
    print("Gesture controller stopped.")


if __name__ == "__main__":
    main()