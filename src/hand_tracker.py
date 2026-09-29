import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self, camera_index=0, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.camera = cv2.VideoCapture(camera_index)

        # Initialize the MediaPipe Hands solution and drawing utilities
        self.mp_hands = mp.solutions.hands # Initialize the MediaPipe Hands solution
        self.mp_drawing = mp.solutions.drawing_utils # Initialize the MediaPipe drawing utilities

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence, # Set the minimum tracking confidence for hand tracking
        )

        self.latest_hand = None

    def is_camera_open(self):
        return self.camera.isOpened()

    def read_frame(self):
        success, frame = self.camera.read()

        if success == False:
            return None

        frame = cv2.flip(frame, 1)
        return frame

    def find_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame) # Process the RGB frame to detect hands and extract landmarks

        if results.multi_hand_landmarks is None:
            self.latest_hand = None
            return None

        first_hand = results.multi_hand_landmarks[0]
        self.latest_hand = first_hand

        landmark_list = []
        for landmark in first_hand.landmark:
            x_value = landmark.x
            y_value = landmark.y
            landmark_list.append((x_value, y_value))

        return landmark_list

    def draw_hand(self, frame):
        if self.latest_hand is None:
            return

        self.mp_drawing.draw_landmarks(
            frame,
            self.latest_hand,
            self.mp_hands.HAND_CONNECTIONS,
        )

    def release(self):
        self.camera.release()
        self.hands.close()
        cv2.destroyAllWindows()