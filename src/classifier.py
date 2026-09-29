from src.features import calculate_distance


WRIST_INDEX = 0
THUMB_TIP_INDEX = 4
INDEX_TIP_INDEX = 8
MIDDLE_TIP_INDEX = 12


def get_point(feature_list, landmark_index):
    x_position = landmark_index * 2
    y_position = landmark_index * 2 + 1
    x_value = feature_list[x_position]
    y_value = feature_list[y_position]
    return (x_value, y_value)


class GestureClassifier:
    def __init__(self, pinch_distance_threshold=0.4, open_distance_threshold=1.3):
        self.pinch_distance_threshold = pinch_distance_threshold
        self.open_distance_threshold = open_distance_threshold

    def predict(self, feature_list):
        wrist_point = get_point(feature_list, WRIST_INDEX)
        thumb_tip_point = get_point(feature_list, THUMB_TIP_INDEX)
        index_tip_point = get_point(feature_list, INDEX_TIP_INDEX)
        middle_tip_point = get_point(feature_list, MIDDLE_TIP_INDEX)

        pinch_distance = calculate_distance(thumb_tip_point, index_tip_point)
        middle_finger_length = calculate_distance(wrist_point, middle_tip_point)

        if pinch_distance < self.pinch_distance_threshold:
            gesture_label = "pinch"
            confidence_score = 1.0 - (pinch_distance / self.pinch_distance_threshold)
            return gesture_label, confidence_score

        if middle_finger_length > self.open_distance_threshold:
            gesture_label = "open"
            confidence_score = min(middle_finger_length / self.open_distance_threshold, 1.0)
            return gesture_label, confidence_score

        gesture_label = "idle"
        confidence_score = 0.5
        return gesture_label, confidence_score