def calculate_distance(point_a, point_b):
    x_difference = point_a[0] - point_b[0]
    y_difference = point_a[1] - point_b[1]
    distance = (x_difference ** 2 + y_difference ** 2) ** 0.5
    return distance


def extract_features(landmark_list):
    wrist_point = landmark_list[0]
    middle_finger_base = landmark_list[9]

    relative_points = []
    for point in landmark_list:
        relative_x = point[0] - wrist_point[0]
        relative_y = point[1] - wrist_point[1]
        relative_points.append((relative_x, relative_y))

    scale_distance = calculate_distance(relative_points[0], relative_points[9]) # Calculate the distance between the wrist and the base of the middle finger to use as a scale for normalization

    if scale_distance == 0:
        scale_distance = 0.0001

    feature_list = []
    for point in relative_points:
        normalized_x = point[0] / scale_distance
        normalized_y = point[1] / scale_distance
        feature_list.append(normalized_x)
        feature_list.append(normalized_y)

    return feature_list