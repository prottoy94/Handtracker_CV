import time


class Debouncer:
    def __init__(self, hold_frames=8, cooldown_seconds=1.5):
        self.hold_frames = hold_frames
        self.cooldown_seconds = cooldown_seconds

        self.candidate_label = None
        self.frame_count = 0
        self.locked_until_time = 0.0

    def reset(self):
        self.candidate_label = None
        self.frame_count = 0

    def update(self, label):
        current_time = time.monotonic()

        if current_time < self.locked_until_time:
            self.reset()
            return None

        if label == "idle":
            self.reset()
            return None

        if label == self.candidate_label:
            self.frame_count = self.frame_count + 1
        else:
            self.candidate_label = label
            self.frame_count = 1

        if self.frame_count >= self.hold_frames:
            confirmed_label = self.candidate_label
            self.locked_until_time = current_time + self.cooldown_seconds
            self.reset()
            return confirmed_label

        return None