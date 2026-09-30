import pyautogui


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


class ActionRunner:
    def __init__(self):
        self.actions_enabled = True

    def is_enabled(self):
        return self.actions_enabled

    def toggle_enabled(self):
        if self.actions_enabled == True:
            self.actions_enabled = False
        else:
            self.actions_enabled = True
        return self.actions_enabled

    def press_key(self, key_name):
        try:
            pyautogui.press(key_name)
            return True
        except pyautogui.FailSafeException:
            self.actions_enabled = False
            print("Fail-safe triggered. Actions are now disabled.")
            return False

    def run(self, action_name):
        if self.actions_enabled == False:
            return False

        if action_name == "next_slide":
            return self.press_key("right")

        if action_name == "previous_slide":
            return self.press_key("left")

        if action_name == "toggle_mute":
            return self.press_key("volumemute")

        print("Unknown action: " + str(action_name))
        return False