from controller import Robot
import math
import cv2
import numpy as np
import tensorflow as tf


MODEL_FILE = "hsv_color_classifier.keras"

CLASS_NAMES = ["Red", "Green", "Blue"]

ARM_UP = 0.00
ARM_DOWN = 0.05
GRIPPER_OPEN = 0.065
GRIPPER_CLOSED = 0.030

SEARCH_SPEED = 0.8
BASE_FORWARD_SPEED = 0.7
SLOW_SPEED = 0.10
SIDE_ADJUST_SPEED = 0.12
BACK_SPEED = -0.6

MAX_WHEEL_SPEED = 3.0
KP_CENTER = 0.45

CONFIDENCE_THRESHOLD = 0.70
MIN_MASK_AREA = 80

TURN_SPEED = 0.35
ZONE_SPEED = 0.55
ANGLE_TOL = 0.15
DIST_TOL = 0.25

MIN_DEPTH = 0.05
MAX_DEPTH = 3.0

DROP_LOWER_TIME = 900
DROP_RELEASE_TIME = 2500
DROP_BACK_SPEED = -0.18
DROP_RAISE_TIME = 1500

BACK_AWAY_TIME = 20000
TURN_RESET_TIME = 1200
TURN_RESET_SPEED = 0.5
RESET_READY_TIME = 1500

SEARCH_TIMEOUT = 15000

DROP_ZONE_POINTS = {
    "Blue": (-2.0, 2.0),
    "Red": (-2.0, 0.0),
    "Green": (-2.0, -2.0),
}

DROP_SLOT_OFFSETS = [
    (0.0, 0.35),
    (0.0, -0.35),
    (0.35, 0.0),
    (-0.35, 0.0),
    (0.0, 0.0),
]

drop_count = {
    "Blue": 0,
    "Red": 0,
    "Green": 0,
}

MID_POINT = (0.6, -0.2)
RESET_POINT = (1.0, 1.0)


robot = Robot()
timestep = int(robot.getBasicTimeStep())


def get_device_any(names):
    for name in names:
        device = robot.getDevice(name)
        if device is not None:
            return device
    return None


color_camera = robot.getDevice("color camera")
depth_camera = robot.getDevice("depth camera")
front_touch = robot.getDevice("gripper touch sensor")
center_touch = robot.getDevice("center touch sensor")

color_camera.enable(timestep)
depth_camera.enable(timestep)
front_touch.enable(timestep)
center_touch.enable(timestep)

left_gripper_touch = get_device_any(["left gripper touch", "left gripper touch "])
right_gripper_touch = get_device_any(["right gripper touch", "right gripper touch "])

if left_gripper_touch:
    left_gripper_touch.enable(timestep)

if right_gripper_touch:
    right_gripper_touch.enable(timestep)

gps = robot.getDevice("gps")
compass = robot.getDevice("compass")

gps.enable(timestep)
compass.enable(timestep)

left_motor = robot.getDevice("left wheel")
right_motor = robot.getDevice("right wheel")

left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

lift_motor = robot.getDevice("lift motor")
left_finger_motor = robot.getDevice("finger motor::left")
right_finger_motor = robot.getDevice("finger motor::right")

lift_motor.setVelocity(0.12)
left_finger_motor.setVelocity(0.12)
right_finger_motor.setVelocity(0.12)

lift_motor.setPosition(ARM_UP)
left_finger_motor.setPosition(GRIPPER_OPEN)
right_finger_motor.setPosition(GRIPPER_OPEN)

model = tf.keras.models.load_model(MODEL_FILE, compile=False)


state = "SEARCH"
last_state = None
state_timer = 0
object_color = None


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def set_speed(left, right):
    left_motor.setVelocity(left)
    right_motor.setVelocity(right)


def stop_robot():
    set_speed(0.0, 0.0)


def set_gripper(position):
    left_finger_motor.setPosition(position)
    right_finger_motor.setPosition(position)


def predict_color(h, s, v):
    sample = np.array([[h / 179.0, s / 255.0, v / 255.0]], dtype=np.float32)
    probabilities = model.predict(sample, verbose=0)[0]

    label_index = int(np.argmax(probabilities))
    color_confidence = float(probabilities[label_index])

    return CLASS_NAMES[label_index], color_confidence


def get_hsv_frame():
    image = color_camera.getImage()
    width = color_camera.getWidth()
    height = color_camera.getHeight()

    frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    bgr = frame[:, :, :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    return hsv, width, height


def get_depth_frame():
    width = depth_camera.getWidth()
    height = depth_camera.getHeight()

    return np.array(depth_camera.getRangeImage(), dtype=np.float32).reshape((height, width))


def build_color_mask(hsv, color_name):
    if color_name == "Red":
        lower_red = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        upper_red = cv2.inRange(hsv, (170, 80, 80), (179, 255, 255))
        return cv2.bitwise_or(lower_red, upper_red)

    if color_name == "Green":
        return cv2.inRange(hsv, (45, 80, 80), (85, 255, 255))

    return cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))


def get_largest_component(mask):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    if count <= 1:
        return None, None, None, None

    best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    best_area = stats[best_label, cv2.CC_STAT_AREA]

    if best_area < MIN_MASK_AREA:
        return None, None, None, None

    return best_label, labels, centroids, best_area


def get_color_depth_target(color_name):
    hsv, width, height = get_hsv_frame()
    depth = get_depth_frame()

    mask = build_color_mask(hsv, color_name)

    if cv2.countNonZero(mask) < MIN_MASK_AREA:
        return None, None, None

    label, labels, centroids, object_area = get_largest_component(mask)

    if label is None:
        return None, None, None

    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height))

    object_mask = labels == label

    valid_depth = (
        object_mask
        & np.isfinite(depth)
        & (depth > MIN_DEPTH)
        & (depth < MAX_DEPTH)
    )

    if np.count_nonzero(valid_depth) < MIN_MASK_AREA:
        return None, None, None

    center_x = centroids[label][0]
    object_offset = (center_x - width / 2.0) / (width / 2.0)
    object_distance = float(np.median(depth[valid_depth]))

    return object_offset, object_distance, object_area


def get_camera_target_by_color(color_name):
    hsv, width, _ = get_hsv_frame()
    mask = build_color_mask(hsv, color_name)

    camera_area = cv2.countNonZero(mask)

    if camera_area < 20:
        return None, None

    moments = cv2.moments(mask)

    if moments["m00"] == 0:
        return None, None

    object_x = moments["m10"] / moments["m00"]
    camera_offset = (object_x - width / 2.0) / (width / 2.0)

    return camera_offset, camera_area


def drive_to_object(object_offset, object_distance):
    correction = 0.0 if abs(object_offset) < 0.08 else KP_CENTER * object_offset

    if object_distance > 0.80:
        forward = 0.75
    elif object_distance > 0.50:
        forward = 0.55
    elif object_distance > 0.30:
        forward = 0.35
    else:
        forward = 0.18

    if abs(object_offset) > 0.45:
        forward = 0.10

    left = clamp(forward + correction, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    right = clamp(forward - correction, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

    set_speed(left, right)


def drive_to_camera_target(camera_offset):
    correction = KP_CENTER * camera_offset
    forward = 0.12 if abs(camera_offset) > 0.45 else BASE_FORWARD_SPEED

    left = clamp(forward + correction, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)
    right = clamp(forward - correction, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)

    set_speed(left, right)


def get_robot_position():
    position = gps.getValues()
    return position[0], position[1]


def get_robot_angle():
    values = compass.getValues()
    return math.atan2(values[0], values[1])


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle


def get_zone_drop_point(color_name):
    base_x, base_y = DROP_ZONE_POINTS[color_name]

    slot_index = drop_count[color_name] % len(DROP_SLOT_OFFSETS)
    offset_x, offset_y = DROP_SLOT_OFFSETS[slot_index]

    return base_x + offset_x, base_y + offset_y


def go_to_point(target_x, target_y):
    x, y = get_robot_position()
    heading = get_robot_angle()

    dx = target_x - x
    dy = target_y - y

    distance = math.sqrt(dx * dx + dy * dy)
    target_angle = math.atan2(dy, dx)
    error = normalize_angle(target_angle - heading)

    if distance < DIST_TOL:
        stop_robot()
        return True

    if abs(error) > ANGLE_TOL:
        if error > 0:
            set_speed(-TURN_SPEED, TURN_SPEED)
        else:
            set_speed(TURN_SPEED, -TURN_SPEED)
    else:
        set_speed(ZONE_SPEED, ZONE_SPEED)

    return False


while robot.step(timestep) != -1:
    front_touch_value = front_touch.getValue()
    center_touch_value = center_touch.getValue()

    left_touch_value = left_gripper_touch.getValue() if left_gripper_touch else 0.0
    right_touch_value = right_gripper_touch.getValue() if right_gripper_touch else 0.0

    if state != last_state:
        if state == "SEARCH":
            print("Searching for object")

        elif state == "APPROACH":
            print("Going to object:", object_color)

        elif state == "CENTER":
            print("Centering object")

        elif state == "BACK_BEFORE_PICK":
            print("Backing before pick")

        elif state == "LOWER_ARM":
            print("Lowering arm")

        elif state == "GRAB":
            print("Grabbing object")

        elif state == "LIFT":
            print("Lifting object")

        elif state == "GO_TO_ZONE":
            print("Going to zone:", object_color)

        elif state == "LOWER_DROP":
            print("Lowering object")

        elif state == "RELEASE":
            print("Releasing object")

        elif state == "BACK_AWAY":
            print("Backing away")

        elif state == "RAISE_AFTER_DROP":
            print("Raising arm after drop")

        elif state == "TURN":
            print("Turning to reset")

        elif state == "GO_TO_MID":
            print("Going to mid point")

        elif state == "GO_TO_RESET":
            print("Going to reset point")

        elif state == "READY":
            print("Ready")

        elif state == "STOP_NO_OBJECT":
            print("No object found")

        last_state = state

    if state == "SEARCH":
        hsv, width, height = get_hsv_frame()

        center_x = width // 2
        center_y = height // 2

        h, s, v = hsv[center_y, center_x]
        detected_color, color_confidence = predict_color(float(h), float(s), float(v))

        if color_confidence < CONFIDENCE_THRESHOLD or s < 80 or v < 80:
            state_timer += timestep

            if state_timer > SEARCH_TIMEOUT:
                stop_robot()
                state = "STOP_NO_OBJECT"
                state_timer = 0
            else:
                set_speed(SEARCH_SPEED, -SEARCH_SPEED)

        else:
            object_color = detected_color
            lift_motor.setPosition(ARM_UP)
            set_gripper(GRIPPER_OPEN)
            state = "APPROACH"
            state_timer = 0

    elif state == "APPROACH":
        object_offset, object_distance, object_area = get_color_depth_target(object_color)

        if object_offset is None:
            camera_offset, camera_area = get_camera_target_by_color(object_color)

            if camera_offset is None:
                set_speed(SEARCH_SPEED, -SEARCH_SPEED)
            else:
                drive_to_camera_target(camera_offset)

        else:
            drive_to_object(object_offset, object_distance)

        if front_touch_value > 0:
            stop_robot()
            lift_motor.setPosition(ARM_UP)
            set_gripper(GRIPPER_OPEN)
            state = "CENTER"
            state_timer = 0

    elif state == "CENTER":
        if left_touch_value > 0 and right_touch_value == 0:
            set_speed(SIDE_ADJUST_SPEED, -SIDE_ADJUST_SPEED)

        elif right_touch_value > 0 and left_touch_value == 0:
            set_speed(-SIDE_ADJUST_SPEED, SIDE_ADJUST_SPEED)

        else:
            set_speed(SLOW_SPEED, SLOW_SPEED)

        if center_touch_value > 0:
            stop_robot()
            state = "BACK_BEFORE_PICK"
            state_timer = 0

    elif state == "BACK_BEFORE_PICK":
        set_speed(BACK_SPEED, BACK_SPEED)

        state_timer += timestep
        if state_timer > 500:
            stop_robot()
            lift_motor.setPosition(ARM_DOWN)
            state = "LOWER_ARM"
            state_timer = 0

    elif state == "LOWER_ARM":
        stop_robot()
        lift_motor.setPosition(ARM_DOWN)

        state_timer += timestep
        if state_timer > 1200:
            state = "GRAB"
            state_timer = 0

    elif state == "GRAB":
        stop_robot()
        set_gripper(GRIPPER_CLOSED)

        state_timer += timestep
        if state_timer > 1800:
            state = "LIFT"
            state_timer = 0

    elif state == "LIFT":
        stop_robot()
        set_gripper(GRIPPER_CLOSED)
        lift_motor.setPosition(ARM_UP)

        state_timer += timestep
        if state_timer > 2200:
            state = "GO_TO_ZONE"
            state_timer = 0

    elif state == "GO_TO_ZONE":
        set_gripper(GRIPPER_CLOSED)
        lift_motor.setPosition(ARM_UP)

        if object_color is None or object_color not in DROP_ZONE_POINTS:
            object_color = None
            state = "READY"
            state_timer = 0
            continue

        drop_x, drop_y = get_zone_drop_point(object_color)

        if go_to_point(drop_x, drop_y):
            stop_robot()
            state = "LOWER_DROP"
            state_timer = 0

    elif state == "LOWER_DROP":
        stop_robot()
        lift_motor.setPosition(ARM_DOWN)
        set_gripper(GRIPPER_CLOSED)

        state_timer += timestep
        if state_timer > DROP_LOWER_TIME:
            state = "RELEASE"
            state_timer = 0

    elif state == "RELEASE":
        stop_robot()
        lift_motor.setPosition(ARM_DOWN)

        progress = min(state_timer / DROP_RELEASE_TIME, 1.0)
        gripper_position = GRIPPER_CLOSED + (GRIPPER_OPEN - GRIPPER_CLOSED) * progress
        set_gripper(gripper_position)

        state_timer += timestep

        if state_timer > DROP_RELEASE_TIME + 1000:
            set_gripper(GRIPPER_OPEN)

            if object_color in drop_count:
                drop_count[object_color] += 1

            object_color = None
            state = "BACK_AWAY"
            state_timer = 0

    elif state == "BACK_AWAY":
        lift_motor.setPosition(ARM_DOWN)
        set_gripper(GRIPPER_OPEN)
        set_speed(DROP_BACK_SPEED, DROP_BACK_SPEED)

        state_timer += timestep
        if state_timer > BACK_AWAY_TIME:
            stop_robot()
            state = "RAISE_AFTER_DROP"
            state_timer = 0

    elif state == "RAISE_AFTER_DROP":
        stop_robot()
        set_gripper(GRIPPER_OPEN)
        lift_motor.setPosition(ARM_UP)

        state_timer += timestep
        if state_timer > DROP_RAISE_TIME:
            state = "TURN"
            state_timer = 0

    elif state == "TURN":
        set_speed(TURN_RESET_SPEED, -TURN_RESET_SPEED)

        state_timer += timestep
        if state_timer > TURN_RESET_TIME:
            stop_robot()
            state = "GO_TO_MID"
            state_timer = 0

    elif state == "GO_TO_MID":
        mid_x, mid_y = MID_POINT

        if go_to_point(mid_x, mid_y):
            state = "GO_TO_RESET"
            state_timer = 0

    elif state == "GO_TO_RESET":
        reset_x, reset_y = RESET_POINT

        if go_to_point(reset_x, reset_y):
            stop_robot()
            object_color = None
            lift_motor.setPosition(ARM_UP)
            set_gripper(GRIPPER_OPEN)
            state = "READY"
            state_timer = 0

    elif state == "READY":
        stop_robot()
        lift_motor.setPosition(ARM_UP)
        set_gripper(GRIPPER_OPEN)

        state_timer += timestep
        if state_timer > RESET_READY_TIME:
            object_color = None
            state = "SEARCH"
            state_timer = 0

    elif state == "STOP_NO_OBJECT":
        stop_robot()
        lift_motor.setPosition(ARM_UP)
        set_gripper(GRIPPER_OPEN)