import cv2
import math
import time

REAL_FACE_WIDTH = 0.15
FOCAL_LENGTH = 709.20


def calculate_distance(face_width_pixels):
    if face_width_pixels <= 0:
        return 0

    return (
        FOCAL_LENGTH * REAL_FACE_WIDTH
    ) / face_width_pixels


def calculate_angle(face_center_x, image_center_x):
    angle_radians = math.atan(
        (face_center_x - image_center_x)
        / FOCAL_LENGTH
    )

    return math.degrees(angle_radians)


face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("Camera open aagala")
    raise SystemExit

previous_time = time.time()

while True:
    success, frame = camera.read()

    if not success:
        print("Frame capture aagala")
        break

    frame_height, frame_width = frame.shape[:2]

    image_center_x = frame_width // 2

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    current_time = time.time()

    elapsed_time = current_time - previous_time

    if elapsed_time > 0:
        fps = 1 / elapsed_time
    else:
        fps = 0

    previous_time = current_time

    # Professional dashboard background
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (frame_width, 120),
        (30, 30, 30),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0
    )

    cv2.putText(
        frame,
        "AI FACE DISTANCE MONITOR",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Faces Detected: {len(faces)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    alert_message = "No face detected"
    alert_color = (0, 0, 255)

    for person_number, (x, y, w, h) in enumerate(
        faces,
        start=1
    ):
        face_center_x = x + w // 2

        distance = calculate_distance(w)

        angle = calculate_angle(
            face_center_x,
            image_center_x
        )

        if angle > 2:
            direction = "RIGHT"
        elif angle < -2:
            direction = "LEFT"
        else:
            direction = "CENTER"

        if distance < 0.45:
            status = "TOO CLOSE"
            box_color = (0, 0, 255)
            alert_message = "ALERT: Please move back"
            alert_color = (0, 0, 255)

        elif distance < 0.80:
            status = "CAUTION"
            box_color = (0, 255, 255)
            alert_message = "CAUTION: Maintain distance"
            alert_color = (0, 255, 255)

        else:
            status = "SAFE"
            box_color = (0, 255, 0)
            alert_message = "SAFE DISTANCE"
            alert_color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            3
        )

        text_y = max(y - 10, 145)

        cv2.putText(
            frame,
            f"Person {person_number}",
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )

        cv2.putText(
            frame,
            f"{distance:.2f} m",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )

        cv2.putText(
            frame,
            f"{angle:.1f} deg | {direction}",
            (x, y + h + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            box_color,
            2
        )

        cv2.putText(
            frame,
            status,
            (x, y + h + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )

    # Bottom alert panel
    cv2.rectangle(
        frame,
        (0, frame_height - 60),
        (frame_width, frame_height),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        alert_message,
        (20, frame_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        alert_color,
        2
    )

    cv2.imshow(
        "AI Face Distance Monitor",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()