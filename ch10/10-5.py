import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 획득에 실패하여 루프를 나갑니다.")
        break

    res = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.detections:
        for detection in res.detections:
            mp_drawing.draw_detection(frame, detection)

    cv2.imshow("MediaPipe Face Detection from video", cv2.flip(frame, 1))
    if cv2.waitKey(5) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
