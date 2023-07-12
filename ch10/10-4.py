import cv2
import mediapipe as mp

img = cv2.imread("BSDS_376001.jpg")

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)
res = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if not res.detections:
    print("얼굴 검출에 실패했습니다. 다시 시도하세요.")
else:
    for detection in res.detections:
        mp_drawing.draw_detection(img, detection)
    cv2.imshow("Face detection by MediaPipe", img)

cv2.waitKey()
cv2.destroyAllWindows()
