import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 획득에 실패하여 루프를 나갑니다.")
        break

    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    mp_drawing.draw_landmarks(
        frame,
        res.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
    )

    cv2.imshow("MediaPipe pose", cv2.flip(frame, 1))  # 좌우반전
    if cv2.waitKey(5) == ord("q"):
        mp_drawing.plot_landmarks(res.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        break

cap.release()
cv2.destroyAllWindows()
