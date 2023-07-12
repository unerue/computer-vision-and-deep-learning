import cv2
import sys

img = cv2.imread("soccer.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

BrushSiz = 5  # 붓의 크기
LColor, RColor = (255, 0, 0), (0, 0, 255)  # 파란색과 빨간색


def painting(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), BrushSiz, LColor, -1)  # 마우스 왼쪽 버튼 클릭하면 파란색
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.circle(img, (x, y), BrushSiz, RColor, -1)  # 마우스 오른쪽 버튼 클릭하면 빨간색
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img, (x, y), BrushSiz, LColor, -1)  # 왼쪽 버튼 클릭하고 이동하면 파란색
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON:
        cv2.circle(img, (x, y), BrushSiz, RColor, -1)  # 오른쪽 버튼 클릭하고 이동하면 빨간색

    cv2.imshow("Painting", img)  # 수정된 영상을 다시 그림


cv2.namedWindow("Painting")
cv2.imshow("Painting", img)

cv2.setMouseCallback("Painting", painting)

while True:
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        break
