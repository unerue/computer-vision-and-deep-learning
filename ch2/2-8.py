import cv2
import sys


img = cv2.imread("girl_laughing.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")


def draw(event, x, y, flags, param):
    global ix, iy

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭했을 때 초기 위치 저장
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:  # 마우스 왼쪽 버튼 클릭했을 때 직사각형 그리기
        cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)

    cv2.imshow("Drawing", img)


cv2.namedWindow("Drawing")
cv2.imshow("Drawing", img)

cv2.setMouseCallback("Drawing", draw)

while True:
    if cv2.waitKey(1) == ord("q"):
        cv2.destroyAllWindows()
        break
