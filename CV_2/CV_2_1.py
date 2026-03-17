import cv2 #opencv라이브러리 불러오기
import numpy as np #수칙계산용 NumPy 라이브러리 불러오기
import glob #특정 경로의 여러 이미지 파일명을 한번에 불러오기위한 모듈

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = []
imgpoints = []

#이미지 파일들을 모두 불러오기
images = glob.glob("L02_lab\images\calibration_images/left*.jpg")
img_size = None #이미지 크기를 저장하기 위한 변수

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------

for fname in images: #불러온 모든 이미지 파일에 대해 반복 수행
    img = cv2.imread(fname) #현재 파일 경로의 이미지를 읽어옴
    if img is None: #이미지가 정상적으로 읽히는지 확인
        print(f"이미지를 읽을 수 없습니다: {fname}")
        continue
    #코너검출을 위해 이미지를 그레이스케일로 변환
    #코너 검출은 색이 아닌 명암의 변화를 보고 계산 속도를 줄일 수 있기 때문
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #현재 이미지의 크기 저장
    img_size = gray.shape[::-1]  # (width, height)

    #체크보드 내부 코너를 검출함
    #ret=검출 성공 여부, corners=검출된 코너 좌표
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    #코너가 정상적으로 검출된 경우
    if ret:
        #이미지에 대한 실제 좌표를 objpoints 리스트에 추가
        objpoints.append(objp) 

        # 코너 정밀화
        #(11,11)->탐색 윈도우 크기, (-1,-1)-> dead zone 없음을 의미
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #정밀화된 코너 좌표를 imgpoints 리스트에 추가
        imgpoints.append(corners2)
    #체크보드 코너 검출에 실패한 경우 출력
    else:
        print(f"코너 검출 실패: {fname}")
cv2.destroyAllWindows()#반복이 끝난 뒤 opencv 창을 모두 닫기

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------

#검출한 실제 좌표(obgpoint)와 이미지 좌표(imgpoint)를 이용해 
#카메라 내부 행렬(k), 외곡계수(dist), 회전/이동 벡터를 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, #실제 세계 좌표(3d)
    imgpoints, #이미지 좌표(2d)
    img_size,  #이미지 크기
    None,      #초기 카메라 행렬값 없음
    None       #초기 왜곡 계수값 없음
)

#계산된 카메라 내부 행렬 출력
print("Camera Matrix K:")
print(K)

#계산된 왜곡 계수 출력
print("\nDistortion Coefficients:")
print(dist)

#재투영 오차 출력
#값이 작을수록 보정이 잘 된 것으로 간주
print("\nReprojection Error:")
print(ret)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
#첫번째 이미지를 불러와 왜곡 보정 전을 보여줌
test_img = cv2.imread(images[0])

#계산한 카메라 행렬 K와 왜곡 계수 dist를 이용해 이미지를 보정
undistorted = cv2.undistort(test_img, K, dist, None, K)

cv2.imshow("Original", test_img) #원본 이미지를 화면에 출력
cv2.imshow("Undistorted", undistorted) #왜곡 보정된 이미지를 화면에 출력
cv2.waitKey(0) #키입력이 들어올 때까지 화면을 유지
cv2.destroyAllWindows() #모든 opencv창 닫기
