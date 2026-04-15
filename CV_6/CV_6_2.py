import cv2 #OpenCV 라이브러리 불러오기
import mediapipe as mp #MediaPipe 라이브러리 불러오기

# MediaPipe의 Face Mesh 솔루션 초기화
mp_face_mesh = mp.solutions.face_mesh
# Face Mesh 객체 생성
face_mesh = mp_face_mesh.FaceMesh()

# 웹캠 열기
#0은 기본 카메라를 의미
cap = cv2.VideoCapture(0)

#웹캠에서 프레임을 반복적으로 읽어오기
while True:
    ret, frame = cap.read()
    if not ret: #프레임을 읽어오지 못하면 종료
        break
    # 현재 프레임의 크기
    h, w, _ = frame.shape

    # BGR → RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 얼굴 랜드마크 검출
    result = face_mesh.process(rgb_frame)

    # 검출 된 얼굴 랜드마크 그리기
    if result.multi_face_landmarks:
        #검출된 각 얼국에 대해 반복
        for face_landmarks in result.multi_face_landmarks:
            #얼굴의 각 랜드마크 점에 대해 반복
            for lm in face_landmarks.landmark:
                #랜드마크 좌표를 이미지 크기에 맞게 변환
                x = int(lm.x * w)
                y = int(lm.y * h)
                #좌표 위 초록색 점 그리기
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # 결과 영상을 화면에 출력
    cv2.imshow("FaceMesh", frame)

    # ESC키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 웹캠 자원 해제
cap.release()
# openCV 창 닫기
cv2.destroyAllWindows()