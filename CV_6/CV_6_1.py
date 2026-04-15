import cv2 # OpenCV 라이브러리 불러오기
import numpy as np # NumPy 라이브러리 불러오기
from deep_sort_realtime.deepsort_tracker import DeepSort # Deep SORT 트래커 불러오기

weights_path = "yolov3.weights" # YOLOv3 가중치 파일 경로
config_path = "yolov3.cfg" # YOLOv3 구성 파일 경로
video_path = "slow_traffic_small.mp4" # 비디오 파일 경로

# YOLOv3 모델 로드
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
#모든 레이어 이름 가져오기
layer_names = net.getLayerNames()
#출력 레이어만 추출
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Deep SORT 추적기 생성
# max_age=30: 객체가 30 프레임 동안 검출되지 않으면 객체 제거
tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(video_path) #비디오 파일 열기

#영상 프레임 반복 처리
while True:
    #한 프레임 읽기
    ret, frame = cap.read()
    if not ret: #더이상 프레임이 없으면 종료
        break
    #프레임 크기 가져오기
    height, width, _ = frame.shape

    # YOLO 입력용 이미지 생성
    blob = cv2.dnn.blobFromImage(
        frame,          #입력 이미지
        1 / 255.0,      #정규화
        (416, 416),     #YOLO 입력 크기
        swapRB=True,    #RGB로 변환
        crop=False      #이미지 자르지 않음
    )

    net.setInput(blob) #모델에 입력 설정

    # 객체 검출(출력 레이어 결과 얻기)
    outputs = net.forward(output_layers)
    
    boxes = [] #바운딩 박스 저장 리스트
    confidences = [] #신뢰도 저장
    class_ids = [] #클래스 ID 저장

    # 검출 결과 처리
    for output in outputs:
        for detection in output:
            scores = detection[5:] #클레스별 신뢰도 점수
            class_id = np.argmax(scores) #가장 높은 신뢰도의 클레스 선택
            confidence = scores[class_id] #해당 클래스의 확률
            if confidence > 0.5: #신뢰도가 0.5 이상인 경우
                # 중심 좌표
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                # 박스 크기
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌상단 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h]) #박스 정보 저장
                confidences.append(float(confidence)) #신뢰도 저장
                class_ids.append(class_id) #클래스 ID 저장

    # NMS 적용
    #중복 검출 제거
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detections = [] #Deep SORT 입력용 리스트

    # Deep SORT 입력 생성
    if len(indices) > 0: #검출된 객체가 있을 때
        for i in indices.flatten():
            x, y, w, h = boxes[i] #박스 정보 가져오기
            score = confidences[i]#신뢰도

            # class_id를 문자열로 사용
            detections.append(([x, y, w, h], score, str(class_ids[i])))

    # Deep SORT 업데이트
    tracks = tracker.update_tracks(detections, frame=frame)

    # 결과 시각화
    for track in tracks:
        #확정된 객체만
        if not track.is_confirmed():
            continue

        track_id = track.track_id #객체 고유ID
        ltrb = track.to_ltrb() #좌표

        x1, y1, x2, y2 = map(int, ltrb) #좌표 정수로 변환

        # 박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ID 표시
        cv2.putText(
            frame,                     
            f"ID: {track_id}",          #텍스트 내용  
            (x1, y1 - 10),              #위치
            cv2.FONT_HERSHEY_SIMPLEX,   #폰트
            0.6,                        #크기
            (0, 255, 0),                #색상
            2                           #두께
        )
    #결과 화면 출력
    cv2.imshow("YOLOv3 + Deep SORT", frame)
    #'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# 비디오 종료
cap.release()
#모든 창 닫기
cv2.destroyAllWindows()