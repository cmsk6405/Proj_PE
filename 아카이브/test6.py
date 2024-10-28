

import cv2
import numpy as np
from ultralytics import YOLO
import time

def draw_person_contour(frame, mask):
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        return approx
    return None

def get_reference_contour(model, reference_image):
    results = model(reference_image, stream=True)
    for r in results:
        masks = r.masks
        if masks is not None:
            for mask in masks:
                contour = draw_person_contour(reference_image, mask.data[0].cpu().numpy())
                if contour is not None:
                    return contour
    return None

def draw_contour_in_bbox(frame, contour, bbox, color=(0, 255, 0)):
    x, y, w, h = bbox
    contour_points = contour.reshape(-1, 2)
    normalized_contour = (contour_points - contour_points.min(axis=0)) / (contour_points.max(axis=0) - contour_points.min(axis=0))
    scaled_contour = (normalized_contour * np.array([w, h]) + np.array([x, y])).astype(int)
    cv2.drawContours(frame, [scaled_contour], 0, color, 2)
    return scaled_contour

def calculate_pose_similarity(contour1, contour2):
    if contour1 is None or contour2 is None or len(contour1) < 4 or len(contour2) < 4:
        return 0.0

    # Normalize contours
    contour1 = (contour1 - contour1.min(axis=0)) / (contour1.max(axis=0) - contour1.min(axis=0))
    contour2 = (contour2 - contour2.min(axis=0)) / (contour2.max(axis=0) - contour2.min(axis=0))
    
    # Calculate Hausdorff distance
    distances = np.max([
        np.min([np.linalg.norm(p1 - p2) for p2 in contour2]) for p1 in contour1
    ])
    
    # Convert distance to similarity score (higher is more similar)
    similarity = 1 / (1 + distances)
    return similarity


class PoseReference:
    def __init__(self, image_path, target_count):
        self.image_path = image_path
        self.target_count = target_count
        self.image = cv2.imread(image_path)
        self.contour = None

def load_reference_images(model, reference_data):
    references = []
    for data in reference_data:
        ref = PoseReference(data['path'], data['count'])
        if ref.image is None:
            print(f"레퍼런스 이미지를 불러올 수 없습니다: {ref.image_path}")
            continue
        ref.contour = get_reference_contour(model, ref.image)
        if ref.contour is None:
            print(f"레퍼런스 이미지에서 사람을 찾을 수 없습니다: {ref.image_path}")
            continue
        references.append(ref)
    return references

def overlay_next_reference(current_ref_image, next_ref_image, overlay_size=0.3):
    # 현재 레퍼런스 이미지 크기
    h, w = current_ref_image.shape[:2]
    
    # 다음 레퍼런스 이미지 리사이즈
    overlay_h, overlay_w = int(h * overlay_size), int(w * overlay_size)
    next_ref_resized = cv2.resize(next_ref_image, (overlay_w, overlay_h))
    
    # 오버레이할 위치 계산
    x_offset = w - overlay_w
    y_offset = h - overlay_h
    
    # 알파 채널을 사용하여 이미지 합성
    alpha = 1
    for c in range(0, 3):
        current_ref_image[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w, c] = \
            (alpha * next_ref_resized[:, :, c] +
             (1. - alpha) * current_ref_image[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w, c])
    
    return current_ref_image

def draw_contour_fixed_center(frame, contour, color, scale, move_x, move_y):
    frame_height, frame_width = frame.shape[:2]
    contour_points = contour.reshape(-1, 2)
    
    # 윤곽선의 경계 상자 계산
    x, y, w, h = cv2.boundingRect(contour)
    contour_aspect_ratio = w / h

    # 프레임에 맞는 최대 크기 계산
    if frame_width / frame_height > contour_aspect_ratio:
        # 프레임이 더 넓은 경우
        new_height = int(frame_height * scale)
        new_width = int(new_height * contour_aspect_ratio)
    else:
        # 프레임이 더 좁은 경우
        new_width = int(frame_width * scale)
        new_height = int(new_width / contour_aspect_ratio)

    # 윤곽선 정규화 및 스케일링
    normalized_contour = (contour_points - [x, y]) / [w, h]
    scaled_contour = normalized_contour * [new_width, new_height]

    # 윤곽선을 화면 중앙에 위치시킴
    center_x = (frame_width // 2) - new_width // 2
    center_y = (frame_height // 2) - new_height // 2
    centered_contour = scaled_contour + [center_x + move_x, center_y + move_y]

    centered_contour = centered_contour.astype(int)
    cv2.drawContours(frame, [centered_contour], 0, color, 2)
    return centered_contour

def is_contour_smaller(contour1, contour2):
    """contour1이 contour2보다 작은지 확인합니다."""
    area1 = cv2.contourArea(contour1)
    area2 = cv2.contourArea(contour2)
    return area1 < area2


def main():
    # YOLO 모델 로드
    model = YOLO('yolov8n-seg.pt')

    # 레퍼런스 이미지 데이터 (경로와 목표 횟수)
    reference_data = [
        {'path': './image/yoga.png', 'count': 2},
        {'path': './image/yoga2.png', 'count': 3},
        {'path': './image/yoga3.png', 'count': 4},
    ]


    # 레퍼런스 이미지들 로드
    references = load_reference_images(model, reference_data)
    if not references:
        print("유효한 레퍼런스 이미지가 없습니다.")
        return

    # 웹캠 설정
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    color =(255,255,255) # 처음 외곽선의 색, 흰 색
    current_ref_index = 0
    pose_count = 0
    frame_count = 0
    start_time = time.time()
    wrong_pose_start_time = None
    
    similarity_threshold = 0.5 # 유사도 임계값
    cooldown_frames = 15 # 쿨다운 프레임 수 (약 1초, 30fps 기준)
    reference_scale = 0.8 # 초기 기준 포즈 크기 (0.1 ~ 1.0)
    time_per_image = 4 # 각 이미지당 시간 (초)
    wrong_pose_delay = 0.5 # 자세 틀렸을 시 외곽선 색 변하기전 딜레이(초)
    move_x = 0
    move_y = 0
    wrong_pose_color = (0, 0, 255)
    right_pose_color = (255, 0, 0)
    

    while True:
        # 웹 캠 읽어오기
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 프레임을 받아올 수 없습니다.")
            break

        # 현재 사진 번호
        current_ref = references[current_ref_index]
        # 다음 사진 번호
        next_ref_index = (current_ref_index + 1) % len(references)
        # 다음 사진
        next_ref = references[next_ref_index]

        # 현재 레퍼런스 이미지에 다음 레퍼런스 이미지 오버레이
        current_ref_with_next = overlay_next_reference(current_ref.image.copy(), next_ref.image)

        # 기준 포즈의 실루엣을 화면 중앙에 고정 (크기 조절 가능)
        reference_fixed = draw_contour_fixed_center(frame, current_ref.contour, color, reference_scale, move_x, move_y)

        # YOLO로 세그멘테이션 수행
        results = model(frame)[0]

        # seg로 구한 마스크로 for문
        for r in results:
            for data in r.boxes.data:
                # 사람인지 확인
                _,_,_,_,_, cls =data
                if cls == 0:
                    masks = r.masks
                    # 마스크가 있다면 외곽선을 그린다
                    if masks is not None:
                        for mask in masks:
                            current_contour = draw_person_contour(frame, mask.data[0].cpu().numpy())
                            
                            if current_contour is not None:
                                cv2.drawContours(frame, [current_contour], 0, (0, 255, 0), 2)
                                # 정확도 비교
                                similarity = calculate_pose_similarity(current_contour, reference_fixed)
                                #정확도와 쿨다운이 지나면 다음 이미지로 넘어감
                                if (similarity > similarity_threshold and 
                                    frame_count >= cooldown_frames and 
                                    is_contour_smaller(current_contour, reference_fixed)):
                                    pose_count += 1
                                    frame_count = 0
                                    color = right_pose_color
                                    # 횟수 카운트 하고 이미지 대기 시간 초기화
                                    if pose_count >= current_ref.target_count:
                                        current_ref_index = next_ref_index
                                        pose_count = 0
                                        start_time = time.time()  # 새 이미지로 넘어갈 때 시간 초기화
                                        print(f"다음 포즈로 이동: {references[current_ref_index].image_path}")

                                        wrong_pose_start_time = time.time()
#
                                else:
                                    #w자세가 틀리면 색이 다르게 해야함
                                    if wrong_pose_start_time is None:
                                        wrong_pose_start_time = time.time()
                                        color = (255, 255, 255)  # 하얀색
                                        
                                    elif time.time() - wrong_pose_start_time >= wrong_pose_delay:
                                        # 0.5초 이후 색상을 빨간색으로 변경
                                        print(f"{time.time() - wrong_pose_start_time >= wrong_pose_delay}")
                                        color = wrong_pose_color  # 빨간색

                                # 정확도 텍스트 출력
                                cv2.putText(frame, f"Similarity: {similarity:.2f}", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                
                                # 크기 조건 표시 - 이것 없어도 될거 같음
                                size_condition = "Size OK" if is_contour_smaller(current_contour, reference_fixed) else "Too Large"
                                cv2.putText(frame, f"Size: {size_condition}", (10, 150),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            break  # 첫 번째 사람만 처리

        frame_count += 1

        # 횟수 텍스트 출력
        cv2.putText(frame, f"Pose Count: {pose_count}/{current_ref.target_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        #이것도 없어도 될듯
        if frame_count < cooldown_frames:
            cv2.putText(frame, "Cooldown", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 남은 시간 표시 - 초만보여주면 될듯
        elapsed_time = time.time() - start_time
        remaining_time = max(0, time_per_image - elapsed_time)
        cv2.putText(frame, f"Time left: {remaining_time:.1f}s", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 4초가 지났는지 확인
        if elapsed_time >= time_per_image:
            current_ref_index = next_ref_index
            pose_count = 0
            start_time = time.time()  # 시간 초기화
            print(f"시간 초과, 다음 포즈로 이동: {references[current_ref_index].image_path}")
            wrong_pose_start_time = None

            
        # 출력할 이미지 사이즈와 출력 화면 합치기
        reference_resized = cv2.resize(current_ref_with_next, (frame.shape[1] // 2, frame.shape[0]))
        combined_frame = np.hstack((reference_resized, frame))
        # 화면 출력
        cv2.imshow('Pose Guide', combined_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            reference_scale = min(reference_scale + 0.05, 1.0)
        elif key == ord('-') or key == ord('_'):
            reference_scale = max(reference_scale - 0.05, 0.1)
        elif key == ord('w'):
            move_y -= 1
        elif key == ord('a'):
            move_x -= 1
        elif key == ord('s'):
            move_y += 1
        elif key == ord('d'):
            move_x += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()