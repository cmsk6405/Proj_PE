from typing import Dict, List, Tuple, Optional
import mediapipe as mp

import cv2
import numpy as np

from util.count import count_repetition_angle, count_repetition_func
from util.helpers import (
    get_angle,
    kpts_angle,
    joint_pairs
)


from collections import defaultdict

from fastdtw import fastdtw
from scipy.spatial.distance import cosine


class PoseCompare:

    def __init__(self) -> None:
        # None
        self.person_states = defaultdict(lambda: [2, 2])
        self.person_reps = defaultdict(int)
        self.person_previous_poses = {}
        self.person_flags = defaultdict(lambda: -1)


        self.person_states2 = defaultdict(lambda: {joint[0]: 2 for joint in joint_pairs})
        self.person_reps2 = defaultdict(int)

        self.landmarks_ref = None
        self.landmarks_trgt = None
        


    def inference(self, frame,  yolo, pose, mp_pose, dest) -> Tuple[Dict, Dict]:

        # yolo result
        yolo_results = yolo(frame)[0]
        mp_drawing = mp.solutions.drawing_utils

        angles = {}
        landmarks = None
        bbox = 0,0,0,0
        for i, yolo_result in enumerate(yolo_results.boxes.data):
            # 신뢰도, 분류 클래스
            x1, y1, x2, y2, conf, cls = yolo_result
            x1, y1, x2, y2, cls = map(int, [x1, y1, x2, y2, cls])
            bbox = x1, y1, x2, y2

            if cls == 0:  # 사람 클래스일 경우
                # 바운딩 박스 영역 추출
                # 1번. 웹캠이 timeout일 떄 발생
                person_img = frame[y1:y2, x1:x2]

                # MediaPipe를 사용하여 포즈 추정
                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                # media pipe result
                mp_result = pose.process(person_rgb)
                # TODO: 추후 삭제 필요
                # 원본 프레임에 포즈 랜드마크 및 연결선 그리기
                if mp_result.pose_landmarks:
                    mp_drawing.draw_landmarks(frame[y1:y2, x1:x2], 
                                            mp_result.pose_landmarks, 
                                            mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
                    
                    landmarks = mp_result.pose_landmarks.landmark

                    #횟수용
                    if dest =="trgt":
                        person_id = f"person_{i}"
                        if person_id not in self.person_previous_poses:
                            self.person_previous_poses[person_id] = landmarks

                        # 함수 횟수 세기
                        previous_pose, current_state, flag = count_repetition_func(
                            self.person_previous_poses[person_id], 
                            landmarks,
                            self.person_states[person_id],
                            self.person_flags[person_id]
                        )
                        self.person_previous_poses[person_id] = previous_pose
                        self.person_states[person_id] = current_state 
                        self.person_flags[person_id] = flag

                        if flag == 1:
                            self.person_reps[person_id] += 1
                            self.person_flags[person_id] = -1

                        text = f"angle: {self.person_reps[person_id]}, "
                        # cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            


                # Calculate angle
                if mp_result.pose_landmarks != None:
                    for k, v in kpts_angle.items():
                        angles[k] = get_angle(landmarks, v)

                # 각도횟수 세기
                    if dest =="trgt":
                        # print(f"self.person_states2[person_id] ===== {self.person_states2[person_id]}")
                        current_state2, flag2 = count_repetition_angle(angles, self.person_states2[person_id])
                        self.person_states2[person_id] = current_state2
                        if flag2:
                            self.person_reps2[person_id] += 1
                        
                        text += f"func == {self.person_reps2[person_id]}"
                        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        print(f"x1, y1 ====={x1, y1}")


        return angles, bbox, landmarks


    #TODO: 함수 이름 추후 변경 필요
    def draw_test(self, trgt: str = "ref"):
        #이 함수 꼭 필요하지 않은 것 같음 추후 확인 후 삭제

        if trgt == "ref":
            angle = self.angle_ref
            frame = self.tensor_ref
        else:
            angle = self.angle_trgt
            frame = self.tensor_trgt
        return frame, angle

    def compare(self, offset:int):
        #TODO: 지금 비디오와 캠화면 하나로 합치는 부분과 결과ok 부분이 합쳐있는데 분리해야 할수도 있을듯

        # 그냥 self.tensor_ref로 불러도될것같음
        ref_frame, ref_angle = self.draw_test(trgt="ref")
        trgt_frame, trgt_angle = self.draw_test(trgt="trgt")


        #TODO: all ok 조건 생각 - 현재는 모든 부분의 각도가 일정 오차 내여야만 all ok가 출력 - 60 ~ 70 퍼센트
        #8개 항목중 5개 이상이면 o, 3~4개 세모,  1~2 x - ❌⭕🔺🔼⏺️🔺, ▲ ● ⨉
        # ---- Maximum Angle Diff Calculation ---- #
        # Difference in angle - 동영상과 웹캠의 각도 차이 계산
        angle_diff = self.calc_angle_diff()

        similarity = self.calculate_similarity(self.landmarks_ref, self.landmarks_trgt)

        # TODO: 굳이 출력할 필요없는 문장이라고 생각
        angle_diff_str = []#f"Maximum Angle Diff: {offset}"

        all_ok = ""
        ok_cnt = 0
        for k, v in angle_diff.items():
            if abs(v) < offset:
                ok_cnt += 1

        if ok_cnt >= 5:
            # print("●")
            all_ok = "O"
        elif 3 <= ok_cnt and ok_cnt <= 4:
            # print("▲")
            all_ok = "triangle"
        else:
            # print("⨉")
            all_ok = "X"
        # 각도 계산 결과 str 만들기 angle_diff_str이 리스트인데 굳이 join안쓰고 첨부터 str이면 될듯?
        # angle_diff_str.extend([f"{k}: {v}" for k, v in angle_diff.items()])
        angle_diff_str.append(f"ALL OK: {all_ok}")
        angle_diff_str = "\n".join(angle_diff_str)

        # 출력부 설정
        # 영상 출력
        ref_frame = cv2.resize(ref_frame, (960 , 720))
        trgt_frame = cv2.resize(trgt_frame, (960, 720))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (0, 0, 0)  # 흰색 텍스트
        line_type = 2
        cv2.putText(trgt_frame, angle_diff_str, (self.bbox_trgt[0], self.bbox_trgt[1]), font, font_scale, font_color, line_type)
        print(f"self.bbox_trgt[0], self.bbox_trgt[1])  ===  {self.bbox_trgt[0], self.bbox_trgt[1]}")
        # 비디오와 웹캠 합치기
        hcombine_frame = np.hstack((ref_frame,trgt_frame))

        return hcombine_frame
    

    def load_img(self, frame, model, dest: str):

        angles, bbox, landmarks = self.inference(frame, *model, dest)#, landmarks, bbox

        # Assign image and output to the relevant spot.
        if dest == "trgt":
            self.angle_trgt = angles
            self.tensor_trgt = frame
            self.bbox_trgt = bbox
            self.landmarks_trgt = landmarks

        else:
            self.angle_ref = angles
            self.tensor_ref = frame
            self.bbox_ref = bbox
            self.landmarks_ref = landmarks



    def calc_angle_diff(self) -> Dict[str, int]:
   
        if len(self.angle_ref) == len(self.angle_trgt):
            angle_diff = {k: self.angle_ref[k] - self.angle_trgt[k] for k in self.angle_ref}
        else:
            angle_diff = {'right_shoulder': 0, 'right_arm': 0, 'left_shoulder': 0, 'left_arm': 0, 'right_hip': 0, 'right_leg': 0, 'left_hip': 0, 'left_leg': 0}

        return angle_diff
    
    def calculate_similarity(self, landmarks1, landmarks2):
        if landmarks1 is None or landmarks2 is None:
            return 1  # 최대 오류값 반환
        lmList1 = [[lm.x, lm.y, lm.z] for lm in landmarks1]
        lmList2 = [[lm.x, lm.y, lm.z] for lm in landmarks2]
        error, _ = fastdtw(lmList1, lmList2, dist=cosine)
        return error