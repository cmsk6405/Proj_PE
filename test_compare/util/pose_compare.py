from typing import Dict, List, Tuple, Optional
import mediapipe as mp

import cv2
import numpy as np

from util.count import count_repetition_angle, count_repetition_func
from util.helpers import (
    get_angle,
    kpts_angle,
)



class PoseCompare:

    def __init__(self) -> None:
        None

    def inference(self, frame,  yolo, pose, mp_pose) -> Tuple[Dict, Dict]:
        """
        create_model에서 생성된 모델을 전달받아서 frame들을 모델에 넣고 실행
        get_angle를 통해서 각 부분의 각도를 계산

        Args:
            frame (_type_): _description_
            yolo (_type_): _description_
            pose (_type_): _description_
            mp_pose (_type_): _description_

        Returns:
            angles(dict) = 자세의 각도 계산결과
        """

        # yolo result
        yolo_results = yolo(frame)[0]
        mp_drawing = mp.solutions.drawing_utils

        # 안에 있을때 yolo가 object를 찾지 못했을때 에러 발생하므로 밖에서 선언
        angles = {}

        for yolo_result in yolo_results.boxes.data:
            # 신뢰도, 분류 클래스
            x1, y1, x2, y2, conf, cls = yolo_result
            if int(cls) == 0:  # 사람 클래스일 경우
                # 바운딩 박스 영역 추출
                # None type 일 때 어떻게 하면 좋을까? None type일까?
                person_img = frame[int(y1):int(y2), int(x1):int(x2)]

                # MediaPipe를 사용하여 포즈 추정
                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                # media pipe result
                mp_result = pose.process(person_rgb)
                # TODO: 추후 삭제 필요
                # 원본 프레임에 포즈 랜드마크 및 연결선 그리기
                if mp_result.pose_landmarks:
                    mp_drawing.draw_landmarks(frame[int(y1):int(y2), int(x1):int(x2)], 
                                            mp_result.pose_landmarks, 
                                            mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))
                # Calculate angle
                if mp_result.pose_landmarks != None:
                    for k, v in kpts_angle.items():
                        angles[k] = get_angle(mp_result.pose_landmarks.landmark, v)

        return angles



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
        """
		compare_img = pose.draw_compare(fps=fps, offset=20)
        웹캠과 비디오를 받아서 하나로 만들기
        각도 계산 결과 하단에 출력하기
        fps 없어도 됨
        offset 설정만 있게
        """
        # 그냥 self.tensor_ref로 불러도될것같음
        ref_frame, ref_angle = self.draw_test(trgt="ref")
        trgt_frame, trgt_angle = self.draw_test(trgt="trgt")


        #TODO: all ok 조건 생각 - 현재는 모든 부분의 각도가 일정 오차 내여야만 all ok가 출력 - 60 ~ 70 퍼센트
        #8개 항목중 5개 이상이면 o, 3~4개 세모,  1~2 x - ❌⭕🔺🔼⏺️🔺, ▲ ● ⨉
        # ---- Maximum Angle Diff Calculation ---- #
        # Difference in angle - 동영상과 웹캠의 각도 차이 계산
        angle_diff = self.calc_angle_diff()




        # 횟수 세기
        # person_states = kpts_angle.copy()
        # current_state, flag = count_repetition_angle(self.angle_ref, person_states)
        # person_states = current_state
        



        # TODO: 굳이 출력할 필요없는 문장이라고 생각
        angle_diff_str = [f"Maximum Angle Diff: {offset}"]

        all_ok = ""
        ok_cnt = 0
        for k, v in angle_diff.items():
            if abs(v) < offset:
                status = f"OK ({v})"
                ok_cnt += 1
            else:
                status = f"NOT OK ({v})"
                # all_ok = False
        if ok_cnt >= 5:
            print("●")
            all_ok = "O"
        elif 3 <= ok_cnt and ok_cnt <= 4:
            print("▲")
            all_ok = "triangle"
        else:
            print("⨉")
            all_ok = "X"

            angle_diff[k] = status
        # 각도 계산 결과 str 만들기 angle_diff_str이 리스트인데 굳이 join안쓰고 첨부터 str이면 될듯?
        angle_diff_str.extend([f"{k}: {v}" for k, v in angle_diff.items()])
        angle_diff_str.append(f"ALL OK: {all_ok}")
        angle_diff_str = "\n".join(angle_diff_str)

        # 출력부 설정
        # 영상 출력
        ref_frame = cv2.resize(ref_frame, (640, 480))
        trgt_frame = cv2.resize(trgt_frame, (640, 480))

        # 비디오와 웹캠 합치기
        hcombine_frame = np.hstack((ref_frame,trgt_frame))

        # 텍스트 출력부
        # 이 방법이 아니면 PIL을 사용하여야 함
        text_frame = np.zeros((400, 1280, 3), dtype=np.uint8)#h*w(웹캠과 비디오 w의 합)
        # 여러 줄의 텍스트를 개별적으로 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # 흰색 텍스트
        line_type = 2

        y0, dy = 30, 30  # 첫 번째 줄의 y 위치와 각 줄 간의 간격 설정
        for i, line in enumerate(angle_diff_str.split('\n')):
            y = y0 + i * dy
            cv2.putText(text_frame, line, (10, y), font, font_scale, font_color, line_type)

        hvcombined_frame = np.vstack((hcombine_frame, text_frame))

        return hvcombined_frame


    def load_img(self, frame, model, dest: str):
        """
        영상의 frame을 읽고 모델에 넣어 실행후 결과를 저장
        frame = 영상의 frame
        model = yolo, mp, mp 설정
        dest = 입력값의 ref(비디오), trgt(웹캠) 설정
        """
        angles = self.inference(frame, *model)

        # Assign image and output to the relevant spot.
        if dest == "trgt":
            self.angle_trgt = angles
            self.tensor_trgt = frame
        else:
            self.angle_ref = angles
            self.tensor_ref = frame



    def calc_angle_diff(self) -> Dict[str, int]:
        """
        동영상과 웹캠간의 자세의 각도차이 계산
        angle_diff = 각 부분별 각도 차이지만 mp가 사람을 찾지 못했을때 각 부분의 각도 값을 0으로 처리
        Returns:
                angle_diff(dict) = 동영상과 웹캠간의 자세의 각도차이 계산결과
        """
        if len(self.angle_ref) == len(self.angle_trgt):
            angle_diff = {k: self.angle_ref[k] - self.angle_trgt[k] for k in self.angle_ref}
        else:
            angle_diff = {'right_shoulder': 0, 'right_arm': 0, 'left_shoulder': 0, 'left_arm': 0, 'right_hip': 0, 'right_leg': 0, 'left_hip': 0, 'left_leg': 0}

        return angle_diff