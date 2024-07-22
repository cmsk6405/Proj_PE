from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision.io import read_image
from torchvision.ops import nms
from torchvision.transforms import functional as F

import mediapipe as mp
from ultralytics import YOLO

from util.helpers import (
    connect_skeleton,
    get_angle,
    # have_cuda,
    kpts_angle,
    # model,
    transforms,
)

# 추가 import
import cv2
import torch


class PoseCompare:

    def __init__(self) -> None:
        None

    def inference(self, frame,  yolo, pose, mp_pose) -> Tuple[Dict, Dict]:

        # Inference

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

        #TODO : 추후에 angles, angles 수정
        return angles


    def draw_one(self, trgt: str = "ref", include_angles: bool = True) -> Image:
        """Draw reference or target image

        Args:
                trgt (str): Target image, acceptable input: ["trgt" or "ref"]
                include_angles (bool): Write joint angles to image. Default to True
        """
        # ==== Setup ==== #
        # ---- Variables ---- #
        # Assign values based on target or reference image
        if trgt == "ref":
            angle = self.angle_ref
            img = self.img_ref
        else:
            angle = self.angle_trgt
            img = self.img_trgt

        # Setup Font
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font = ImageFont.truetype(font_path, size=20)

        # ---- Image ---- #
        # Copy image
        img = img.copy()
        # Establish draw object
        img_draw = ImageDraw.Draw(img)

        # Angles
        if include_angles:
            angle_str = "\n".join([f"{k}: {v}" for k, v in angle.items()])

            img_draw.text((0, 0), angle_str, font=font)

        return img


#TODO: 함수 이름 추후 변경 필요
    def draw_test(self, trgt: str = "ref"):
        """
        draw_trgt = self.draw_one(trgt="trgt", include_angles=False)
        draw_ref = self.draw_one(trgt="ref", include_angles=False)
        """
        if trgt == "ref":
            angle = self.angle_ref
            img = self.img_ref
            frame = self.tensor_ref
        else:
            angle = self.angle_trgt
            img = self.img_trgt
            frame = self.tensor_trgt
        return frame, angle

    def compare(self, offset:int):
        """
		compare_img = pose.draw_compare(fps=fps, offset=20)
        웹캠과 비디오를 받아서 하나로 만들기
        각도 계산 결과 하단에 출력하기
        fps 없어도 됨
        offset 설정만 있게
        """
        ref_frame, ref_angle = self.draw_test(trgt="ref")
        trgt_frame, trgt_angle = self.draw_test(trgt="trgt")


        #TODO: all ok 조건 생각 
        # ---- Maximum Angle Diff Calculation ---- #
        # Difference in angle
        angle_diff = self.calc_angle_diff()

        angle_diff_str = [f"Maximum Angle Diff: {offset}"]

        all_ok = True
        for k, v in angle_diff.items():
            if abs(v) < offset:
                status = f"OK ({v})"
            else:
                status = f"NOT OK ({v})"
                # If at least one is not OK then all_ok will be False
                all_ok = False

            angle_diff[k] = status
        # 각도 계산 결과 str 만들기
        angle_diff_str.extend([f"{k}: {v}" for k, v in angle_diff.items()])
        angle_diff_str.append(f"ALL OK: {all_ok}")
        angle_diff_str = "\n".join(angle_diff_str)
        print(f"최종 텍스트 출력 테스트{angle_diff_str}")

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


#TODO: PIL을 이용한 출력 방법인데 이 방법을 사용하지 않을거라면 삭제 하면 될 것 같음(draw_one을 포함)
    def draw_compare(self, fps: Optional[int] = None, offset: int = 20):
        """Draw comparison between reference and trgt

        Args:
                fps (int, Optional): Display FPS information. If set to None no FPS will be output. Default to None
                offset (int): Maximum Angle Diff. Default to 20
        """
        # ==== Setup ==== #
        draw = Image.new("RGB", (1200, 600))
        draw_ = ImageDraw.Draw(draw)
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font = ImageFont.truetype(font_path, size=20)

        # Draw inference
        draw_ref = self.draw_one(trgt="ref", include_angles=False)
        draw_trgt = self.draw_one(trgt="trgt", include_angles=False)

        print(f"============================{type(draw_ref)}")
        # Image Box
        ref_box = draw_ref.resize((640, 480))
        trgt_box = draw_trgt.resize((640, 480))

        # Angle text
        angle_ref = "\n".join([f"{k}: {v}" for k, v in self.angle_ref.items()])
        angle_trgt = "\n".join([f"{k}: {v}" for k, v in self.angle_trgt.items()])

        # ---- Maximum Angle Diff Calculation ---- #
        # Difference in angle
        angle_diff = self.calc_angle_diff()

        angle_diff_str = [f"Maximum Angle Diff: {offset}"]

        all_ok = True
        for k, v in angle_diff.items():
            if abs(v) < offset:
                status = f"OK ({v})"
            else:
                status = f"NOT OK ({v})"
                # If at least one is not OK then all_ok will be False
                all_ok = False

            angle_diff[k] = status

        angle_diff_str.extend([f"{k}: {v}" for k, v in angle_diff.items()])
        angle_diff_str.append(f"ALL OK: {all_ok}")
        angle_diff_str = "\n".join(angle_diff_str)

        # ==== Draw ==== #
        # Ref/Trgt to Image
        draw.paste(ref_box, (200, 150))
        draw.paste(trgt_box, (800, 150))

        # Straight Line in the middle
        draw_.line([(600, 0), (600, 600)])

        # Angles boxes
        # draw_.text((10, 400), angle_ref, font=font)
        # draw_.text((610, 400), angle_trgt, font=font)
        draw_.text((610, 0), angle_diff_str, font=font)

        # Who is who
        # draw_.text((400, 0), "Reference", font=font)
        # draw_.text((1000, 0), "Target", font=font)

        # FPS
        if fps is not None:
            draw_.text((1100, 580), f"FPS: {int(fps)}", font=font)

        return draw


    def load_img(self, frame, model, dest: str):
        #TODO : 정말 필요한 부분만 확인해서 남기기
        print(f"어디 img확인 - {dest}")
        angles = self.inference(frame, *model)

        # Assign image and output to the relevant spot.
        if dest == "trgt":
            self.angle_trgt = angles
            self.tensor_trgt = frame
            self.img_trgt = F.to_pil_image(frame)
        else:
            self.angle_ref = angles
            self.tensor_ref = frame
            self.img_ref = F.to_pil_image(frame)



    def calc_angle_diff(self) -> Dict[str, int]:
        """Calculate target and reference's angle difference

        Returns:
                A dictionary containing the joint as key, angle difference as the value
        """

        if len(self.angle_ref) == len(self.angle_trgt):
            print("안에 들어옴")
            angle_diff = {k: self.angle_ref[k] - self.angle_trgt[k] for k in self.angle_ref}
            print(f"-------------안에들어옴 - {angle_diff}")
        else:
            print("예외됨")
            angle_diff = {'right_shoulder': 0, 'right_arm': 0, 'left_shoulder': 0, 'left_arm': 0, 'right_hip': 0, 'right_leg': 0, 'left_hip': 0, 'left_leg': 0}
            print(f"-------------예외일때{angle_diff}")

        return angle_diff