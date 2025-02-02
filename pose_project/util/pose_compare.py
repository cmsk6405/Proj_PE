from typing import Dict, List, Tuple, Optional
import mediapipe as mp

import cv2
import numpy as np

from util.count import count_repetition, count_repetition2
from util.helpers import (
    get_angle,
    kpts_angle
)


from collections import defaultdict

from fastdtw import fastdtw
from scipy.spatial.distance import cosine


mp_pose = mp.solutions.pose

joint_pairs = [
    ('Left Shoulder', mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    ('Right Shoulder', mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    ('Left Elbow', mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    ('Right Elbow', mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    ('Left Hip', mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    ('Right Hip', mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    ('Left Knee', mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    ('Right Knee', mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
]


class PoseCompare:

    def __init__(self) -> None:

        # person info(bbox, angle, landmarks)
        self.cam_person_info = {}
        self.video_person_info = {}

        # 랜드마크로 횟수 세기
        self.person_states = defaultdict(lambda: [2, 2])
        self.person_reps = defaultdict(int)
        self.person_previous_poses = {}
        self.person_flags = defaultdict(lambda: -1)

        self.person_states2 = defaultdict(lambda: {joint[0]: 2 for joint in joint_pairs})
        self.person_reps2 = defaultdict(int)


    def inference(self, frame,  yolo, pose, mp_pose, dest):
        """
        using yolo, mediapipe track person track and find bbox, landmarks
        calculate joint angles
        save person information

        Args:
            frame : video, webcam frame
            yolo : yolov8l-seg.pt
            mp_pose : mp.solutions.pose
            pose : mp_pose.Pose
            dest : ref(video), trgt(webcam)
        """
        # track setting
        conf = 0.2
        iou = 0.5
        yolo_results = yolo.track(frame, persist=True, conf=conf, iou=iou, show=False, tracker="bytetrack.yaml")[0]
        mp_drawing = mp.solutions.drawing_utils

        # clear person info
        self.cam_person_info.clear()

        # find person info
        for yolo_result in (yolo_results.boxes.data):
            if len(yolo_result) == 7:
                x1, y1, x2, y2, id, _, cls = yolo_result
            else:
                x1, y1, x2, y2, _, cls = yolo_result
                id = 0
            x1, y1, x2, y2, id, cls = map(int, [x1, y1, x2, y2, id, cls])

            # cls = 0 is person
            if cls == 0:
                person_img = frame[y1:y2, x1:x2]
                person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                mp_result = pose.process(person_rgb)


                if mp_result.pose_landmarks:
                    
                    if dest == "ref":
                    # landmark drawing
                        mp_drawing.draw_landmarks(frame[y1:y2, x1:x2], 
                                                mp_result.pose_landmarks, 
                                                mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

                    landmarks = mp_result.pose_landmarks.landmark

                    # Calculate angle
                    angles = {}
                    if mp_result.pose_landmarks != None:
                        for k, v in kpts_angle.items():
                            angles[k] = get_angle(landmarks, v)

                    angles2 = {}
                    for joint_name, start, mid, end in joint_pairs:
                        start_point = np.array([landmarks[start].x, landmarks[start].y])
                        mid_point = np.array([landmarks[mid].x, landmarks[mid].y])
                        end_point = np.array([landmarks[end].x, landmarks[end].y])
                        
                        angle = self.calculate_angle(start_point, mid_point, end_point)
                        angles2[joint_name] = angle

                    # save the info separately for webcam and video.
                    person_id = f"person_{id}"
                    if dest =="trgt":
                        self.cam_person_info[person_id] = {
                                            'bbox': (x1, y1, x2, y2),
                                            'angles': angles,
                                            'landmarks': landmarks,
                                            'angles2': angles2,
                                            
                                            }
                    else:
                        self.video_person_info = {
                                            'bbox': (x1, y1, x2, y2),
                                            'angles': angles,
                                            'landmarks': landmarks
                                            }

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)


    def counting(self):
        """
        repetition pose counting and put result text
        """

        trgt_frame = self.frame_trgt        

        result_str = ""

        for person_id, result in self.cam_person_info.items():
            cam_x1, cam_y1, _, _ = result['bbox']
            cam_person_landmarks = result['landmarks']

            if person_id not in self.person_previous_poses:
                self.person_previous_poses[person_id] = cam_person_landmarks

            # 함수 횟수 세기
            previous_pose, current_state, flag = count_repetition(
                self.person_previous_poses[person_id], 
                cam_person_landmarks,
                self.person_states[person_id],
                self.person_flags[person_id]
            )
            self.person_previous_poses[person_id] = previous_pose
            self.person_states[person_id] = current_state 
            self.person_flags[person_id] = flag

            if flag == 1:
                self.person_reps[person_id] += 1
                self.person_flags[person_id] = -1

            
            result_str = f"count: {self.person_reps[person_id]} "
            cv2.putText(trgt_frame, result_str, (cam_x1, cam_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        
            



    def compare(self, offset:int):
        """to use join angle to calculate angle difference

        Args:
            offset (int): Threshold for the difference between poses
        """

        trgt_frame = self.frame_trgt
        
        for _, result in self.cam_person_info.items():

            cam_x1, cam_y1, _, _ = result['bbox']
            cam_person_angles = result['angles']            

            # calculate angle difference
            angle_diff = self.calc_angle_diff(cam_person_angles)
            # # 결과 문자열 생성
            ok_cnt = sum(1 for v in angle_diff.values() if abs(v) < offset)
            if ok_cnt >= 5:
                all_ok = "O"
            elif 3 <= ok_cnt <= 4:
                all_ok = "triangle"
            else:
                all_ok = "X"
            
            result_str = f"ALL OK: {all_ok}" 
            cv2.putText(trgt_frame, result_str, (cam_x1, cam_y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


            # if use fastdtw

            # similarity = self.calculate_similarity(result['landmarks'])

            # if similarity < 2.55:  
            #     all_ok = "O"
            # elif 2.55 <= similarity < 2.8:
            #     all_ok = "triangle"
            # else:  # 유사하지 않음
            #     all_ok = "X"

            # result_str = f"ALL OK: {all_ok}" 
            # cv2.putText(trgt_frame, result_str, (cam_x1, cam_y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            



    def load_img(self, frame, model, dest: str):
        """
        inference and save frames separately for video and webcam

        Args:
            frame : video or webcam frame
            model : yolo, mp, mp_pose
            dest : ref(video), trgt(webcam)
        """

        # inference
        self.inference(frame, *model, dest)

        # save frame
        if dest =="trgt":
            self.frame_trgt = frame
        else:
            self.frame_ref = frame


    def calc_angle_diff(self, cam_angle) -> Dict[str, int]:
        """
        calculate each joint angle difference

        Args:
            cam_angle : joint angle of a specific part(kpts_angle)

        Returns:
            angle_diff: angle difference result
        """

        video_angle = self.video_person_info['angles']

        if len(video_angle) == len(cam_angle):
            angle_diff = {k: video_angle[k] - cam_angle[k] for k in video_angle}

        return angle_diff


    def calculate_similarity(self, cam_landmarks):
        """
        Pose comparison using fastdtw

        Args:
            cam_landmarks: webcam person landmarks

        Returns:
            distance: dtw result(the lower, the better the result)
        """
        vid_landmarks = self.video_person_info["landmarks"]
        if vid_landmarks is None or cam_landmarks is None:
            return 1  # 최대 오류값 반환
        
        lmList1 = [[lm.x, lm.y, lm.z] for lm in vid_landmarks]
        lmList2 = [[lm.x, lm.y, lm.z] for lm in cam_landmarks]
        distance, _ = fastdtw(lmList1, lmList2, dist=cosine)

        return distance