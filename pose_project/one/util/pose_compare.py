from typing import Dict, List, Tuple, Optional
import mediapipe as mp

import cv2
import numpy as np

from util.count import count_repetition, count_repetition2
from util.helpers import (
    get_angle,
    kpts_angle,
    get_vec_angle,
    get_vec_angle2,
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
        self.person_reps = 0
        self.person_previous_poses = {}
        self.person_flags = -1 #defaultdict(lambda: -1)

        self.person_states2 = defaultdict(lambda: {joint[0]: 2 for joint in joint_pairs})
        self.person_reps2 = defaultdict(int)


    def inference(self, frame, pose, mp_pose, dest):
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
      
        mp_drawing = mp.solutions.drawing_utils

        # clear person info
        self.cam_person_info.clear()

        # find person info
       

        person_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result = pose.process(person_rgb)


        if mp_result.pose_landmarks:
            
            # if dest == "ref":
            # landmark drawing
            mp_drawing.draw_landmarks(frame, 
                                    mp_result.pose_landmarks, 
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            landmarks = mp_result.pose_landmarks.landmark


            if landmarks:
                landmarks_arr = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks])
                # landmarks_arr= 0

                # Calculate angle
                angles = {}
                vec = {}
                vec2 = {}
                if mp_result.pose_landmarks != None:
                    for k, v in kpts_angle.items():
                        angles[k] = get_angle(landmarks, v)
                        vec[k] = get_vec_angle(landmarks, v)
                        

                # save the info separately for webcam and video.
                if dest =="trgt":
                    self.cam_person_info = {
                                        'angles': angles,
                                        'landmarks': landmarks,
                                        'vec' : vec,
                                        }
                else:
                    self.video_person_info = {
                                        'angles': angles,
                                        'landmarks': landmarks,
                                        'vec' : vec,
                                        }
                    
            
            h, w, _ = frame.shape
            if landmarks[0]:
                self.test_x, self.test_y = int(landmarks[0].x * w), int(landmarks[0].y * h)



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


        if 'landmarks' in self.cam_person_info:
            if len(self.person_previous_poses) < 1:
                self.person_previous_poses = self.cam_person_info['landmarks']
            # 함수 횟수 세기
            previous_pose, current_state, flag = count_repetition(
                self.person_previous_poses, 
                self.cam_person_info['landmarks'],
                self.person_states,
                self.person_flags
            )
            self.person_previous_poses = previous_pose
            self.person_states = current_state 
            self.person_flags = flag

            if flag == 1:
                self.person_reps += 1
                self.person_flags = -1

            # h, w, _ = trgt_frame.shape
            # x, y = int(self.cam_person_info["landmarks"][0].x * w), int(self.cam_person_info["landmarks"][0].y * h)
            result_str = f"count: {self.person_reps} "
            # cv2.putText(trgt_frame, result_str, (self.test_x, self.test_y - 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(trgt_frame, result_str, (150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        




    def compare(self):
        """to use join angle to calculate angle difference

        Args:
            offset (int): Threshold for the difference between poses
        """

        trgt_frame = self.frame_trgt

        # 기존 x,y 각도 비교
        if 'angles' in self.cam_person_info:
            cam_person_angles = self.cam_person_info['angles']
            vid_person_angles = self.video_person_info['angles']

            # calculate angle difference
            angle_diff = self.calc_angle_diff(cam_person_angles, vid_person_angles)
            
            #offset
            offset1 = 15
            
            # # 결과 문자열 생성
            ok_cnt = sum(1 for v in angle_diff.values() if abs(v) < offset1)
            if ok_cnt == 8:
                all_ok = "O"
            elif 5 <= ok_cnt <= 7:
                all_ok = "triangle"
            else:
                all_ok = "X"
            
            result_str = f"angle: {all_ok}" 
            # cv2.putText(trgt_frame, result_str, (self.test_x, self.test_y - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(trgt_frame, result_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            print(f"angle_diff = {angle_diff}")
            


        # x,y,z 벡터 각도 비교1
        if 'vec' in self.cam_person_info:
            cam_vec_angle = self.cam_person_info['vec']
            vid_vec_angle = self.video_person_info['vec']
            
            #offset
            offset2 = 30
            
            # calculate angle difference
            vec_diff = self.calc_angle_diff(cam_vec_angle, vid_vec_angle)
            # # 결과 문자열 생성
            vec_ok_cnt = sum(1 for v in vec_diff.values() if abs(v) < offset2)
            if vec_ok_cnt == 8:
                vec_ok = "O"
            elif 5 <= vec_ok_cnt <= 7:
                vec_ok = "triangle"
            else:
                vec_ok = "X"
            

            vec_result_str = f"vector: {vec_ok}" 
            # cv2.putText(trgt_frame, result_str, (self.test_x, self.test_y - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(trgt_frame, vec_result_str, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            print(f"vec_diff = {vec_diff}")



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


    def calc_angle_diff(self, cam_angle, vid_angle) -> Dict[str, int]:
        """
        calculate each joint angle difference

        Args:
            cam_angle : joint angle of a specific part(kpts_angle)

        Returns:
            angle_diff: angle difference result
        """


        if len(vid_angle) == len(cam_angle):
            angle_diff = {k: vid_angle[k] - cam_angle[k] for k in vid_angle}

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