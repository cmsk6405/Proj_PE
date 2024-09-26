import math
from typing import List, Tuple

# ==== Variable ==== #
# media pipe Keypoints mapping
kpts_name = {
    "0" :" nose",
    "1" : "left eye (inner)",
    "2" : "left eye",
    "3" : "left eye (outer)",
    "4" : "right eye (inner)",
    "5" : "right eye",
    "6" : "right eye (outer)",
    "7" : "left ear",
    "8" : "right ear",
    "9" : "mouth:(left)",
    "10" : "mouth:(right)",
    "11" : "left shoulder",
    "12" : "right shoulder",
    "13" : "left elbow",
    "14" : "right elbow",
    "15" : "left wrist",
    "16" : "right wrist",
    "17" : "left pinky",
    "18" : "right pinky",
    "19" : "left index",
    "20" : "right index",
    "21" : "left thumb",
    "22" : "right thumb",
    "23" : "left hip",
    "24" : "right hip",
    "25" : "left knee",
    "26" : "right knee",
    "27" : "left ankle",
    "28" : "right ankle",
    "29" : "left heel",
    "30" : "right heel",
    "31" : "left foot index",
    "32" : "right foot index",
}

# keypoints id required to calculate angle
kpts_angle = {
    "right_shoulder": [11, 12, 14],
    "right_arm": [12, 14, 16],
    "left_shoulder": [13, 11, 12],
    "left_arm": [15, 13, 11],
    "right_hip": [23, 24, 26],
    "right_leg": [24, 26, 28],
    "left_hip": [25, 23, 24],
    "left_leg": [23, 25, 27]
}

# ==== Functions ==== #
def get_angle(kpts_coord: List[Tuple[int, int]], angle_kpts: List[Tuple[int, int]]):
    """
    Calculate the joint angles using the landmarks of each joint required for pose comparison.

    Args:
        kpts_coord : landmarks
        angle_kpts : kpts_angle

    Returns:
        ang: calculated angle
    """

    # Get coords from the 3 points
    a = [kpts_coord[angle_kpts[0]].x, kpts_coord[angle_kpts[0]].y]
    b = [kpts_coord[angle_kpts[1]].x, kpts_coord[angle_kpts[1]].y]
    c = [kpts_coord[angle_kpts[2]].x, kpts_coord[angle_kpts[2]].y]
    

    # Calculate angle
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    # 양수로만 만들었을때 음수일때 360 더해주기만 하기
    # Sanity check, angle must be between 0-180
    ang = int(ang + 360 if ang < 0 else ang)
    ang = int(ang - 180 if ang > 270 else ang)

    print(f"ang = {ang}")

    return ang


import numpy as np

def get_vec_angle(kpts_coord: List[Tuple[int, int]], angle_kpts: List[Tuple[int, int]]):
    """
    Calculate the joint angles using the landmarks of each joint required for pose comparison.

    Args:
        kpts_coord : landmarks
        angle_kpts : kpts_angle

    Returns:
        ang: calculated angle
    """
    
     # 벡터 정의
    vector1 = np.array([kpts_coord[angle_kpts[0]].x - kpts_coord[angle_kpts[1]].x, kpts_coord[angle_kpts[0]].y -  kpts_coord[angle_kpts[1]].y,  kpts_coord[angle_kpts[0]].z -  kpts_coord[angle_kpts[1]].z])
    vector2 = np.array([kpts_coord[angle_kpts[2]].x - kpts_coord[angle_kpts[1]].x, kpts_coord[angle_kpts[2]].y -  kpts_coord[angle_kpts[1]].y,  kpts_coord[angle_kpts[2]].z -  kpts_coord[angle_kpts[1]].z])
    
    # 벡터 크기
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # 내적
    dot_product = np.dot(vector1, vector2)
    
    # 각도 계산 (라디안)
    angle_rad = np.arccos(dot_product / (magnitude1 * magnitude2))
    
    # 각도를 도(degree)로 변환
    angle_deg = np.degrees(angle_rad)
    
    print(f"angle_deg = {angle_deg}")
    return angle_deg
