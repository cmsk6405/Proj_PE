# 각도
import numpy as np


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


def count_repetition_angle(angles, previous_state, tolerance=30):
    """
    각 변화량을 통해서 자세 반복 카운팅
    Args:
        angles (_type_): mp의 3개의 lm을 이용해서 각도 계산
        previous_state (_type_): ?
        tolerance (int, optional): 임계값

    Returns:
        _type_: _description_
    """
    current_state = previous_state.copy()
    flag = 0
    
    print(f"angles = {current_state}")
    for joint, angle in angles.items():
        print(f"joint = {joint}")
        print(f"angle = {angle}")

        if angle > (180 - tolerance):
            current_state[joint] = 1
        elif angle < tolerance:
            current_state[joint] = 0
        else:
            current_state[joint] = 0

    
    if current_state != previous_state:
        flag = 1
    
    print(f"current_state = {current_state}")

    return current_state, flag



# 이차함수
def count_repetition_func(previous_pose, current_pose, previous_state, flag, tolerance=30):
    """
    이차 함수의 좌표를 통해서 자세 반복 카운팅

    Args:
        previous_pose (_type_): mp의 lm 33개
        current_pose (_type_): mp의 lm 33개
        previous_state (_type_): ?
        flag (_type_): ?
        tolerance (int, optional): 임계값

    Returns:
        _type_: _description_
    """
    if current_pose is None or len(current_pose) == 0:
        return previous_pose, previous_state, flag
    else:
        current_state = previous_state.copy()
        sdx, sdy = 0, 0
        
        for i in range(33):  # MediaPipe는 33개의 랜드마크를 사용합니다
            dx = current_pose[i].x - previous_pose[i].x
            dy = current_pose[i].y - previous_pose[i].y
            
            if abs(dx) < tolerance / 100:  # tolerance를 100으로 나누어 0~1 스케일에 맞춥니다
                dx = 0
            if abs(dy) < tolerance / 100:
                dy = 0
            
            sdx += dx
            sdy += dy
        
        if sdx > (tolerance * 3 / 100):
            current_state[0] = 1
        elif sdx < (tolerance * -3 / 100):
            current_state[0] = 0
        if sdy > (tolerance * 3 / 100):
            current_state[1] = 1
        elif sdy < (tolerance * -3 / 100):
            current_state[1] = 0
        
        if current_state != previous_state:
            flag = (flag + 1) % 2
        
        return current_pose, current_state.copy(), flag