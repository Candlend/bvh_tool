import math


translation_list = ["root_tx", "root_ty", "root_tz"]

input_joint_list = [
    "root_tx", "root_ty", "root_tz",
    "root_rx", "root_ry", "root_rz",
    "left_hip_rx", "left_hip_ry", "left_hip_rz",
    "right_hip_rx", "right_hip_ry", "right_hip_rz",
    "spine_2_rx", "spine_2_ry", "spine_2_rz",
    "left_knee_rx", "left_knee_ry", "left_knee_rz",
    "right_knee_rx", "right_knee_ry", "right_knee_rz",
    "spine_3_rx", "spine_3_ry", "spine_3_rz",
    "left_ankle_rx", "left_ankle_ry", "left_ankle_rz",
    "right_ankle_rx", "right_ankle_ry", "right_ankle_rz",
    "spine_4_rx", "spine_4_ry", "spine_4_rz",
    "left_foot_rx", "left_foot_ry", "left_foot_rz",
    "right_foot_rx", "right_foot_ry", "right_foot_rz",
    "neck_rx", "neck_ry", "neck_rz",
    "left_clavicle_rx", "left_clavicle_ry", "left_clavicle_rz",
    "right_clavicle_rx", "right_clavicle_ry", "right_clavicle_rz",
    "head_rx", "head_ry", "head_rz",
    "left_shoulder_rx", "left_shoulder_ry", "left_shoulder_rz",
    "right_shoulder_rx", "right_shoulder_ry", "right_shoulder_rz",
    "left_elbow_rx", "left_elbow_ry", "left_elbow_rz",
    "right_elbow_rx", "right_elbow_ry", "right_elbow_rz",
    "left_lowarm_rx", "left_lowarm_ry", "left_lowarm_rz",
    "right_lowarm_rx", "right_lowarm_ry", "right_lowarm_rz",
    "left_hand_rx", "left_hand_ry", "left_hand_rz",
    "right_hand_rx", "right_hand_ry", "right_hand_rz",
]


output_joint_list = [
    "root_tx", "root_ty", "root_tz",
    "root_rx", "root_ry", "root_rz",
    "left_hip_rx", "left_hip_ry", "left_hip_rz",
    "left_knee_rx", "left_knee_ry", "left_knee_rz",
    "left_ankle_rx", "left_ankle_ry", "left_ankle_rz",
    "left_foot_rx", "left_foot_ry", "left_foot_rz",
    "right_hip_rx", "right_hip_ry", "right_hip_rz",
    "right_knee_rx", "right_knee_ry", "right_knee_rz",
    "right_ankle_rx", "right_ankle_ry", "right_ankle_rz",
    "right_foot_rx", "right_foot_ry", "right_foot_rz",
    "spine_2_rx", "spine_2_ry", "spine_2_rz",
    "spine_3_rx", "spine_3_ry", "spine_3_rz",
    "spine_4_rx", "spine_4_ry", "spine_4_rz",
    "neck_rx", "neck_ry", "neck_rz",
    "head_rx", "head_ry", "head_rz",
    "left_clavicle_rx", "left_clavicle_ry", "left_clavicle_rz",
    "left_shoulder_rx", "left_shoulder_ry", "left_shoulder_rz",
    "left_elbow_rx", "left_elbow_ry", "left_elbow_rz",
    "left_lowarm_rx", "left_lowarm_ry", "left_lowarm_rz",
    "left_hand_rx", "left_hand_ry", "left_hand_rz",
    "right_clavicle_rx", "right_clavicle_ry", "right_clavicle_rz",
    "right_shoulder_rx", "right_shoulder_ry", "right_shoulder_rz",
    "right_elbow_rx", "right_elbow_ry", "right_elbow_rz",
    "right_lowarm_rx", "right_lowarm_ry", "right_lowarm_rz",
    "right_hand_rx", "right_hand_ry", "right_hand_rz",
]


def get_order(elem):
    return elem


def motion2bvh(input_path, output_path, framerate=30):
    with open(input_path, "r") as f:
        input_lines = f.readlines()
    output_lines = []
    for line in input_lines:
        pose = list(map(float, list(filter(lambda x: x, line.split(" ")))))
        new_pose = []
        for i in range(len(pose)):
            if output_joint_list[i] in translation_list:
                new_pose.append(pose[input_joint_list.index(output_joint_list[i])] * 1000)
            else:
                new_pose.append(math.degrees(pose[input_joint_list.index(output_joint_list[i])]))
        new_line = " ".join(list(map(str, new_pose))) + "\n"
        output_lines.append(new_line)
        output_lines.append(new_line)
    with open(output_path, "w") as f:
        with open("hierarchy.txt", "r") as h:
            f.write(h.read())
        f.write("\n")
        f.write("MOTION\n")
        f.write("Frames: %d\n" % len(output_lines))
        f.write("Frame Time: %f\n" % (1 / framerate))
        f.writelines(output_lines)
    pass


if __name__ == '__main__':
    input_path = "C:\\Projects\\Mocap\\output\\0.motion"
    output_path = "0_1.bvh"
    framerate = 60
    motion2bvh(input_path, output_path, framerate)
