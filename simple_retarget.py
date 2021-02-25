import numpy as np
from bvhtoolbox import BvhTree, get_motion_data, set_motion_data, get_euler_angles
from scipy.spatial.transform import Rotation as R
import math
import argparse


source_path = 'data/pure_rgb.bvh'
target_path = 'data/pure_imu.bvh'
output_path = 'data/retargeted.bvh'
retarget_trans = True

class Bvh_tree_mod(BvhTree):
    def __init__(self, data):
        super(Bvh_tree_mod, self).__init__(data)
        # Rename End Sites.
        end_sites = self.search('End')
        for end in end_sites:
            end.value[1] = end.parent.name + "_End"

    def scaling_joints_offset(self, scale):

        def update_node(node):
            for child in node:
                if child.value[0] == 'OFFSET':
                    child.value[1] = str(float(child.value[1]) * scale)
                    child.value[2] = str(float(child.value[2]) * scale)
                    child.value[3] = str(float(child.value[3]) * scale)
                update_node(child)

        update_node(self.root)

    def _get_joint_string(self, joint):
        depth = self.get_joint_depth(joint.name)

        if not self.joint_children(joint.name):
            s = '{0}{1}\n'.format('  ' * depth, 'End Site')
            s += '{0}{{\n'.format('  ' * depth)
            s += '{0}{1} {2}\n'.format('  ' * (depth + 1), 'OFFSET', ' '.join(joint['OFFSET']))
            s += '{0}}}\n'.format('  ' * depth)
        else:
            s = '{0}{1}\n'.format('  ' * depth, str(joint))
            s += '{0}{{\n'.format('  ' * depth)
            for attribute in ['OFFSET', 'CHANNELS']:
                try:
                    s += '{0}{1} {2}\n'.format('  ' * (depth + 1), attribute, ' '.join(joint[attribute]))
                except:
                    print("no " + attribute + " for " + joint.name)
        return s


def get_rotations(bvh_tree, joint_name, channel_order):
    eulers = get_euler_angles(bvh_tree, joint_name, channel_order.lower())
    rotations = R.from_euler(channel_order, eulers, degrees=True)
    return rotations


retarget_dict = {
    "Hips": "root_tx",
    "RightUpLeg": "right_hip_rx",
    "RightLeg": "right_knee_rx",
    "RightFoot": "right_ankle_rx",
    "LeftUpLeg": "left_hip_rx",
    "LeftLeg": "left_knee_rx",
    "LeftFoot": "left_ankle_rx",

    "Spine": "spine_2_rx",
    "Spine1": "spine_3_rx",
    "Spine2": "spine_4_rx",
    "Neck": None,
    "Neck1": "neck_rx",
    "Head": "head_rx",

    "RightShoulder": "right_clavicle_rx",
    "RightArm": "right_shoulder_rx",
    "RightForeArm": "right_elbow_rx",
    "RightHand": "right_lowarm_rx",

    "LeftShoulder": "left_clavicle_rx",
    "LeftArm": "left_shoulder_rx",
    "LeftForeArm": "left_elbow_rx",
    "LeftHand": "left_lowarm_rx",
}


def bvh_modify(args):
    with open(args.source_path) as f:
        source_bvh = Bvh_tree_mod(f.read())
    with open(args.target_path) as f:
        target_bvh = Bvh_tree_mod(f.read())

    source_frames = get_motion_data(source_bvh)
    target_frames = get_motion_data(target_bvh)
    length = min(source_frames.shape[0], target_frames.shape[0])
    source_frames = source_frames[:length]
    target_frames = target_frames[:length]

    source_root_name = source_bvh.get_joints_names()[0]
    target_root_name = target_bvh.get_joints_names()[0]

    source_channel_names = source_bvh.joint_channels(source_root_name)
    target_channel_names = target_bvh.joint_channels(target_root_name)

    source_channel_order = ''.join([channel[:1].upper() for channel in source_channel_names if channel.endswith("rotation")])
    target_channel_order = ''.join([channel[:1].upper() for channel in target_channel_names if channel.endswith("rotation")])

    if not args.ignore_trans:
        source_channels_idx = source_bvh.get_joint_channels_index(source_root_name)
        target_channels_idx = target_bvh.get_joint_channels_index(target_root_name)
        source_pos_channels_idx = source_channels_idx + source_bvh.get_joint_channel_index(source_root_name, source_channel_names[0])
        target_pos_channels_idx = target_channels_idx + target_bvh.get_joint_channel_index(target_root_name, target_channel_names[0])
        target_frames[:, target_pos_channels_idx:target_pos_channels_idx + 3] = source_frames[:, source_pos_channels_idx:source_pos_channels_idx + 3]

    for target_joint, source_joint in retarget_dict.items():
        target_channels_idx = target_bvh.get_joint_channels_index(target_joint)
        target_rot_channels_idx = target_channels_idx + target_bvh.get_joint_channel_index(target_joint, target_channel_names[3])

        if source_joint is None:
            target_frames[:, target_rot_channels_idx:target_rot_channels_idx + 3] = 0
            continue

        source_joint_rotations = get_rotations(source_bvh, source_joint, source_channel_order)[:length]
        target_frames[:, target_rot_channels_idx:target_rot_channels_idx + 3] = source_joint_rotations.as_euler(target_channel_order, degrees=True)

    set_motion_data(target_bvh, target_frames)

    target_bvh.write_file(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_path', type=str, default=source_path)
    parser.add_argument('-t', '--target_path', type=str, default=target_path)
    parser.add_argument('-o', '--output_path', type=str, default=output_path)
    parser.add_argument('-i', '--ignore_trans', action='store_true')
    args = parser.parse_args()
    bvh_modify(args)
