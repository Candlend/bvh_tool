import numpy as np
from bvhtoolbox import BvhTree, get_motion_data, set_motion_data, get_euler_angles
from scipy.spatial.transform import Rotation as R
import math
import argparse


input_path = 'data/out_in_imu.bvh'
mat_path = 'data/RGB2IMU_mat.npy'
output_path = 'data/out_in_rgb.bvh'


position_offset = np.array([0, -10, 0])


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


def bvh_modify(args):
    with open(args.input_path) as f:
        mocap = Bvh_tree_mod(f.read())
    rgb2imu = np.load(args.mat_path)
    imu2rgb = np.linalg.inv(rgb2imu)
    scale = 1 / math.sqrt(imu2rgb[0, 0] ** 2 + imu2rgb[0, 1] ** 2 + imu2rgb[0, 2] ** 2)
    imu2rgb_r = imu2rgb[:3, :3] * scale
    mocap.scaling_joints_offset(1/scale)

    root_name = mocap.get_joints_names()[0]

    frames = get_motion_data(mocap)
    channel_names = mocap.joint_channels(root_name)

    channel_order = ''.join([channel[:1].upper() for channel in channel_names if channel.endswith("rotation")])
    print(channel_order)

    joint_rotations = get_rotations(mocap, root_name, channel_order)
    rotation_offset = R.from_matrix(imu2rgb_r)

    new_rotations = rotation_offset * joint_rotations

    channels_idx = mocap.get_joint_channels_index(root_name)
    rot_channels_idx = channels_idx + mocap.get_joint_channel_index(root_name, channel_names[3])
    pos_channels_idx = channels_idx + mocap.get_joint_channel_index(root_name, channel_names[0])

    nonhomogen_a = frames[:, pos_channels_idx:pos_channels_idx + 3]
    nonhomogen_a += position_offset
    homogen_a = np.hstack((nonhomogen_a, np.ones((nonhomogen_a.shape[0], 1))))
    homogen_b = (imu2rgb @ homogen_a.T).T
    nonhomogen_b = homogen_b[:, :3] / (np.vstack((homogen_b[:, 3], homogen_b[:, 3], homogen_b[:, 3]))).T

    frames[:, pos_channels_idx:pos_channels_idx + 3] = nonhomogen_b
    frames[:, rot_channels_idx:rot_channels_idx + 3] = new_rotations.as_euler(channel_order, degrees=True)

    for joint_name in mocap.get_joints_names():
        if joint_name == root_name:
            continue
        channels_idx = mocap.get_joint_channels_index(joint_name)
        pos_channels_idx = channels_idx + mocap.get_joint_channel_index(joint_name, channel_names[0])
        frames[:, pos_channels_idx:pos_channels_idx + 3] /= scale

    set_motion_data(mocap, frames)

    mocap.write_file(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default=input_path)
    parser.add_argument('-m', '--mat_path', type=str, default=mat_path)
    parser.add_argument('-o', '--output_path', type=str, default=output_path)
    args = parser.parse_args()
    bvh_modify(args)
