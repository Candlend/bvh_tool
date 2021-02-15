from bvhtoolbox import BvhTree, get_motion_data, set_motion_data, get_euler_angles
import torch
from torch import nn
import argparse


rgb_path = 'data/rgb_in_imu.bvh'
imu_path = 'data/imu_in_imu.bvh'
output_path = 'data/out_in_imu.bvh'
total_step = 100000
weight_0 = 100
weight_1 = 10
delta = 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GradLoss(nn.Module):
    def forward(self, a, b):
        loss = torch.sum(torch.pow(a - b, 2) / (torch.abs(b) + delta))
        return loss


def get_grad(input):
    grad = input[1:] - input[:-1]
    zeros = torch.Tensor([[0, 0, 0]]).to(device)
    tmp1 = torch.vstack((grad, zeros))
    tmp2 = torch.vstack((zeros, grad))
    grad = (tmp1 + tmp2) / 2
    return grad


mse = nn.MSELoss()
grad_criteria = GradLoss()


def get_loss(output_root_translation, imu_root_translation, rgb_root_translation):
    loss_0 = mse(output_root_translation, rgb_root_translation)
    output_first_order = get_grad(output_root_translation)
    imu_first_order = get_grad(imu_root_translation)
    loss_1 = grad_criteria(output_first_order, imu_first_order)
    return weight_0 * loss_0 + weight_1 * loss_1


def optimize(imu_root_translation, rgb_root_translation):
    output_root_translation = rgb_root_translation.clone().requires_grad_()

    optimizer = torch.optim.Adam([output_root_translation], lr=1e-2)
    for step in range(total_step):
        loss = get_loss(output_root_translation, imu_root_translation, rgb_root_translation)
        print("%s: %s" % (step, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return output_root_translation


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


def postprocess(args):
    with open(args.rgb_path) as f:
        rgb_bvh = Bvh_tree_mod(f.read())
    with open(args.imu_path) as f:
        imu_bvh = Bvh_tree_mod(f.read())

    rgb_frames = get_motion_data(rgb_bvh)
    imu_frames = get_motion_data(imu_bvh)
    root_name = rgb_bvh.get_joints_names()[0]
    channel_names = rgb_bvh.joint_channels(root_name)

    channels_idx = rgb_bvh.get_joint_channels_index(root_name)
    pos_channels_idx = channels_idx + rgb_bvh.get_joint_channel_index(root_name, channel_names[0])

    output_frames = rgb_frames.copy()

    rgb_root_translation = torch.Tensor(rgb_frames[:, pos_channels_idx:pos_channels_idx + 3]).to(device)
    imu_root_translation = torch.Tensor(imu_frames[:, pos_channels_idx:pos_channels_idx + 3]).to(device)

    target_root_translation = optimize(imu_root_translation, rgb_root_translation)

    output_frames[:, pos_channels_idx:pos_channels_idx + 3] = target_root_translation.cpu().detach().numpy()

    set_motion_data(rgb_bvh, output_frames)
    rgb_bvh.write_file(args.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rgb_path', type=str, default=rgb_path)
    parser.add_argument('-i', '--imu_path', type=str, default=imu_path)
    parser.add_argument('-o', '--output_path', type=str, default=output_path)
    parser.add_argument('--total_step', type=int, default=total_step)
    parser.add_argument('--weight_0', type=int, default=weight_0)
    parser.add_argument('--weight_1', type=int, default=weight_1)
    parser.add_argument('--delta', type=int, default=delta)

    args = parser.parse_args()

    total_step = total_step
    weight_0 = weight_0
    weight_1 = weight_1
    delta = delta

    postprocess(args)
