from utils import BVH, Animation
import numpy as np
import json
import cv2
import os
import argparse
from multiprocessing import Process


bvh_path = "data/input.bvh"
json_path = "data/calibration.json"
image_path = "images"
output_path = "output"
frame_offset = 145
frame_scale = 2
frame_start = 145
frame_end = 3097
t_scale = 0.169
down_sample = 4

position_offset = np.array([-0.00215498, -0.240341, 0.0286243])


def load_camera(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
    return json_data


def bvh_reproj_for_cam(args, cam_name, camera_data, positions, anim):
    for frame_id in range(args.frame_start, args.frame_end):
        save_path = os.path.join(args.output_path, cam_name, str(frame_id - args.frame_start) + ".jpg")
        if args.save_img:
            if os.path.exists(save_path):
                continue

        cur_positions = positions[(frame_id - args.frame_offset) * args.frame_scale] + position_offset
        # get cam info
        origin_K = np.array(camera_data[cam_name]['K']).reshape((3, 3))
        img_size = camera_data[cam_name]['imgSize']
        dist = np.array(camera_data[cam_name]["distCoeff"])
        K = cv2.getOptimalNewCameraMatrix(origin_K, dist, tuple(img_size), 0)[0]
        rt = np.array(camera_data[cam_name]['RT']).reshape(3, 4)
        r = rt[:3, :3].reshape((3, 3))
        t = rt[:3, 3] * args.t_scale

        cur_positions_cam = (r @ cur_positions.T).T + t
        cur_positions_img = (K @ cur_positions_cam.T).T
        cur_positions_img = cur_positions_img[:, :2] / np.tile(cur_positions_img[:, 2], (2, 1)).T

        lines = []
        for child, parent in enumerate(anim.parents):
            if parent >= 0:
                pt1 = tuple(map(int, cur_positions_img[child]))
                pt2 = tuple(map(int, cur_positions_img[parent]))
                lines.append((pt1, pt2))

        img = cv2.imread(os.path.join(args.image_path, cam_name, str(frame_id - args.frame_start) + ".jpg"))
        # img = cv2.imread(os.path.join(args.image_path, cam_name, str(frame_id).zfill(5) + ".jpg"))
        for line in lines:
            cv2.line(img, line[0], line[1], (255, 255, 255), 3)
        dir_path = os.path.join(args.output_path, cam_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if args.show_preview:
            cv2.imshow(cam_name, img[::args.down_sample, ::args.down_sample])
            cv2.waitKey(1)

        if args.save_img:
            cv2.imwrite(save_path, img)

        print(save_path)


def bvh_reproj(args):
    anim, names, frametime = BVH.load(args.bvh_path)
    positions = Animation.positions_global(anim)
    print(positions.shape)

    camera_data = load_camera(args.json_path)

    for cam_name in camera_data.keys():
        if args.multi_process:
            Process(target=bvh_reproj_for_cam, args=(args, cam_name, camera_data, positions, anim)).start()
        else:
            bvh_reproj_for_cam(args, cam_name, camera_data, positions, anim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bvh_path', type=str, default=bvh_path)
    parser.add_argument('-j', '--json_path', type=str, default=json_path)
    parser.add_argument('-i', '--image_path', type=str, default=image_path)
    parser.add_argument('-o', '--output_path', type=str, default=output_path)
    parser.add_argument('--frame_offset', type=int, default=frame_offset)
    parser.add_argument('--frame_scale', type=int, default=frame_scale)
    parser.add_argument('--frame_start', type=int, default=frame_start)
    parser.add_argument('--frame_end', type=int, default=frame_end)
    parser.add_argument('--down_sample', type=int, default=down_sample)
    parser.add_argument('--t_scale', type=float, default=t_scale)
    parser.add_argument('-p', '--show_preview', action='store_true')
    parser.add_argument('-s', '--save_img', action='store_true')
    parser.add_argument('-m', '--multi_process', action='store_true')
    args = parser.parse_args()
    if not args.show_preview and not args.save_img:
        print("Choose show_preview or save_img.")
        exit(0)
    bvh_reproj(args)
