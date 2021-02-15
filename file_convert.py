import cv2
import json
import os
import numpy as np
from multiprocessing import Process
import argparse


video_dir_path = "videos"
image_dir_path = "images"
json_path = "data/calibration.json"
suffix = ".mov"
framerate = 50

ffmpeg = 'ffmpeg.exe'



def load_camera(json_path):
    with open(json_path) as f:
        json_data = json.load(f)
    return json_data


def img2video(video_path, image_path, framerate):
    cmd = f'{ffmpeg} -i ./{image_path}/%d.jpg "{video_path}" -r {framerate}'
    os.system(cmd)


def video2img(video_path, image_path, framerate):
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    while cap.isOpened():
        file_path = os.path.join(image_path, str(frame_id) + ".jpg")
        ret, frame = cap.read()

        if os.path.exists(file_path):
            frame_id += 1
            continue

        cv2.imwrite(file_path, frame)
        frame_id += 1
        print(file_path)
    cap.release()


def process(args):
    json_data = load_camera(args.json_path)
    for cam_name in json_data.keys():
        video_path = os.path.join(args.video_dir_path, cam_name + args.suffix)
        image_path = os.path.join(args.image_dir_path, cam_name)

        selected = None
        if args.video2img:
            selected = video2img
        elif args.img2video:
            selected = img2video

        if args.multi_process:
            Process(target=selected, args=(video_path, image_path, args.framerate)).start()
        else:
            selected(video_path, image_path, args.framerate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_dir_path', type=str, default=video_dir_path)
    parser.add_argument('-i', '--image_dir_path', type=str, default=image_dir_path)
    parser.add_argument('-j', '--json_path', type=str, default=json_path)
    parser.add_argument('-s', '--suffix', type=str, default=suffix)
    parser.add_argument('-f', '--framerate', type=int, default=framerate)
    parser.add_argument('-m', '--multi_process', action='store_true')
    parser.add_argument('--video2img', action='store_true')
    parser.add_argument('--img2video', action='store_true')
    args = parser.parse_args()
    if not args.video2img and not args.img2video:
        print("Choose video2img or img2video.")
        exit(0)
    process(args)
