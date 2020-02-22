import argparse
import os
import glob
import cv2


def get_parser():
    parser = argparse.ArgumentParser(
        description="Inspect metrics.json files(output of detectron2 training)")
    parser.add_argument("--folder",
                        default="../data/ball_output/COCO-Detection",
                        help="folder where training outputs reside",
                        )
    return parser


def get_video_files(dir):
    return sorted(glob.glob(f"{dir}/*"))


def get_video_metadata(video_file, get_frame_rate):
    video = cv2.VideoCapture(video_file)

    fps = get_frame_rate(video)

    return fps


def frame_rate_getter(major_version):
    if int(major_version) < 3:
        return lambda video: video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        return lambda video: video.get(cv2.CAP_PROP_FPS)


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Find OpenCV version
    (major_version, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    get_frame_rate = frame_rate_getter(major_version)

    videos = get_video_files(args.folder)

    for video in videos:
        get_video_metadata(video)
