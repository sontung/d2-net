import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

import imageio

import torch
import glob
from tqdm import tqdm

import scipy.io
import scipy.misc
import cv2
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale
from PIL import Image


def run_extraction(keypoint_only=False):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Argument parsing
    parser = argparse.ArgumentParser(description="Feature extraction script")

    parser.add_argument(
        "--image_list_file",
        type=str,
        required=True,
        help="path to a file containing a list of images to process",
    )

    parser.add_argument(
        "--preprocessing",
        type=str,
        default="caffe",
        help="image preprocessing (caffe or torch)",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="models/d2_tf.pth",
        help="path to the full model",
    )

    parser.add_argument(
        "--max_edge", type=int, default=1600, help="maximum image size at network input"
    )
    parser.add_argument(
        "--max_sum_edges",
        type=int,
        default=2800,
        help="maximum sum of image sizes at network input",
    )

    parser.add_argument(
        "--output_extension",
        type=str,
        default=".d2-net",
        help="extension for the output",
    )
    parser.add_argument(
        "--output_type", type=str, default="npz", help="output file type (npz or mat)"
    )

    parser.add_argument(
        "--multiscale",
        dest="multiscale",
        action="store_true",
        help="extract multiscale features",
    )
    parser.set_defaults(multiscale=False)

    parser.add_argument(
        "--no-relu",
        dest="use_relu",
        action="store_false",
        help="remove ReLU after the dense feature extraction module",
    )
    parser.set_defaults(use_relu=True)

    args = parser.parse_args()

    # Creating CNN model
    model = D2Net(model_file=args.model_file, use_relu=args.use_relu, use_cuda=use_cuda)

    # Process the file
    lines = ["/home/sontung/work/d2-net/qualitative/images/pair_1/2.jpg"]
    for line in tqdm(lines, total=len(lines)):
        path = line.strip()

        image = imageio.imread(path)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
        resized_image = image
        if max(resized_image.shape) > args.max_edge:
            print("resize")
            pil_img = Image.fromarray(resized_image)
            scale = args.max_edge / max(resized_image.shape)
            pil_img = pil_img.resize(
                (
                    int(resized_image.shape[1] * scale),
                    int(resized_image.shape[0] * scale),
                )
            )
            resized_image = np.array(pil_img).astype("float")

        if sum(resized_image.shape[:2]) > args.max_sum_edges:
            resized_image = scipy.misc.imresize(
                resized_image, args.max_sum_edges / sum(resized_image.shape[:2])
            ).astype("float")

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(resized_image, preprocessing=args.preprocessing)

        with torch.no_grad():
            if args.multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device,
                    ),
                    model,
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device,
                    ),
                    model,
                    scales=[1],
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]

        if keypoint_only:
            return keypoints

        # img = resized_image.astype("uint8")
        # for x, y, _ in keypoints:
        #     x, y = map(int, (x, y))
        #     # cv2.circle(img, (y, x), 5, (0, 0, 255), -1)
        #     cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        # img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        # cv2.imshow("t", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        if args.output_type == "npz":
            with open(path + args.output_extension, "wb") as output_file:
                np.savez(
                    output_file,
                    keypoints=keypoints,
                    scores=scores,
                    descriptors=descriptors,
                )
        elif args.output_type == "mat":
            with open(path + args.output_extension, "wb") as output_file:
                scipy.io.savemat(
                    output_file,
                    {
                        "keypoints": keypoints,
                        "scores": scores,
                        "descriptors": descriptors,
                    },
                )
        else:
            raise ValueError("Unknown output type.")


def load_model():
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Argument parsing
    parser = argparse.ArgumentParser(description="Feature extraction script")

    parser.add_argument(
        "--preprocessing",
        type=str,
        default="caffe",
        help="image preprocessing (caffe or torch)",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="models/d2_tf.pth",
        help="path to the full model",
    )

    parser.add_argument(
        "--max_edge", type=int, default=1600, help="maximum image size at network input"
    )
    parser.add_argument(
        "--max_sum_edges",
        type=int,
        default=2800,
        help="maximum sum of image sizes at network input",
    )

    parser.add_argument(
        "--output_extension",
        type=str,
        default=".d2-net",
        help="extension for the output",
    )
    parser.add_argument(
        "--output_type", type=str, default="npz", help="output file type (npz or mat)"
    )

    parser.add_argument(
        "--multiscale",
        dest="multiscale",
        action="store_true",
        help="extract multiscale features",
    )
    parser.set_defaults(multiscale=False)

    parser.add_argument(
        "--no-relu",
        dest="use_relu",
        action="store_false",
        help="remove ReLU after the dense feature extraction module",
    )
    parser.set_defaults(use_relu=True)

    args = parser.parse_args()

    # Creating CNN model
    model = D2Net(model_file=args.model_file, use_relu=args.use_relu, use_cuda=use_cuda)
    return model, args, device


def extract_using_d2_net(image, model, args, device):

    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        scale = args.max_edge / max(resized_image.shape)
        pil_img = Image.fromarray(resized_image)
        pil_img = pil_img.resize(
            (int(resized_image.shape[1] * scale), int(resized_image.shape[0] * scale))
        )
        resized_image = np.array(pil_img).astype("float")

    if sum(resized_image.shape[:2]) > args.max_sum_edges:
        scale = args.max_sum_edges / sum(resized_image.shape[:2])
        pil_img = Image.fromarray(resized_image)
        pil_img = pil_img.resize(
            (int(resized_image.shape[1] * scale), int(resized_image.shape[0] * scale))
        )
        resized_image = np.array(pil_img).astype("float")

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(resized_image, preprocessing=args.preprocessing)

    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32), device=device
                ),
                model,
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32), device=device
                ),
                model,
                scales=[1],
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]
    indices = np.argsort(scores)[-500:]
    keypoints = keypoints[indices, :]
    descriptors = descriptors[indices, :]
    scores = scores[indices]
    return keypoints, scores, descriptors


def extract_and_describe_using_d2_net(image):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Argument parsing
    parser = argparse.ArgumentParser(description="Feature extraction script")

    parser.add_argument(
        "--preprocessing",
        type=str,
        default="caffe",
        help="image preprocessing (caffe or torch)",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="d2-net/models/d2_tf.pth",
        help="path to the full model",
    )

    parser.add_argument(
        "--max_edge", type=int, default=1600, help="maximum image size at network input"
    )
    parser.add_argument(
        "--max_sum_edges",
        type=int,
        default=2800,
        help="maximum sum of image sizes at network input",
    )

    parser.add_argument(
        "--output_extension",
        type=str,
        default=".d2-net",
        help="extension for the output",
    )
    parser.add_argument(
        "--output_type", type=str, default="npz", help="output file type (npz or mat)"
    )

    parser.add_argument(
        "--multiscale",
        dest="multiscale",
        action="store_true",
        help="extract multiscale features",
    )
    parser.set_defaults(multiscale=False)

    parser.add_argument(
        "--no-relu",
        dest="use_relu",
        action="store_false",
        help="remove ReLU after the dense feature extraction module",
    )
    parser.set_defaults(use_relu=True)

    args = parser.parse_args()

    # Creating CNN model
    model = D2Net(model_file=args.model_file, use_relu=args.use_relu, use_cuda=use_cuda)

    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        scale = args.max_edge / max(resized_image.shape)
        pil_img = Image.fromarray(resized_image)
        pil_img = pil_img.resize(
            (int(resized_image.shape[1] * scale), int(resized_image.shape[0] * scale))
        )
        resized_image = np.array(pil_img).astype("float")

    if sum(resized_image.shape[:2]) > args.max_sum_edges:
        scale = args.max_sum_edges / sum(resized_image.shape[:2])
        pil_img = Image.fromarray(resized_image)
        pil_img = pil_img.resize(
            (int(resized_image.shape[1] * scale), int(resized_image.shape[0] * scale))
        )
        resized_image = np.array(pil_img).astype("float")

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(resized_image, preprocessing=args.preprocessing)

    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32), device=device
                ),
                model,
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32), device=device
                ),
                model,
                scales=[1],
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]
    return keypoints, descriptors, scores


def run_d2_detector_on_folder(file_name, images_folder, save_folder, image_list=None):
    precomputed_file = f"{save_folder}/{file_name}"
    my_file = Path(precomputed_file)

    if image_list is None:
        image_list = os.listdir(images_folder)
    name2kp = {}
    model, args, device = load_model()
    for name in tqdm(image_list, desc="Running D2 detector on database images"):
        im_name = os.path.join(images_folder, name)
        img = cv2.imread(im_name)
        keypoints, responses, descriptors = extract_using_d2_net(
            img, model, args, device
        )
        keypoints = keypoints.astype(np.int16)
        name2kp[name] = (keypoints, descriptors)
    with open(precomputed_file, "wb") as handle:
        pickle.dump(name2kp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return precomputed_file


def d2_feature_detection(img_, vis=False):
    model, args, device = load_model()
    strongest_keypoints, responses = extract_using_d2_net(img_, model, args, device)
    if vis:
        indices = np.argsort(responses)[-1000:]
        for idx, (x2, y2, _) in enumerate(strongest_keypoints[indices]):
            x2, y2 = map(int, (x2, y2))
            cv2.circle(img_, (x2, y2), 5, (128, 128, 0), 1)
        cv2.imwrite("test.png", img_)
    return strongest_keypoints


def find_images_7_scenes(images_folder):
    images = glob.glob(f"{images_folder}/*/*.color.png")
    return images


def read_image_list(in_dir):
    sys.stdin = open(in_dir, "r")
    lines = sys.stdin.readlines()
    data = []
    for line in lines:
        data.append(line[:-1])
    return data


if __name__ == "__main__":
    list_ = find_images_7_scenes("/home/n11373598/work/redkitchen/images")
    test_file = (
        "/home/n11373598/work/7scenes_reference_models/redkitchen/sfm_gt/list_test.txt"
    )
    test_images_list = read_image_list(test_file)
    train_list_ = [
        file_
        for file_ in list_
        if file_.split("/home/n11373598/work/redkitchen/images/")[-1]
        not in test_images_list
    ]

    run_d2_detector_on_folder(
        "d2_kp_and_desc_train.pkl",
        "/home/n11373598/work/redkitchen/images",
        "/home/n11373598/work/d2-net",
        train_list_,
    )

    test_list_ = [
        file_
        for file_ in list_
        if file_.split("/home/n11373598/work/redkitchen/images/")[-1]
        in test_images_list
    ]

    run_d2_detector_on_folder(
        "d2_kp_and_desc_test.pkl",
        "/home/n11373598/work/redkitchen/images",
        "/home/n11373598/work/d2-net",
        test_list_,
    )
