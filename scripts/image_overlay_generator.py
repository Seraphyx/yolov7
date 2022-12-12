import os
import pathlib
import argparse
import yaml
from pprint import pprint
import cv2
import logging
import numpy as np
from matplotlib import pyplot as plt
# import contextlib2
import random
from PIL import Image
import imgaug.augmenters as iaa
import copy
import shutil
import sys

import tensorflow.compat.v1 as tf

# from object_detection.dataset_tools import tf_record_creation_util
# from object_detection.utils import dataset_util

from odp.generator.augment import Augmenter, augment_image
from odp.utils.image import load_image, get_bb, get_mask_bb, trim_around_segmentation
from odp.utils.data import open_sharded_output_tfrecords

# import warnings
# warnings.filterwarnings("ignore")

"""
How to use:
from the root folder run:
python scripts/image_overlay_generator.py --clear_output_dir --coco_format --n_images_per_background=16 --n_images_per_area=1
python scripts/image_overlay_generator.py --instruction=data/generator/instructions/generate_data_pre_augmented.yaml --clear_output_dir
python scripts/image_overlay_generator.py --instruction=data/generator/instructions/generate_data_pre_augmented.yaml --n_images_per_background=16 --clear_output_dir


python scripts/image_overlay_generator.py --instruction=data/generator/instructions/toxic_image_dataset.yaml --clear_output_dir --coco_format --n_images_per_background=2 --n_images_per_area=1 --full_area


python scripts/image_overlay_generator.py --instruction=data/generator/instructions/generate_data_train.yaml
python scripts/image_overlay_generator.py --instruction=data/generator/instructions/generate_data_valid.yaml --clear_output_dir
python scripts/image_overlay_generator.py --instruction=data/generator/instructions/generate_data_test.yaml

"""

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Create synthetic training examples.')
parser.add_argument('--instruction',
                    default="data/generator/instructions/training_image_generator.yaml",
                    help="instructions how to overlay'")
parser.add_argument('--output_coco_dir',
                    default="data/generator/coco",
                    help="output of the coco datasets'")
parser.add_argument('--output_h',
                    type=int,
                    help="output image height'")
parser.add_argument('--output_w',
                    type=int,
                    help="output image height'")
parser.add_argument('--n_images_per_background',
                    type=int,
                    help="Repeat images per background'")
parser.add_argument('--n_images_per_area',
                    type=int,
                    help="Repeat images per background'")
parser.add_argument('--n_ignore_images',
                    type=int,
                    help="Ignore images also included'")
parser.add_argument('--output_tf_shards',
                    type=int,
                    help="TFRecord Shards")
parser.add_argument('--seed',
                    type=int,
                    help="seed")
parser.add_argument('--output_dir',
                    type=str,
                    help="Output pictures to directory")
parser.add_argument('--output_tf_dir',
                    type=str,
                    help="Output tfrecords to directory")
parser.add_argument('--output_tf_prefix',
                    type=str,
                    help="Prefix of tfrecords")
parser.add_argument('--clear_output_dir',
                    action="store_true",
                    help="Delete output directories for clean output")
parser.add_argument('--augment',
                    action="store_true",
                    help="Apply augmentation or not")
parser.add_argument('--tf_format',
                    action="store_true",
                    help="Output tfrecords format")
parser.add_argument('--coco_format',
                    action="store_true",
                    help="Output coco format")
parser.add_argument('--full_area',
                    action="store_true",
                    help="Ignore plot_areas innstruction and just place images anywhere")
parser.add_argument('--ext_valid',
                    nargs='+',
                    help='Only consider valid extensions',
                    default=["jpg", "jpeg", "png"])

# Check Arguments
args = parser.parse_args()


def create_label_map(label_map, label_map_path):
    label_map_string = ""
    for key, value in label_map.items():
        item_template = '''item {{\n\tid: {item_id}\n\tname: '{item_name}'\n}}\n\n'''
        item_template = item_template.format(item_id=value, item_name=key)

        label_map_string += item_template

    with tf.gfile.Open(label_map_path, 'wb') as f:
        f.write(label_map_string)


# def create_tf_record(output_path, num_shards, image_id, img, img_path, labels_text, labels_id, bbox, masks=None):
#     with contextlib2.ExitStack() as tf_record_close_stack:
#         output_tfrecords = open_sharded_output_tfrecords(
#             tf_record_close_stack, output_path, num_shards)
#
#         tf_example = create_dataset(img, img_path, labels_text, labels_id, bbox, masks)
#         shard_idx = image_id % num_shards
#         output_tfrecords[shard_idx].write(tf_example.SerializeToString())


# def create_dataset(img, img_path, labels_text, labels_id, bbox, masks=None, normalize_bb=True):
#     image_shape = img.shape
#
#     img_suffix = pathlib.Path(img_path).suffix
#
#     img_path = str(pathlib.Path(img_path))
#
#     image_string = open(img_path, 'rb').read()
#
#     if normalize_bb:
#         bbox['xmin'] = [b / image_shape[1] for b in bbox['xmin']]
#         bbox['xmax'] = [b / image_shape[1] for b in bbox['xmax']]
#         bbox['ymin'] = [b / image_shape[0] for b in bbox['ymin']]
#         bbox['ymax'] = [b / image_shape[0] for b in bbox['ymax']]
#
#         for i in range(len(bbox['xmin'])):
#             try:
#                 assert bbox['ymin'][i] < bbox['ymax'][i], "ymin [{}] >= ymax [{}]".format(bbox['ymin'][i],
#                                                                                           bbox['ymax'][i])
#                 assert bbox['xmin'][i] < bbox['xmax'][i], "xmin [{}] >= xmax [{}]".format(bbox['xmin'][i],
#                                                                                           bbox['xmax'][i])
#             except AssertionError as msg:
#                 print(msg)
#                 raise
#
#     feature_dict = {
#         'image/height': dataset_util.int64_feature(image_shape[0]),
#         'image/width': dataset_util.int64_feature(image_shape[1]),
#         'image/object/class/text': dataset_util.bytes_list_feature(labels_text),
#         'image/object/class/label': dataset_util.int64_list_feature(labels_id),
#         'image/encoded': dataset_util.bytes_feature(image_string),
#         'image/filename': dataset_util.bytes_feature(img_path.encode('utf8')),
#         'image/format': dataset_util.bytes_feature(img_suffix.encode('utf8')),
#         'image/object/bbox/xmin': dataset_util.float_list_feature(bbox['xmin']),
#         'image/object/bbox/xmax': dataset_util.float_list_feature(bbox['xmax']),
#         'image/object/bbox/ymin': dataset_util.float_list_feature(bbox['ymin']),
#         'image/object/bbox/ymax': dataset_util.float_list_feature(bbox['ymax']),
#     }
#
#     if masks:
#         mask_stack = np.stack(masks).astype(np.float32)
#         masks_flattened = np.reshape(mask_stack, [-1])
#         feature_dict['image/object/mask'] = (dataset_util.float_list_feature(masks_flattened.tolist()))
#
#     example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
#
#     return example


def resize_target_range(img, target_h_range=(240, 480), target_w_range=(360, 720)):
    # Max height and width
    overlay_min_height, overlay_max_height = target_h_range
    overlay_min_width, overlay_max_width = target_w_range

    # Choose random size within range, anchored on height
    foreground_target_h = np.random.uniform(low=overlay_min_height, high=overlay_max_height, size=None)
    ratio_target_h = foreground_target_h / img.shape[0]
    foreground_target_w = img.shape[1] * ratio_target_h

    # If this exceeds the max size of the width, then resize to max width range
    if foreground_target_w > overlay_max_width:
        #         print("---foreground_target_w > overlay_max_width: {} > {}".format(foreground_target_w, overlay_max_width))
        foreground_target_w = overlay_max_width
        ratio_target_w = foreground_target_w / img.shape[1]
        foreground_target_h = img.shape[0] * ratio_target_w

    # Target Size
    foreground_target = (int(foreground_target_w), int(foreground_target_h))

    img_resize = cv2.resize(img, dsize=foreground_target, interpolation=cv2.INTER_AREA)

    return img_resize


def resize_img_bbox(img, plot_areas, target_h, target_w, show_box=False):

    source_h, source_w, source_c = img.shape

    source_ratio = source_h / source_w

    plot_area_new = []

    # Widescreen
    if source_ratio < 1:
        convert_ratio = target_w / source_w
        target_h_adj = int(convert_ratio * source_h)

        img_resize = cv2.resize(img, (target_w, target_h_adj))

        # Padding
        padding = target_h - target_h_adj
        padding_vertical = padding / 2

        # int is a floor function
        padding_top = int(padding_vertical)
        padding_bot = int(padding_vertical)

        # Put extra pixel on top
        if padding % 2 == 1:
            padding_top += 1

        # blank
        img_blank = np.zeros((target_h, target_w, source_c), dtype=np.uint8)

        # Input image
        img_blank[padding_top:(padding_top + target_h_adj), :, :] = img_resize

        # Convert plotareas
        for plot_area in plot_areas:
            plot_area = copy.deepcopy(plot_area)
            start_point = plot_area['top_left']
            end_point = plot_area['bottom_right']

            start_point = [round(p * convert_ratio) for p in start_point]
            end_point = [round(p * convert_ratio) for p in end_point]
            start_point[1] = start_point[1] + padding_top
            end_point[1] = end_point[1] + padding_top
            plot_area['top_left'] = start_point
            plot_area['bottom_right'] = end_point
            plot_area_new.append(plot_area)

    # Portrait
    else:
        convert_ratio = target_h / source_h
        target_w_adj = int(convert_ratio * source_w)

        img_resize = cv2.resize(img, (target_w_adj, target_h))

        # Padding
        padding = target_w - target_w_adj
        padding_horizontal = padding / 2

        # int is a floor function
        padding_left = int(padding_horizontal)
        padding_right = int(padding_horizontal)

        # Put extra pixel on top
        if padding % 2 == 1:
            padding_left += 1

        # blank
        img_blank = np.zeros((target_h, target_w, source_c), dtype=np.uint8)

        # Input image
        img_blank[:, padding_left:(padding_left + target_w_adj), :] = img_resize

        # Convert plotareas
        for plot_area in plot_areas:
            start_point = plot_area['top_left']
            end_point = plot_area['bottom_right']

            # Convert plotareas
            start_point = [round(p * convert_ratio) for p in start_point]
            end_point = [round(p * convert_ratio) for p in end_point]
            start_point[0] = start_point[0] + padding_left
            end_point[0] = end_point[0] + padding_left

            plot_area_new.append({
                'top_left': start_point,
                'bottom_right': end_point
            })

    if show_box:
        color = (0, 255, 0, 255)
        thickness = 2
        start_point = tuple(start_point)
        end_point = tuple(end_point)
        img_blank = cv2.rectangle(img_blank, start_point, end_point, color, thickness)

    return img_blank, plot_area_new


def overlay_image(img_background, img_foreground, start_point=(500, 300), end_point=(1300, 550), show_box=False,
                  scale_min=0.25, scale_max=0.50):

    img_comb = img_background

    # Blue color in BGR
    if show_box:
        color = (0, 255, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img_comb = cv2.rectangle(img_comb, start_point, end_point, color, thickness)

    ###==== Add Overlay
    target_h_max = end_point[1] - start_point[1]
    target_w_max = end_point[0] - start_point[0]
    target_h_range = (target_h_max * scale_min, target_h_max * scale_max)
    target_w_range = (target_w_max * scale_min, target_w_max * scale_max)

    # Resize foreground
    img_foreground_resize = resize_target_range(img_foreground, target_h_range=target_h_range,
                                                target_w_range=target_w_range)

    # Place in random spot within the plot_area
    pos_x_offset = img_foreground_resize.shape[1]
    pos_y_offset = img_foreground_resize.shape[0]
    # img_foreground_pos_x_range = (start_point[0], pos_x_offset + start_point[0])
    # img_foreground_pos_y_range = (start_point[1], pos_y_offset + start_point[1])

    # Choose a random position
    img_foreground_pos_x_choose = int(
        np.random.uniform(low=(start_point[0] + pos_x_offset), high=(end_point[0] - pos_x_offset), size=None))
    img_foreground_pos_y_choose = int(
        np.random.uniform(low=(start_point[1] + pos_y_offset), high=(end_point[1] - pos_y_offset), size=None))
    # print(f"img_comb = {img_comb.shape}")
    # print(f"img_foreground_pos_x_choose = {img_foreground_pos_x_choose}")
    # print(f"img_foreground_pos_y_choose = {img_foreground_pos_y_choose}")

    # Mask overalay
    mask_overlay = np.zeros(img_comb.shape[:2] + tuple([1]))
    alpha_overlay = np.zeros(img_comb.shape)

    # Overlay masking alpha channel
    overlay_full = np.zeros(img_comb.shape)
    overlay_alpha = img_foreground_resize[:, :, 3]
    overlay_alpha = (overlay_alpha > 0).astype(np.int)
    f_h, f_w = overlay_alpha.shape

    # Trim if hitting borders
    max_y, max_x, _ = mask_overlay.shape
    f_h += min(0, max_y - (img_foreground_pos_y_choose + f_h))
    f_w += min(0, max_x - (img_foreground_pos_x_choose + f_w))
    overlay_alpha = overlay_alpha[:f_h, :f_w]
    img_foreground_resize = img_foreground_resize[:f_h, :f_w]


    # Make a mask for labels
    # print("=" * 30)
    # print(f"mask_overlay = \n\t{mask_overlay.shape}")
    # print(f"overlay_alpha = \n\t{overlay_alpha.shape}")
    # print(f"img_foreground_pos_y_choose = \n\t{img_foreground_pos_y_choose}")
    # print(f"f_h = \n\t{f_h}")
    # print(f"img_foreground_pos_y_choose + f_h = \n\t{img_foreground_pos_y_choose + f_h}")
    # print(f"img_foreground_pos_x_choose = \n\t{img_foreground_pos_x_choose}")
    # print(f"f_w = \n\t{f_w}")
    # print(f"img_foreground_pos_x_choose + f_w = \n\t{img_foreground_pos_x_choose + f_w}")

    # Redo with smaller window
    if f_w <= 0 or f_h <= 0:
        print("---------- REDOING")
        end_point = end_point if f_w > 0 else [end_point[0] + f_w, end_point[1]]
        end_point = end_point if f_h > 0 else [end_point[0], end_point[1] + f_h]
        return overlay_image(
            img_background=img_background,
            img_foreground=img_foreground,
            start_point=start_point,
            end_point=end_point,
            show_box=show_box,
            scale_min=scale_min,
            scale_max=scale_max
        )

    mask_overlay[img_foreground_pos_y_choose:(img_foreground_pos_y_choose + f_h),
        img_foreground_pos_x_choose:(img_foreground_pos_x_choose + f_w), 0] = overlay_alpha

    # Isolate alpha channel for np.where filtering
    alpha_overlay[img_foreground_pos_y_choose:(img_foreground_pos_y_choose + f_h),
        img_foreground_pos_x_choose:(img_foreground_pos_x_choose + f_w), 3] = overlay_alpha
    alpha_overlay[:, :, 0] = alpha_overlay[:, :, 3]
    alpha_overlay[:, :, 1] = alpha_overlay[:, :, 3]
    alpha_overlay[:, :, 2] = alpha_overlay[:, :, 3]

    # Pad overlay to background size
    overlay_full[img_foreground_pos_y_choose:(img_foreground_pos_y_choose + f_h),
    img_foreground_pos_x_choose:(img_foreground_pos_x_choose + f_w)] = img_foreground_resize

    # Combine by replacing where alpha is positive
    img_comb = np.where(alpha_overlay > 0, overlay_full, img_comb)
    img_comb = img_comb.astype(np.uint8)

    # Tensorflow Object Detection API requires bottom left and top right points
    bb_pos = {
        'x1': img_foreground_pos_x_choose,
        'x2': img_foreground_pos_x_choose + f_w,
        'y1': img_foreground_pos_y_choose + f_h,
        'y2': img_foreground_pos_y_choose,
    }

    # Blue color in BGR
    if show_box:
        color = (0, 0, 255, 255)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        img_comb = cv2.rectangle(img_comb, (bb_pos['x1'], bb_pos['y2']), (bb_pos['x2'], bb_pos['y1']), color, thickness)

    return img_comb, mask_overlay, bb_pos


def class_to_image_paths(image_yaml_list, ext_valid=['jpg', 'png', 'jpeg'], class_key="class"):
    """
    Desc:
        Read instructions and return dictionary that allows looping through images
    """
    import random

    def generator_shuffler(img_list):
        """
        Generator that shuffles the list given while being infinite
        """
        while True:
            random.shuffle(img_list)
            yield from img_list

    class_dict = dict()
    for img in image_yaml_list:
        # Walk through each directory and get files
        images_file = [os.path.join(root, file) for root, subdirs, files in os.walk(img['path']) for
                       file in files if os.path.splitext(file)[-1].replace(".", "") in ext_valid]
        img['img_paths'] = images_file
        img['generator'] = generator_shuffler(images_file)
        class_dict[img[class_key]] = img
    return class_dict


def bbox_to_coco(bbox, labels_id, full_w=None, full_h=None, xy_format=False):
    n = len(bbox['xmin'])
    norm = False

    if full_w is None != full_h is None:
        assert "You need to provide both full_w and full_h"
    elif full_w is not None and full_h is not None:
        norm = True

    coco = []
    for i in range(n):
        # Segment: x1, y1, x2, y2,...
        if xy_format:
            xmin = bbox['xmin'][i]
            xmax = bbox['xmax'][i]
            ymin = bbox['ymin'][i]
            ymax = bbox['ymax'][i]
            if norm:
                xmin /= full_w
                xmax /= full_w
                ymin /= full_h
                ymax /= full_h
            coco.append([labels_id[i], xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])

        # COCO: Center X, Center Y, Width, Height
        else:
            width = bbox['xmax'][i] - bbox['xmin'][i]
            height = bbox['ymax'][i] - bbox['ymin'][i]
            center_x = (bbox['xmax'][i] + bbox['xmin'][i]) / 2
            center_y = (bbox['ymax'][i] + bbox['ymin'][i]) / 2
            if norm:
                width /= full_w
                height /= full_h
                center_x /= full_w
                center_y /= full_h
            coco.append([labels_id[i], center_x, center_y, width, height])
    return coco


def main():

    print("===== Splitting Clips =====")

    # Logging
    logging.basicConfig(
        format='%(asctime)s --- %(message)s',
        datefmt='%Y%m%d_%I:%M:%S%p',
        filename='image_overlay_generator.log',
        filemode='w',
        # stream=sys.stdout,
        level=logging.DEBUG)

    # Read Instructions
    instruction_path = pathlib.Path(args.instruction)
    with open(instruction_path) as file:
        instruction = yaml.load(file, Loader=yaml.FullLoader)
    pprint(instruction)
    if instruction is None:
        print("Instuctions are empty...")
        return

    # Defaults
    default_args = {
        'output_h': 1024,
        'output_w': 1024,
        'n_images_per_background': 2,
        'n_images_per_area': 2,
        'output_tf_shards': 4,
        'augment': False,
        'clear_output_dir': False,
        'output_dir': "data/generator/coco/images",
        'output_tf_dir': "data/generator/output_tf",
        'output_coco_dir': "data/generator/coco",
        'output_tf_prefix': "default",
        'ext_valid': ["jpg", "jpeg", "png"],
        'seed': 32456
    }

    def args_order(arg_key):
        """
        return the arguments in order of
        1) supplied by command
        2) defined in instruction yaml
        3) default within default_args
        :param arg_key:
        :param instructions:
        :return:
        """
        # if arg_key == 'output_dir':
        #     print("----------------------" * 10)
        #     print(hasattr(args, arg_key))
        #     print(getattr(args, arg_key))
        #     print(instruction.get('config', {}))
        #     print(default_args[arg_key])
        if getattr(args, arg_key, None):
            return getattr(args, arg_key)
        elif instruction.get('config', {}).get(arg_key, False):
            return instruction.get('config', {}).get(arg_key)
        else:
            return default_args[arg_key]

    # Order of option override
    output_h = args_order('output_h')
    output_w = args_order('output_w')
    n_images_per_background = args_order('n_images_per_background')
    n_images_per_area = args_order('n_images_per_area')
    output_tf_shards = args_order('output_tf_shards')
    clear_output_dir = args_order('clear_output_dir')
    output_dir = args_order('output_dir')
    output_tf_dir = args_order('output_tf_dir')
    output_coco_dir = args_order('output_coco_dir')
    output_coco_labels_dir = os.path.join(output_coco_dir, 'labels')
    output_coco_images_dir = os.path.join(output_coco_dir, 'images')
    output_coco_instruction_dir = os.path.join(output_coco_dir, 'instructions')
    # output_coco_instruction_dir = args_order('output_coco_instruction_dir')
    output_tf_prefix = args_order('output_tf_prefix')
    augment = args_order('augment')
    ext_valid = args_order('ext_valid')
    seed = args_order('seed')
    background_dir = instruction['background']['background_dir']
    foreground_dir = instruction['foreground']['foreground_dir']

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Clear Output Dir
    print(f"clear_output_dir = {clear_output_dir}")
    print(f"output_dir = {output_dir}")
    if clear_output_dir:
        if os.path.exists(output_dir):
            logging.info("\t::: removing dir: {}".format(output_dir))
            shutil.rmtree(output_dir)
        if os.path.exists(output_tf_dir):
            logging.info("\t::: removing dir: {}".format(output_tf_dir))
            shutil.rmtree(output_tf_dir)
        if os.path.exists(output_coco_dir):
            logging.info("\t::: removing dir: {}".format(output_coco_dir))
            shutil.rmtree(output_coco_dir)
        if os.path.exists(output_coco_instruction_dir):
            logging.info("\t::: removing dir: {}".format(output_coco_instruction_dir))
            shutil.rmtree(output_coco_instruction_dir)

    # Create
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_tf_dir):
        os.makedirs(output_tf_dir)
    if not os.path.exists(output_coco_dir):
        os.makedirs(output_coco_dir)
    if not os.path.exists(output_coco_labels_dir):
        os.makedirs(output_coco_labels_dir)
    if not os.path.exists(output_coco_images_dir):
        os.makedirs(output_coco_images_dir)
    if not os.path.exists(output_coco_instruction_dir):
        os.makedirs(output_coco_instruction_dir)
    output_tf_path = os.path.join(output_tf_dir, output_tf_prefix)
    output_tf_shards = output_tf_shards
    output_tf_pxtxt = str(pathlib.Path(output_tf_dir).joinpath("label_map.pbtxt"))

    # Initialize
    augmenter = Augmenter()
    seq_background = augmenter.seq_background()
    seq_foreground = augmenter.seq_foreground()

    # Target output size
    target_h_background = output_h
    target_w_background = output_w

    # How many times to generate an image per background
    n_images_per_background = n_images_per_background

    # Label map
    if instruction['image_format'] == "folder":
        class_list = os.listdir(foreground_dir)
        if instruction['config'].get('ignore', False):
            label_text = [c for c in class_list if c not in ['ignore']]
            label_text = list(label_text)
            label_text.sort()
            label_text = ['ignore'] + label_text
        else:
            label_text = class_list
            label_text.sort()
    else:
        label_text = set([fg['class'] for fg in instruction['foreground']['images']])
        label_text = list(label_text)
        label_text.sort()
    label_dict = dict(zip(label_text, [i for i in range(len(label_text))]))
    print(label_dict)

    # How many images to place per area
    overlay_per_area_min = 1
    overlay_per_area_max = len(label_text) if n_images_per_area == 0 else n_images_per_area
    print("overlay_per_area_max: {}".format(overlay_per_area_max))

    # Foreground classes and loop generator
    if instruction['image_format'] == "directory":
        background_img_dir = class_to_image_paths(instruction['background']['images'],
                                                  ext_valid=ext_valid,
                                                  class_key="filename")
        foreground_img_dir = class_to_image_paths(instruction['foreground']['images'],
                                                  ext_valid=ext_valid,
                                                  class_key="class")
    # elif instruction['image_format'] == "folder":
        # background_img_dir = class_to_image_paths(instruction['background']['background_dir'],
        #                                           ext_valid=ext_valid,
        #                                           class_key="filename")
        # foreground_img_dir = class_to_image_paths(instruction['foreground']['foreground_dir'],
        #                                           ext_valid=ext_valid,
        #                                           class_key="class")


    # Save Label .pbtxt
    create_label_map(label_dict, output_tf_pxtxt)

    # with contextlib2.ExitStack() as tf_record_close_stack:
    #     output_tfrecords = open_sharded_output_tfrecords(
    #         tf_record_close_stack, output_tf_path, output_tf_shards)

    # Loop through each background
    img_i = 0

    print(instruction['image_format'])
    if instruction['image_format'] == "directory":
        background_images = []
        for k, v in background_img_dir.items():
            for img_path in v['img_paths']:
                item = v.copy()
                del item['img_paths']
                item['path'] = img_path
                background_images.append(item)
    elif instruction['image_format'] == "folder":
        background_filenames = os.listdir(background_dir)
        background_images = [{'path': os.path.join(background_dir, f)} for f in background_filenames]
    else:
        background_images = instruction['background']['images']

    # Hold filenames for coco format
    coco_filelist = []

    # Loop through each background
    for background_i, background in enumerate(background_images):
        print(f"=====> Working on background [{background_i + 1:05} of {len(background_images):05}]")
        logging.debug("----- {}".format(background_i))
        logging.debug(background)

        img_background = load_image(background['path'])

        img_background_mask = img_background[:, :, 3]

        if augment:
            img_backgrounds, imgs_aug_mask, imgs_aug_bb = augment_image(
                img=img_background,
                img_mask=img_background_mask,
                n_iter=n_images_per_background,
                padding_percent=0,
                seq=seq_background)
        else:
            img_backgrounds = [img_background] * n_images_per_background

        # Make multiple of backgrounds
        for background_iter, img_background in enumerate(img_backgrounds):

            logging.debug("\t----- {}".format(background_iter))

            # Add alpha channel
            if img_background.shape[2] == 3:
                img_background = np.dstack((img_background, np.ones(img_background.shape[:2]) * 255))

            # Initialize
            bbox = {
                'xmin': [],
                'xmax': [],
                'ymin': [],
                'ymax': [],
            }
            labels_text = []
            labels_id = []
            masks = []

            # Resize to square
            if args.full_area:
                # print(f"img_background = {img_background.shape}")
                # scale_factor = 0.75
                # bottom_right = [int(img_background.shape[1] * scale_factor), int(img_background.shape[0] * scale_factor)]
                # print(f"bottom_right = {bottom_right}")
                plot_areas = [{'top_left': [0, 0], 'bottom_right': [args.output_w, args.output_h]}]
                background['plot_areas'] = plot_areas
                img_background, plot_areas_new = resize_img_bbox(img=img_background,
                                                                 plot_areas=plot_areas,
                                                                 target_h=target_h_background,
                                                                 target_w=target_w_background, show_box=False)
                # print(f"plot_areas_new = {plot_areas_new}")
            else:
                img_background, plot_areas_new = resize_img_bbox(img=img_background,
                                                                 plot_areas=background['plot_areas'],
                                                                 target_h=target_h_background,
                                                                 target_w=target_w_background, show_box=False)

            # Replace plot are with the new resized one
            # background_plot_area = background['plot_areas']
            # background_plot_area = copy.deepcopy(background['plot_areas'])
            if background_iter == 0:
                background['plot_areas'] = plot_areas_new

            # Background to overlay multiple images
            img_comb = resize_target_range(
                img_background,
                target_h_range=(target_h_background, target_h_background),
                target_w_range=(target_w_background, target_w_background))


            # Iter per area
            for area_i, plot_area in enumerate(background['plot_areas']):
                # plot_area = copy.deepcopy(plot_area)
                logging.debug("\t\t----- area_i={}".format(area_i))
                logging.debug("\t\t----- plot_area={}".format(plot_area))
                # print("\t\t----- area_i={}".format(area_i))

                # Choose
                if plot_area.get("n_images_per_area"):
                    overlay_iter = plot_area.get("n_images_per_area")
                else:
                    overlay_iter = np.random.randint(overlay_per_area_min, overlay_per_area_max + 1)

                ratio_resize = img_comb.shape[0] / img_background.shape[0]
                start_point = tuple([int(s * ratio_resize) for s in plot_area['top_left']])
                end_point = tuple([int(s * ratio_resize) for s in plot_area['bottom_right']])

                #===== Foreground
                for overlay_i in range(overlay_iter):
                    if instruction['image_format'] == "path":
                        foreground = random.choice(instruction['foreground']['images'])
                        img_foreground = load_image(foreground['path'])
                    elif instruction['image_format'] == "directory":
                        choosen_class = random.choice(list(foreground_img_dir.keys()))
                        foreground = {
                            'path': next(foreground_img_dir[choosen_class]['generator']),
                            'class': foreground_img_dir[choosen_class]['class']
                        }
                        img_foreground = load_image(foreground['path'])
                        if img_foreground is None:
                            continue
                    elif instruction['image_format'] == "folder":
                        # class_list = os.listdir(foreground_dir)
                        # class_list = [c for c in class_list if c not in ['ignore']]
                        choosen_class = random.choice(label_text)

                        asset_path_list = os.listdir(os.path.join(foreground_dir, choosen_class))
                        asset_path_chosen = os.path.join(foreground_dir, choosen_class, random.choice(asset_path_list))

                        foreground = {
                            'path': asset_path_chosen,
                            'class': choosen_class
                        }
                        img_foreground = load_image(foreground['path'])
                        if img_foreground is None:
                            continue

                    img_foreground_mask = img_foreground[:, :, 3]

                    if augment:
                        img_foreground, imgs_aug_mask, imgs_aug_bb = augment_image(
                            img=img_foreground,
                            img_mask=img_foreground_mask,
                            n_iter=1,
                            padding_percent=0.25,
                            seq=seq_foreground)
                        img_foreground = img_foreground[0]
                        img_foreground_mask = img_foreground_mask[0]

                    # Trim to fit only the mask
                    img_foreground, img_aug_mask_trim = trim_around_segmentation(
                        img=img_foreground,
                        img_mask=img_foreground_mask)

                    logging.debug(foreground)
                    logging.debug(img_foreground.shape)
                    img_comb, mask_overlay, bb_pos = overlay_image(
                        img_comb,
                        img_foreground,
                        start_point=start_point,
                        end_point=end_point,
                        show_box=False,
                        scale_min=0.10,
                        scale_max=0.25
                    )

                    # Label
                    # print(f"\t\t\t bb_pos = {bb_pos}")
                    bbox['xmin'].append(bb_pos['x1'])
                    bbox['xmax'].append(bb_pos['x2'])
                    bbox['ymin'].append(bb_pos['y2'])
                    bbox['ymax'].append(bb_pos['y1'])
                    labels_text.append(foreground['class'].encode('utf8'))
                    labels_id.append(label_dict[foreground['class']])
                    masks.append(mask_overlay)
                    logging.debug("\t\t\tbbox: {}".format(bbox))

                # # ===== Ignore Images
                # # if instruction['image_format'] == "folder":
                # for ignore_i in range(args.n_ignore_images):
                #     img_comb


            # Output Image
            output_filename = "{:012d}.jpg".format(img_i)
            output_path = pathlib.Path(output_coco_images_dir).joinpath(output_filename)
            coco_filelist.append(str(output_path))
            img_comb = img_comb[:, :, :3]

            # # Plot with annotations
            # img_plot = img_comb
            # for bb_i in range(len(bbox['xmin'])):
            #     img_plot = cv2.rectangle(
            #         img=img_plot,
            #         pt1=(bbox["xmin"][bb_i], bbox["ymin"][bb_i]),
            #         pt2=(bbox["xmax"][bb_i], bbox["ymax"][bb_i]),
            #         color=(255, 0, 0),
            #         thickness=2)
            cv2.imwrite(str(output_path), img_comb)

            # # tfrecord
            # if args.tf_format:
            #     tf_example = create_dataset(img_comb, str(output_path), labels_text, labels_id, bbox, masks=None)
            #     shard_idx = img_i % output_tf_shards
            #     output_tfrecords[shard_idx].write(tf_example.SerializeToString())

            # Coco format
            if args.coco_format:
                # Output
                coco_bbox = bbox_to_coco(bbox, labels_id, full_w=output_h, full_h=output_w, xy_format=True)
                output_coco_filename = "{:012d}.txt".format(img_i)
                output_coco_label_path = pathlib.Path(output_coco_labels_dir).joinpath(output_coco_filename)
                with open(str(output_coco_label_path), "w") as f:
                    for i, coco_label in enumerate(coco_bbox):
                        # Ignore the 'ignore' class
                        if instruction['config'].get('ignore'):
                            if coco_bbox[i][0] == 0:
                                continue
                            else:
                                coco_bbox[i][0] -= 1
                        new_line = "\n" if i + 1 < len(coco_bbox) else ""
                        out_line = " ".join([str(s) if s_i == 0 else f'{s:.15f}' for s_i, s in enumerate(coco_bbox[i])]) + new_line
                        f.write(out_line)

            img_i += 1

    # Instruction YAML
    if args.coco_format:
        # Filelist
        output_coco_filelist_filename = "custom_filelist.txt"
        output_coco_filelist_path = pathlib.Path(output_coco_instruction_dir).joinpath(output_coco_filelist_filename)
        with open(str(output_coco_filelist_path), "w") as f:
            for filename_i, coco_filename in enumerate(coco_filelist):
                new_line = "\n" if filename_i + 1 < len(coco_filelist) else ""
                f.write(coco_filename + new_line)
        print(f"Writing to: \n\t{output_coco_filelist_path}")

        # Instruction
        nc = len(label_text) - 1 if instruction['config'].get('ignore') else len(label_text)
        names = [l for l in label_text if l not in ['ignore']]

        coco_yaml = dict(
            train=str(output_coco_filelist_path),
            val=str(output_coco_filelist_path).replace("coco-train", "coco-valid"),
            test=str(output_coco_filelist_path).replace("coco-train", "coco-test"),
            nc=nc,
            names=names
        )
        output_coco_yaml_filename = "custom_instructions.yaml"
        output_coco_yaml_path = pathlib.Path(output_coco_instruction_dir).joinpath(output_coco_yaml_filename)
        print(f"Writing to: \n\t{output_coco_yaml_path}")
        with open(str(output_coco_yaml_path), 'w') as f:
            yaml.dump(coco_yaml, f)


if __name__ == "__main__":
    main()
