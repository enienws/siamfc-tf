from __future__ import division
import sys
import os
import numpy as np
from PIL import Image
import src.siamese as siam
from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
import cv2
from siamfc_tracker import SiamFCTracker
from colorization_tracker import ColorizationTracker
from hybrid_tracker import HybridTracker
import collections
from toolkit.datasets import DatasetFactory
import vot
output_folder = "/home/engin/Documents/siamfc-tf/data/output/"

args_dataset = "VOT2018"
dataset_root = "/home/engin/Documents/pysot/testing_dataset/VOT2018"
# result_output = "/home/engin/Documents/pysot/experiments/siamfc/results/VOT2018/siamfc/baseline"
# result_output = "/home/engin/Documents/pysot/experiments/siamcolor/results/VOT2018/siamcolor/baseline"
result_output = "/home/engin/Documents/pysot/experiments/siamhybrid/results/VOT2018/alpha_09/baseline"
visualize = False
save_vis = True
vis_output = "/home/engin/Documents/colorization_output"

def get_bbox_rect(region):
    bbox_tl = (int(region.x), int(region.y))
    bbox_br = (int(region.x + region.width), int(region.y + region.height))
    return bbox_tl, bbox_br


def main():
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # TODO: allow parameters from command line or leave everything in json files?
    # hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    # final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    # filename, image, templates_z, templates_x, scores, scores_original = siam.build_tracking_graph(final_score_sz, design, env)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args_dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    # iterate through all videos of evaluation.dataset
    videos_list = list(dataset.videos.keys())
    videos_list.sort()
    nv = np.size(videos_list)
    tracker = None
    for i in range(nv):
        current_key = sorted(list(dataset.videos.keys()))[i]
        gt, frame_name_list, frame_sz, n_frames = _init_video(dataset, current_key)
        rect = region_to_bbox(gt[0], False)
        # tracker = SiamFCTracker(frame_name_list[0], vot.Rectangle(rect[0],rect[1],rect[2],rect[3]))
        # tracker = ColorizationTracker(frame_name_list[0], vot.Rectangle(rect[0], rect[1], rect[2], rect[3]))
        if tracker is None:
            tracker = HybridTracker(frame_name_list[0], vot.Rectangle(rect[0], rect[1], rect[2], rect[3]))
        else:
            tracker.HotInit(frame_name_list[0], vot.Rectangle(rect[0], rect[1], rect[2], rect[3]))
        bboxes = []
        for i in range(0, n_frames):
            bbox, confidence = tracker.track(frame_name_list[i])
            bboxes.append(bbox)

        #Visualize
        if visualize:
            for i, (bbox, groundt, frame_name) in enumerate(zip(bboxes, gt, frame_name_list)):
                image = cv2.imread(frame_name)
                bbox_pt1, bbox_pt2 = get_bbox_rect(bbox)
                bbox_gt1, bbox_gt2 = get_gt_bbox_cv(groundt)

                #Draw result
                cv2.rectangle(image, bbox_pt1, bbox_pt2, (0,255,0))
                #Draw ground truth
                cv2.rectangle(image, bbox_gt1, bbox_gt2, (0,0,0))
                cv2.imshow("Results:", image)
                if save_vis:
                    cv2.imwrite(os.path.join(vis_output, str(i) + ".jpg"), image)
                cv2.waitKey()

        #Reset the tracker, get prepared for next subset
        tracker.reset()

        # Write the results to disc for evaluation
        bboxes_n = []
        for bbox in bboxes:
            bboxes_n.append([bbox.x, bbox.y, bbox.width, bbox.height])
        bboxes_n[0] = [1]
        target_dir = os.path.join(result_output, current_key)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        results_file = current_key + "_" + "{:03d}".format(1) + ".txt"
        results_abs_file = os.path.join(target_dir, results_file)
        with open(results_abs_file, "w") as f:
            for bbox in bboxes_n:
                if len(bbox) == 1:
                    f.write('%d\n' % (bbox[0]))
                else:
                    f.write('%.2f, %.2f, %.2f, %.2f\n' % (bbox[0], bbox[1], bbox[2], bbox[3]))


def _init_video(dataset, current_key):
    #reimplement using dataset from pysot
    gt = dataset.videos[current_key].gt_traj
    frame_name_list = dataset.videos[current_key].img_names
    frame_width = dataset.videos[current_key].width
    frame_height = dataset.videos[current_key].height
    frame_sz = [frame_width, frame_height]
    n_frames = len(frame_name_list)
    return np.array(gt), frame_name_list, frame_sz, n_frames

def get_bbox_cv(bbox):
    x, y, width, height = bbox
    return (int(x), int(y)), (int(x + width), int(y + height))

def get_gt_bbox_cv(bbox):
    x, y, width, height = region_to_bbox(bbox, False)
    return (int(x), int(y)), (int(x + width), int(y + height))



def region_to_bbox(region, center=True):
    n = len(region)
    assert n == 4 or n == 8, ('GT region format is invalid, should have 4 or 8 entries.')

    if n == 4:
        return _rect(region, center)
    else:
        return _poly(region, center)


# we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
        return cx, cy, w, h
    else:
        # region[0] -= 1
        # region[1] -= 1
        return region


def _poly(region, center):
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx, cy, w, h
    else:
        return cx - w / 2, cy - h / 2, w, h


def convert_bbox(bbox_8):
    if len(bbox_8) == 8:
        x1, y1, x2, y2, x3, y3, x4, y4 = bbox_8
        return x1, y1, x2 - x1, y4 - y1
    else:
        x1, y1, x2, y2 = bbox_8
        return x1, y1, x2, y2


if __name__ == '__main__':
    sys.exit(main())
