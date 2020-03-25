import vot
import sys
from siamfc_tracker import SiamFCTracker
import cv2
import os
import pickle
import time

#flag for visualization
vis_flag = True
vis_dir = "/home/engin/Documents/vot/workspaces/vot2020/results/SiamFC/image/"

def get_subject(imagefile):
    file_list = imagefile.split("/")
    subject = file_list[file_list.index('sequences') + 1]
    return subject

def get_bbox_cv(region):
    bbox_tl = (int(region.x), int(region.y))
    bbox_br = (int(region.x + region.width), int(region.y + region.height))
    return bbox_tl, bbox_br

# Visualize
def visualize(subject, bboxes, gts, frame_name_list):
    counter = 0
    if not os.path.exists(os.path.join(vis_dir, subject)):
        os.mkdir(os.path.join(vis_dir, subject))

    for bbox, groundt, frame_name in zip(bboxes, gts, frame_name_list):
        image = cv2.imread(frame_name)
        bbox_pt1, bbox_pt2 = get_bbox_cv(bbox)
        bbox_gt1, bbox_gt2 = get_bbox_cv(groundt)

        # Draw result
        cv2.rectangle(image, bbox_pt1, bbox_pt2, (0, 255, 0))
        # Draw ground truth
        cv2.rectangle(image, bbox_gt1, bbox_gt2, (0, 0, 0))
        img_file = "{}.jpg".format(counter)
        img_write_path = os.path.join(vis_dir, subject, img_file)
        cv2.imwrite(img_write_path, image)
        counter = counter + 1

# Initialize trax handler
handle = vot.VOT("rectangle")

# Get the gt and related image file
gt = handle.region()
imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

#Get the current subject
subject = get_subject(imagefile)
print("Subject is {}".format(subject))

#Initialize the tracker
tracker = SiamFCTracker(imagefile, gt)

#Start the tracking loop
bboxes = []
gts = []
frame_name_list = []

while True:
# for gt, imagefile in zip(gts, image_files):
    imagefile = handle.frame()
    gt = handle.region()

    if not imagefile:
        print("Image file cannot be read from trax server..")
        break

    #Run the tracker
    region, confidence = tracker.track(imagefile)

    #Check the flag, act accordingly
    if vis_flag:
        frame_name_list.append(imagefile)
        gts.append(gt)
        bboxes.append(region)

    #Send the results to trax server
    handle.report(region, confidence)

# with open("/home/engin/Documents/image_file.pickle", 'wb') as file:
#     pickle.dump(imagefile, file)
#
# with open("/home/engin/Documents/gt_file.pickle", 'wb') as file:
#     pickle.dump(gt, file)

# if vis_flag:
#     visualize(subject, bboxes, gts, frame_name_list)



