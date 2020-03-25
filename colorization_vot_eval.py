import vot
import sys
from colorization_tracker import ColorizationTracker

# Initialize trax handler
# handle = vot.VOT("rectangle")

# Get the gt and related image file
# gt = handle.region()
# imagefile = handle.frame()
# if not imagefile:
#     sys.exit(0)


#Initialize the tracker
tracker = ColorizationTracker()

#Start the tracking loop
bboxes = []
gts = []
frame_name_list = []

while True:
# for gt, imagefile in zip(gts, image_files):
#     imagefile = handle.frame()
#     gt = handle.region()
    imagefile = "/media/engin/63c43c7a-cb63-4c43-b70c-f3cb4d68762a/datasets/DAVIS/JPEGImages/480p/bear/00000.jpg"
    if not imagefile:
        print("Image file cannot be read from trax server..")
        break

    #Run the tracker
    region, confidence = tracker.track(imagefile)


    #Send the results to trax server
    # handle.report(region, confidence)



