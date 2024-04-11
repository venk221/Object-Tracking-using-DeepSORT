import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
import random
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching

metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", 0.2, None)
tracker = Tracker(metric)
data_path = r'/media/venk/DATA/WPI/Kitti_dataset/MOT16'
train_path = os.path.join(data_path, 'train')
test_path = os.path.join(data_path, 'test')
colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for j in range(10)]

model = YOLO("yolov8n.pt")

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")
    print("gt=",groundtruth_file)
    detection_file = os.path.join(sequence_dir, "det/det.txt")
    print("gt=",detection_file)

    detections = None
    if detection_file is not None:
        detections = np.loadtxt(detection_file, delimiter=',')
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0] #.astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def find_all_dirs(data_path):
  folders = []
  for item in os.listdir(data_path):
      if os.path.isdir(os.path.join(data_path, item)):
          dir = os.path.join(data_path, item)
          folders.append(dir)
  return sorted(folders)

dirs = find_all_dirs(train_path)

for k,dir in enumerate(dirs):
  if k == 0:
    print(dir)
    seq_info = gather_sequence_info(dir, None)
    print(seq_info['detections'].shape)
    print(seq_info['groundtruth'].shape)
    print(seq_info['max_frame_idx'])
    detection_mat = seq_info['detections']
    detection_list = []
    img_list = []
    frame_indices = detection_mat[:, 0]
    for i in seq_info['image_filenames']:
        print("New Image")
        mask = frame_indices == i
        rows = np.where(mask)
        image = cv2.imread(
                seq_info["image_filenames"][i], cv2.IMREAD_COLOR)
        for j in detection_mat[rows]: 
            bbox, confidence, feature = j[2:6], j[6], j[10:]
            print("feat",feature)
            if bbox[3] < 0:
              continue
            detection_list.append(Detection(bbox, confidence, feature))
            img_list.append(seq_info["image_filenames"][i])

    print(np.array(detection_list).shape)
    print("images",np.array(img_list).shape)
    detections = [d for d in detection_list if d.confidence >= 0.8] # if d.confidence >= 0.8
    print(np.array(detections).shape)

    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(
        boxes, 1.0, scores)
    detections = [detections[i] for i in indices]

    print(len(boxes), len(scores), len(indices), len(detections))
    # Update tracker.
    tracker.predict()
    tracker.update(detections)
    for k, track in enumerate(tracker.tracks):
      bbox = track.to_tlwh()
      x1, y1, x2, y2 = bbox
      track_id = track.track_id
      image = cv2.imread(
                seq_info["image_filenames"][track_id], cv2.IMREAD_COLOR)
      cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)), (colors[int(track_id)% len(colors)]), 3)
      cv2.imshow('frame', image)
      cv2.waitKey(15)

      cv2.destroyAllWindows()  
        
        # results = model(image)
        # for res in results:
        #     detections = []
        #     for r in res.boxes.data.tolist():
        #         x1, y1, x2, y2, score, class_id = r
        #         x1, x2, y1, y2, class_id = int(x1), int(x2), int(y1), int(y2), int(class_id) 
        #         bbox = x1, x2, y1, y2 
        #         detections.append(Detection(bbox, score, class_id))
        
        #     print(detections)
        #     tracker.predict()
        #     tracker.update(detections)



    else:
      break























































    
