import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from skimage.transform import resize
# import craft functions
from my_craft_text_detector.craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
import os
# os.listdir("/data/ocr/data/text_recognition/ocr_data_v3_280225/Arial_Bold/images/img_18563.jpg")
import matplotlib.pyplot as plt
import cv2
import copy
class SplittingCraft():
    def __init__(self,
        text_threshold=0.6,
        link_threshold=1,
        low_text=0.6,
        cuda=True,
    
    ):
        
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.cuda = cuda
        self.craft_net = load_craftnet_model(cuda=self.cuda)
        self.refine_net = load_refinenet_model(cuda=self.cuda)

    def __call__(self, image):
        h,w,_ = image.shape
        
        desired_long_size = max(h,w)
        ratio = desired_long_size/max(h,w)
        unpadded_target_h = int(h * ratio)
        unpadded_target_w = int(w*ratio)
        padded_h = 32 if not unpadded_target_h%32 else unpadded_target_h%32
        padded_w = 32 if not unpadded_target_w%32 else unpadded_target_w%32
        padded_target_h = unpadded_target_h + 32 - padded_h
        padded_target_w = unpadded_target_w + 32 - padded_w

        prediction_result = get_prediction(
            image=copy.deepcopy(image),
            craft_net=self.craft_net,
            # refine_net=self.refine_net,
            text_threshold=0.6,
            link_threshold=1,
            low_text=0.6,
            cuda=True,
            long_size=desired_long_size
        )
        text_score = prediction_result['score_text']
        link_score = prediction_result['score_link']

        # resized_arr = resize(array_img, (h,w), anti_aliasing=True)
        resized_textscore = resize(text_score, (padded_target_h, padded_target_w), anti_aliasing=True)
        resized_linkscore = resize(link_score, (padded_target_h, padded_target_w), anti_aliasing=True)

        resized_textscore = resized_textscore[:unpadded_target_h, :unpadded_target_w]
        resized_linkscore = resized_linkscore[:unpadded_target_h, :unpadded_target_w]

        resized_textscore = resize(resized_textscore, (h, w), cv2.INTER_LINEAR)
        resized_linkscore = resize(resized_linkscore, (h, w), cv2.INTER_LINEAR)

        ret, link_mask = cv2.threshold(resized_linkscore, 0.5, 1, 0)
        ret, score_mask = cv2.threshold(resized_textscore, 0.6, 1, 0)

        # Get connected components for score_mask
        num_labels_score, labels_im_score = cv2.connectedComponents(score_mask.astype(np.uint8))

        # Get connected components for link_mask
        num_labels_link, labels_im_link = cv2.connectedComponents(link_mask.astype(np.uint8))
        score_boundaries = []
        for label in range(1, num_labels_score):  # skipping background label 0
            rows, cols = np.where(labels_im_score == label)
            if cols.size:
                leftmost = int(np.min(cols))
                rightmost = int(np.max(cols))
                score_boundaries.append((leftmost, rightmost))

        # Create a list of (leftmost, rightmost) for each connected component in link_mask
        link_boundaries = []
        for label in range(1, num_labels_link):  # skipping background label 0
            rows, cols = np.where(labels_im_link == label)
            if cols.size:
                leftmost = int(np.min(cols))
                rightmost = int(np.max(cols))
                link_boundaries.append((leftmost, rightmost))
        nearest_matches = []

        for s_left, s_right in score_boundaries:
            # Find candidates in link_boundaries with leftmost coordinate bigger than score component's left
            right_candidates = [lb for lb in link_boundaries if lb[0] > s_left]
            nearest_right = min(right_candidates, key=lambda lb: lb[0] - s_left) if right_candidates else None
            
            # Find candidates in link_boundaries with rightmost coordinate smaller than score component's right
            left_candidates = [lb for lb in link_boundaries if lb[1] < s_right]
            nearest_left = max(left_candidates, key=lambda lb: lb[1]) if left_candidates else None
            
            nearest_matches.append(((s_left, s_right), nearest_right, nearest_left))
        extracted_coordinates = []
        for score_bounds, nearest_right, nearest_left in nearest_matches:
            # if nearest_right is None or nearest_left is None:
            #     continue
            # print(score_bounds, nearest_right, nearest_left)
            if nearest_right is None:
                # right_coord = score_bounds[1]
                right_coord = w
            else:
                right_coord = nearest_right[1]
            if nearest_left is None:
                # left_coord = score_bounds[0]
                left_coord = 0
            else:
                left_coord = nearest_left[0]
            extracted_coordinates.append((left_coord, right_coord))
            # print(extracted_coordinates)
        

        sorted_extracted_coordinates = sorted(extracted_coordinates, key=lambda x: x[0])
        # Crop each box from the image using the sorted extracted_coordinates.
        sorted_crops = [image[:, x_start:x_end] for (x_start, x_end) in sorted_extracted_coordinates]
        return sorted_crops