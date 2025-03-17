import sys
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))


import cv2
import numpy as np
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt

# First, define the Character_Splitting class
class Character_Splitting:
    def __init__(self, min_area=10, return_cropped_img = False):
        self.min_area = min_area
        self.return_cropped_img = return_cropped_img
    def extract_characters(self, img_array):
        # Convert image array from shape (3, height, width) to (height, width, 3)
        img = np.transpose(img_array, (1, 2, 0))
        img_height, img_width, _ = img.shape
        
        
        
        
        # Convert to grayscale if it is a color image
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to create binary image
        # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        _, binary = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Find connected components (the characters)
        labels = measure.label(binary, connectivity=2)
        
        # Get properties of each labeled region
        regions = regionprops(labels)
        
        # Sort regions by x-coordinate (left to right)
        regions = sorted(regions, key=lambda x: x.bbox[1])
        
        bounding_boxes = []
        characters = []
        
        # Extract bounding box for each character
        for region in regions:
            # Filter out very small regions (noise)
            if region.area < self.min_area:
                continue
            
            # Get bounding box coordinates (min_row, min_col, max_row, max_col)
            minr, minc, maxr, maxc = region.bbox
            
            # Extract the character image, and invert colors to have white on black
            char_img = binary[minr:maxr, minc:maxc]
            char_img = 255 - char_img
            
            # Append bounding box and character image
            bounding_boxes.append((minc, minr, maxc, maxr))
            characters.append(char_img)
        
        # Expand and merge bounding boxes
        merged_boxes = self.expand_and_merge_boxes(bounding_boxes, img_height, iou_threshold=0.1)
        if self.return_cropped_img:
            return merged_boxes, characters
        return merged_boxes
        # return bounding_boxes, characters

    def expand_and_merge_boxes(self, bounding_boxes, img_height, iou_threshold=0.1):
        """
        Given a list of bounding boxes (x1, y1, x2, y2) and the height of the original image,
        expand each box vertically (i.e. set y1 to 0 and y2 to img_height). Then merge any boxes
        that have an Intersection-over-Union greater than iou_threshold based on their x-coordinates.
        Returns a list of merged bounding boxes.
        """
        # Step 1: Expand boxes vertically over the full height.
        expanded_boxes = [(x1, 0, x2, img_height) for (x1, y1, x2, y2) in bounding_boxes]

        # Step 2: Helper function to compute IoU (only using x axis, as y coordinates are identical)
        def get_iou(box1, box2):
            # Unpack x coordinates from the boxes.
            x1_box1, _, x2_box1, _ = box1
            x1_box2, _, x2_box2, _ = box2
            # Compute the horizontal intersection.
            xA = max(x1_box1, x1_box2)
            xB = min(x2_box1, x2_box2)
            inter_width = max(0, xB - xA)
            
            # Compute each box's width.
            width1 = x2_box1 - x1_box1
            width2 = x2_box2 - x1_box2
            
            # Compute union width.
            union_width = width1 + width2 - inter_width
            return inter_width / union_width if union_width != 0 else 0

        # Step 3: Sort the expanded boxes by their left x-coordinate.
        expanded_boxes.sort(key=lambda box: box[0])
        merged_boxes = []
        
        # Step 4: Iterate and merge overlapping boxes.
        while expanded_boxes:
            current_box = expanded_boxes.pop(0)
            i = 0
            while i < len(expanded_boxes):
                if get_iou(current_box, expanded_boxes[i]) > iou_threshold:
                    # Merge the two boxes:
                    new_box = (
                        min(current_box[0], expanded_boxes[i][0]),
                        0,  # top remains 0
                        max(current_box[2], expanded_boxes[i][2]),
                        img_height  # bottom remains img_height
                    )
                    current_box = new_box
                    expanded_boxes.pop(i)
                    i = 0  # Reset and check again from the beginning.
                else:
                    i += 1
            merged_boxes.append(current_box)
        
        return merged_boxes

    def visualize_boxes(self, img_array, bounding_boxes):
        # Convert image array from shape (3, height, width) to (height, width, 3)
        img = np.transpose(img_array, (1, 2, 0)).copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # Draw bounding boxes.
        for (x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img

# Example usage:
if __name__ == "__main__":
    # Create a dummy 3-channel image: shape (3, height, width)
    dummy_img = np.zeros((3, 200, 400), dtype=np.uint8)
    
    # Draw dummy characters (white rectangles) on a black background.
    img_rgb = dummy_img.transpose(1, 2, 0)
    cv2.rectangle(img_rgb, (50, 50), (80, 150), (255, 255, 255), -1)
    cv2.rectangle(img_rgb, (90, 60), (120, 140), (255, 255, 255), -1)
    cv2.rectangle(img_rgb, (130, 40), (160, 160), (255, 255, 255), -1)
    
    splitter = Character_Splitting(min_area=20)
    boxes, chars = splitter.extract_characters(dummy_img)
    print("Merged bounding boxes:", boxes)
    
    vis_img = splitter.visualize_boxes(dummy_img, boxes)
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Merged Bounding Boxes")
    plt.axis("off")
    plt.show()
