import json
from nodes import MAX_RESOLUTION
from collections import namedtuple

import torch
import numpy as np
from numpy import array, float32

openpose_parts = ["pose_keypoints_2d","face_keypoints_2d","hand_left_keypoints_2d","hand_right_keypoints_2d"]
round_divs = [1, 8,16,32,64,128]

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

class OpenPoseKeyPointExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "select_parts": (openpose_parts, { "default": openpose_parts[0]}),
                "points_list": ("STRING", {"multiline": True, "default": "0, 1"}),
                "round_by": (round_divs, { "default": round_divs[0] }),
                "dilate": ("INT", { "min": 0, "max": 128 }, { "default": 0 }),
                "person_number": ("INT", { "default": 0 }),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "MASK",)
    RETURN_NAMES = ("min_x", "min_y", "max_x", "max_y", "width", "height", "mask")
    FUNCTION = "box_keypoints"
    CATEGORY = "utils"

    def round_resolution(self, min_x, min_y, max_x, max_y, i_width, i_height, DIV):
        min_x=max(round(min_x/DIV-0.5)*DIV, 0)
        min_y=max(round(min_y/DIV-0.5)*DIV, 0)
        max_x=max(round(max_x/DIV+0.5)*DIV, 64)
        max_y=max(round(max_y/DIV+0.5)*DIV, 64)
        
        min_x=min(i_width-(max_x - min_x), min_x)
        min_y=min(i_height-(max_y - min_y), min_y)
        return [int(min_x), int(min_y), int(max_x), int(max_y)]

    def get_keypoint_from_list(self, list, item):
        idx_x = item*3
        idx_y = idx_x + 1
        idx_conf = idx_y + 1
        return (list[idx_x], list[idx_y], list[idx_conf])

    def box_keypoints(self, pose_keypoint, select_parts, points_list, round_by, dilate, person_number):
        pose_keypoint = pose_keypoint[0]
        
        canvas_width = int(pose_keypoint["canvas_width"])
        canvas_height = int(pose_keypoint["canvas_height"])

        points_we_want = [int(element) for element in points_list.split(",")]

        min_x = MAX_RESOLUTION
        min_y = MAX_RESOLUTION
        max_x = 0
        max_y = 0
        
        iter : int = 0
        sum_coord = 0.0
        
        for element in points_we_want:
            (x,y,_) = self.get_keypoint_from_list(pose_keypoint["people"][person_number][select_parts], element)
            if (x > 0.0 and y > 0.0):
                iter+= 1
                sum_coord+=x+y
                
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        
        mean_coord = sum_coord/(2*iter)
        
        if mean_coord < 1.0:
            min_x*=canvas_width
            min_y*=canvas_height
            max_x*=canvas_width
            max_y*=canvas_height
        
        min_x = max(min_x - dilate, 0)
        min_y = max(min_y - dilate, 0)
        max_x = min(max_x + dilate, canvas_width)
        max_y = min(max_y + dilate, canvas_height)
        
        min_x, min_y, max_x, max_y = self.round_resolution(min_x, min_y, max_x, max_y, canvas_width, canvas_height, round_by)
        
        width : int = max_x - min_x
        height : int = max_y - min_y
        
        mask = torch.zeros(1, canvas_height, canvas_width)
        mask[:, min_y:max_y, min_x:max_x] = 1.0
        
        return (min_x, min_y, max_x, max_y, width, height, mask,)
    
class OpenPoseSEGSExtractor(OpenPoseKeyPointExtractor):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "pose_keypoint": ("POSE_KEYPOINT",),
                "select_parts": (openpose_parts, { "default": openpose_parts[0]}),
                "points_list": ("STRING", {"multiline": True, "default": ""}),
                "round_by": (round_divs, { "default": round_divs[0] }),
                "dilate": ("INT", { "min": 0, "max": 128 }, { "default": 0 }),
                "person_number": ("INT", { "default": 0 }),
            }
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "extract_SEGS"
    CATEGORY = "utils"
    
    def extract_SEGS(self, image, pose_keypoint, select_parts, points_list, round_by, dilate, person_number):
        
        image_width = image.shape[2]
        image_height = image.shape[1]
        confidence=array(1.0, dtype=float32)
        control_net_wrapper=None
        label="openpose"
        
        kps = self.box_keypoints(pose_keypoint, select_parts, points_list, 1, dilate, person_number)
        bbox_region = list(kps[:4])
        crop_region=self.round_resolution(bbox_region[0], bbox_region[1], bbox_region[2], bbox_region[3], image_width, image_height, round_by)
        
        crop_l, crop_t, crop_r, crop_b = crop_region[:4]
        left, top, right, bottom = bbox_region[:4]
        
        mask = kps[-1]
        mask[:, top:bottom, left:right] = 1.0
        
        cropped_image = image[:,crop_t:crop_b, crop_l:crop_r]
        cropped_mask = mask[:,crop_t:crop_b, crop_l:crop_r].numpy()
        
        segs_header=(image_height, image_width)
        segs_elt = SEG(cropped_image, cropped_mask, confidence, crop_region, bbox_region, label, control_net_wrapper)
        
        seg=(segs_header, [segs_elt])

        return (seg,)

NODE_CLASS_MAPPINGS = {
    "Openpose Keypoint Extractor": OpenPoseKeyPointExtractor,
    "Openpose SEGS Extractor": OpenPoseSEGSExtractor,
}
