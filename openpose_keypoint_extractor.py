import json
from nodes import MAX_RESOLUTION
from collections import namedtuple

import torch
import numpy as np
from numpy import array, float32

openpose_parts = ["pose_keypoints_2d","face_keypoints_2d","hand_left_keypoints_2d","hand_right_keypoints_2d"]
round_divs = [8,16,32,64]

SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])

class OpenPoseKeyPointExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "points_list": ("STRING", {"multiline": True, "default": ""}),
                "dwpose": ([False, True], { "default": False }),
                "rounding_on": ([False, True], { "default": False }),
                "round_by": (round_divs, { "default": round_divs[3] }),
                "dilate": ("INT", { "min": 0, "max": 128 }, { "default": 0 }),
                "select_parts": (openpose_parts, { "default": openpose_parts[0]}),
            },
            "optional": {
                "person_number": ("INT", { "default": 0 }),
                "image_width": ("INT"),
                "image_height": ("INT"),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("min_x", "min_y", "max_x", "max_y", "width", "height")
    FUNCTION = "box_keypoints"
    CATEGORY = "utils"

    def get_keypoint_from_list(self, list, item):
        idx_x = item*3
        idx_y = idx_x + 1
        idx_conf = idx_y + 1
        return (list[idx_x], list[idx_y], list[idx_conf])

    def box_keypoints(self, pose_keypoint, points_list, dwpose, rounding_on, round_by, dilate, select_parts ,person_number=0, image_width=None, image_height=None):
        pose_keypoint = pose_keypoint[0]
        
        if not image_width: image_width = pose_keypoint["canvas_width"]
        if not image_height: image_height = pose_keypoint["canvas_height"]
        
        points_we_want = [int(element) for element in points_list.split(",")]

        min_x = MAX_RESOLUTION
        min_y = MAX_RESOLUTION
        max_x = 0
        max_y = 0
        for element in points_we_want:
            (x,y,_) = self.get_keypoint_from_list(pose_keypoint["people"][person_number][select_parts], element)
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        
        min_x = max(min_x - dilate, 0)
        min_y = max(min_y - dilate, 0)
        max_x = min(max_x + dilate, image_width)
        max_y = min(max_y + dilate, image_height)
        
        if rounding_on:
            DIV = round_by
            min_x=max(round(min_x/DIV-0.5)*DIV, 0)
            min_y=max(round(min_y/DIV-0.5)*DIV, 0)
            max_x=max(round(max_x/DIV+0.5)*DIV, 64)
            max_y=max(round(max_y/DIV+0.5)*DIV, 64)
                
        min_x=min(image_width-(max_x - min_x), min_x)
        min_y=min(image_height-(max_y - min_y), min_y)
        
        width=max_x - min_x
        height=max_y - min_y
        
        if not dwpose:
            min_x*=image_width
            min_y*=image_height
            width*=image_width
            height*=image_height
        
        return tuple(map(int, [min_x, min_y, max_x, max_y, width, height]))
    
class OpenPoseSEGSExtractor(OpenPoseKeyPointExtractor):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "pose_keypoint": ("POSE_KEYPOINT",),
                "points_list": ("STRING", {"multiline": True, "default": ""}),
                "dwpose": ([False, True], { "default": False }),
                "rounding_on": ([False, True], { "default": False }),
                "round_by": (round_divs, { "default": round_divs[3] }),
                "dilate": ("INT", { "min": 0, "max": 128 }, { "default": 0 }),
                "select_parts": (openpose_parts, { "default": openpose_parts[0]}),
            },
            "optional": {
                "person_number": ("INT", { "default": 0 }),
            }
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "extract_SEGS"
    CATEGORY = "utils"
    
    def extract_SEGS(self, image, pose_keypoint, points_list, dwpose, rounding_on, round_by, dilate, select_parts, person_number=0):
        
        image_width = pose_keypoint[0]["canvas_width"]
        image_height = pose_keypoint[0]["canvas_height"]
        confidence=array(1.0, dtype=float32)
        control_net_wrapper=None
        label="openpose"
        
        crop_region=list(self.box_keypoints(pose_keypoint, points_list, dwpose, rounding_on, round_by, dilate, select_parts,person_number)[:4])
        bbox_region=list(self.box_keypoints(pose_keypoint, points_list, dwpose, False, round_by, dilate, select_parts,person_number)[:4])
        
        crop_l, crop_t, crop_r, crop_b = crop_region[:4]
        left, top, right, bottom = bbox_region[:4]
        
        
        mask = torch.zeros_like(image[..., 0:1])
        print(f"\n, coords left:{left}, top:{top}, right:{right}, bottom:{bottom}")
        mask[:, top:bottom, left:right] = 1.0
        print("\nMask Shape:",mask.shape)
        
        cropped_image = image[:,crop_t:crop_b, crop_l:crop_r]
        cropped_mask = mask[:,crop_t:crop_b, crop_l:crop_r]
        print("\nCropped Mask Shape:",mask.shape)
        
        
        segs_header=(image_height, image_width)
        segs_elt = SEG(cropped_image, cropped_mask, confidence, crop_region, bbox_region, label, control_net_wrapper)
        
        seg=(segs_header, [segs_elt])

        return (seg,)

NODE_CLASS_MAPPINGS = {
    "Openpose Keypoint Extractor": OpenPoseKeyPointExtractor,
    "Openpose SEGS Extractor": OpenPoseSEGSExtractor,
}
