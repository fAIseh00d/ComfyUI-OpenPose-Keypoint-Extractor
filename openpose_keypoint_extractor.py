import json
from nodes import MAX_RESOLUTION
from collections import namedtuple

import torch
import numpy as np
from numpy import array, float32

openpose_parts = ["pose_keypoints_2d","face_keypoints_2d","hand_left_keypoints_2d","hand_right_keypoints_2d"]
round_divs = [1, 8,16,32,64,128]
body_points = "0, 3, 4, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17" 
#body_points = "0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17"

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

    @staticmethod
    def round_resolution(min_x, min_y, max_x, max_y, i_width, i_height, DIV):
        min_x=max(round(min_x/DIV-0.5)*DIV, 0)
        min_y=max(round(min_y/DIV-0.5)*DIV, 0)
        max_x=max(round(max_x/DIV+0.5)*DIV, 64)
        max_y=max(round(max_y/DIV+0.5)*DIV, 64)
        
        max_x=min(max_x, i_width)
        max_y=min(max_y, i_height)
        min_x=min(i_width-(max_x - min_x), min_x)
        min_y=min(i_height-(max_y - min_y), min_y)
        return [int(min_x), int(min_y), int(max_x), int(max_y)]

    @staticmethod
    def get_keypoint_from_list(list, item):
        idx_x = item*3
        idx_y = idx_x + 1
        idx_conf = idx_y + 1
        return [list[idx_x], list[idx_y], list[idx_conf]]

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
        
        min_x, min_y, max_x, max_y = self.round_resolution(
            min_x - dilate,
            min_y - dilate,
            max_x + dilate,
            max_y + dilate,
            canvas_width,
            canvas_height,
            round_by,
        )
        
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
                "points_list": ("STRING", {"multiline": True, "default": "0, 1"}),
                "round_by": (round_divs, { "default": round_divs[0] }),
                "dilate_bbox": ("INT", { "min": 0, "max": 128 }, { "default": 0 }),
                "dilate_crop": ("INT", { "min": 0, "max": 128 }, { "default": 0 }),
                "person_number": ("INT", { "default": 0 }),
            }
        }

    RETURN_TYPES = ("SEGS",)
    RETURN_NAMES = ("segs",)
    FUNCTION = "extract_SEGS"
    CATEGORY = "utils"
    
    def extract_SEGS(self, image, pose_keypoint, select_parts, points_list, round_by, dilate_bbox, dilate_crop, person_number):
        
        image_width = image.shape[2]
        image_height = image.shape[1]
        confidence=array(1.0, dtype=float32)
        control_net_wrapper=None
        label="openpose"
        
        kps = self.box_keypoints(pose_keypoint, select_parts, points_list, 1, dilate_bbox, person_number)
        bbox_region = list(kps[:4])
        crop_region = self.round_resolution(
            bbox_region[0] - dilate_crop,
            bbox_region[1] - dilate_crop,
            bbox_region[2] + dilate_crop,
            bbox_region[3] + dilate_crop,
            image_width,
            image_height,
            round_by,
        )
        
        crop_l, crop_t, crop_r, crop_b = crop_region[:4]
        left, top, right, bottom = bbox_region
        
        mask = kps[-1]
        mask[:, top:bottom, left:right] = 1.0
        
        cropped_image = image[:,crop_t:crop_b, crop_l:crop_r]
        cropped_mask = mask[:,crop_t:crop_b, crop_l:crop_r].squeeze(0).numpy()
        
        segs_header=(image_height, image_width)
        segs_elt = SEG(cropped_image, cropped_mask, confidence, crop_region, bbox_region, label, control_net_wrapper)
        
        seg=(segs_header, [segs_elt])

        return (seg,)

class OpenPoseJsonLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_json_file": ("STRING", {"multiline": True, "default": "dwpose/keypoints"}),
                "points_list": ("STRING", {"multiline": True, "default": body_points}),
            },
            "optional": {
                "JSON": ("JSON", {"default": None, "forceInput": True}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT", {"default": None, "forceInput": True})
            }
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", )
    FUNCTION = "main"
    CATEGORY = "utils"
    
    def json_convertor(self, pose_keypoint, points_we_want):
        from .open_pose import PoseResult
        from .open_pose.body import BodyResult, Keypoint
        
        keypoints = []
        
        keypoints_list = pose_keypoint["people"][0]["pose_keypoints_2d"]
        width = pose_keypoint['canvas_width']
        height = pose_keypoint['canvas_height']
        
        for element in range(0, len(keypoints_list)//3):
            kps = OpenPoseKeyPointExtractor.get_keypoint_from_list(keypoints_list, element)
            if kps[0]>1.0 and kps [1]>1.0: 
                kps[0]/=width
                kps[1]/=height
            kps = Keypoint(kps[0], kps[1])
            
            if element not in points_we_want: kps = None
            keypoints.append(kps)
            
        body = BodyResult(keypoints=keypoints, total_score=36.0, total_parts=18)
        result = PoseResult(body=body, left_hand=None, right_hand=None, face=None)
        
        return result
        

    def main(self, pose_json_file, points_list, JSON=None, POSE_KEYPOINT=None):
        from .open_pose import draw_poses
        points_we_want = [int(element) for element in points_list.split(",")]

        if JSON:
            pose_json = json.loads(JSON)
            pose_json = pose_json['extra'].strip('\n[]').replace("'", '"') #For crystools embedded JSON from file
            pose_json = json.loads(pose_json)
        elif POSE_KEYPOINT: 
            pose_json = POSE_KEYPOINT[0]
        else:
            with open(pose_json_file, "r") as f:
                pose_json = json.loads(f.read().strip('\n[]').replace("'", '"'))

        poses = [self.json_convertor(pose_json, points_we_want)]

        canvas = draw_poses(poses, pose_json['canvas_height'], pose_json['canvas_width'], draw_body=True, draw_hand=False, draw_face=False) 
        canvas = torch.from_numpy(canvas.astype(np.float32)/255.0)[None,]
        
        return canvas, [pose_json]

NODE_CLASS_MAPPINGS = {
    "Openpose Keypoint Extractor": OpenPoseKeyPointExtractor,
    "Openpose SEGS Extractor": OpenPoseSEGSExtractor,
    "Load DWPose JSON": OpenPoseJsonLoader,
}
