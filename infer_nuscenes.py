from nuscenes.nuscenes import NuScenes
import os
from pathlib import Path
from pycocotools import mask as mask_utils
from grounding_sam import GroundingSAM, plot_masks_and_boxes
import json
import numpy as np
import cv2
import PIL
from tqdm import tqdm
from typing import List

# declare root dir global
ROOT = Path(__file__).resolve().parents[0]
NUSCENES_DATA = "datasets/nuScenes-mini"
NUSCENES_OUTPUT = "outputs/nuScenes-mini"


def predict_nuscenes(grounding_sam: GroundingSAM, nusc: NuScenes, val_scenes: List[str], relevant_modalities: List[str], save_vis: bool=False):
    """
    Perform object detection and segmentation on NuScenes dataset using GroundingSAM model.
    
    Args:
        grounding_sam (GroundingSAM): Instance of GroundingSAM model.
        nusc (NuScenes): Instance of NuScenes dataset.
        val_scenes (list): List of scene names to process.
        relevant_modalities (list): List of relevant sensor modalities.
        save_vis (bool): Whether to save visualizations of predictions. Default is False.
    """
    categories = [] # ['pedestrian', 'animal', 'car', 'motorcycle', 'bicycle', 'bus', 'truck', 'construction', 'emergency', 'trailer', 'barrier', 'trafficcone', 'pushable_pullable', 'debris', 'bicycle_rack']
    for cat in nusc.category:
        cat = cat['name']
        split = cat.split(".")
        if len(split) > 1 and (not categories or categories[-1] != split[1]):
            categories.append(split[1])
        else:
            categories.append(split[0])

    color_palette = {'animal': [102, 220, 225], 'barrier': [95, 179, 61], 'bicycle': [234, 203, 92],
                     'bicycle_rack': [3, 98, 243], 'bus': [14, 149, 245], 'car': [6, 106, 244],
                     'construction': [99, 187, 71], 'debris': [212, 153, 199], 'emergency': [188, 174, 65],
                     'motorcycle': [153, 20, 44], 'pedestrian': [203, 152, 102], 'pushable_pullable': [214, 240, 39],
                     'trafficcone': [121, 24, 34], 'trailer': [114, 210, 65], 'truck': [239, 39, 214]}
    
    output_preds = []

    for scene in tqdm(nusc.scene):
        if scene['name'] in val_scenes:
            my_scene = scene
            sample_token = my_scene['first_sample_token']

            while sample_token != '':
                sample = nusc.get('sample', sample_token)

                for modality in relevant_modalities:
                    # get the sample data
                    sample_data = nusc.get('sample_data', sample['data'][modality])
                    sample_data_token = sample_data['token']

                    annotations = sample['anns']
                    image_name = sample_data['filename']
                    image_path = ROOT / NUSCENES_DATA / image_name
        
                    # Infer GroundingSAM on this image, and obtain mask
                    bboxes, scores, labels, masks = grounding_sam(image_path, categories)

                    # If there are no visible objects in a frame, skip it
                    if len(bboxes) == 0: 
                        continue

                    for (box, score, label, mask) in zip(bboxes, scores, labels, masks):
                        # use pycocotools to encode mask to RLE
                        mask_rle = mask_utils.encode(np.array(mask[0].cpu(), order='F', dtype=np.uint8))
                        mask_rle['counts'] = mask_rle['counts'].decode('utf-8')
                        pred = {
                            "sample_data_token": sample_data_token,
                            "category": label,
                            "mask": mask_rle,
                            "score": float(score)
                        }
                        output_preds.append(pred)

                    if save_vis:
                        image = cv2.imread(str(image_path))
                        output_img: PIL.Image = plot_masks_and_boxes(
                            image,
                            preds={"boxes": bboxes, "masks": masks, "labels": labels}, 
                            color_palette=color_palette
                        )

                        image_output_path = ROOT / NUSCENES_OUTPUT / image_name
                        os.makedirs(image_output_path.parent, exist_ok=True)
                        output_img.convert("RGB").save(str(image_output_path))

                sample_token = sample['next']

    # Save predictions
    json_save_path = ROOT / NUSCENES_OUTPUT / "mask_results_preds.json"
    os.makedirs(json_save_path.parent, exist_ok=True)
    with open(json_save_path, "w") as f:
        json.dump(output_preds, f, indent=2)
        print(f"Saved mask predictions to {json_save_path}")


def main():
    # Load validation set
    nusc = NuScenes(version='v1.0-mini', dataroot='datasets/nuScenes-mini', verbose=True)
    nusc.list_scenes()
    minival_scenes = ['scene-0103', 'scene-0916']
    relevant_modalities = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    grounding_sam = GroundingSAM()

    predict_nuscenes(grounding_sam, nusc, minival_scenes, relevant_modalities, save_vis=False)


if __name__ == "__main__":
    main()
