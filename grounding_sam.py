import os, sys
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), os.path.join("Grounded-Segment-Anything", "GroundingDINO")))

from pycocotools import mask as mask_utils
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any

import torch

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models.GroundingDINO import groundingdino
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor 
from huggingface_hub import hf_hub_download

# If you have multiple GPUs, you can set the GPU to use here.
# The default is to use the first GPU, which is usually GPU 0.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GroundingSAM:
    def __init__(self):
        self.groundingdino_model = self.init_gdino()
        self.sam_predictor = self.init_sam()

    def __call__(self, image_path: str, categories: List[str], box_threshold: float = 0.3, text_threshold: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.infer(image_path, categories, box_threshold, text_threshold)

    def init_gdino(self) -> groundingdino.GroundingDINO:
        """
        Initialize the Grounding DINO model.

        Returns:
            Grounding DINO model instance.
        """
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        groundingdino_model = self.load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename)
        return groundingdino_model

    def init_sam(self) -> SamPredictor:
        """
        Initialize the SAM (Segment Anything) model.

        Returns:
            SamPredictor: SAM model instance.
        """
        sam_checkpoint = 'Grounded-Segment-Anything/sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=DEVICE)
        sam_predictor = SamPredictor(sam)
        return sam_predictor
    
    def load_model_hf(self, repo_id, filename, ckpt_config_filename, device='cpu'):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file) 
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model  

    def infer(self, image_path: str, categories: List[str], box_threshold: float = 0.3, text_threshold: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform inference to detect objects and segment them in the given image.

        Args:
            image_path (str): Path to the input image.
            categories (List[str]): List of categories to detect.
            box_threshold (float): Threshold for bounding box detection. Default is 0.3.
            text_threshold (float): Threshold for text detection. Default is 0.25.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                Tuple containing the following tensors:
                - bboxes (torch.Tensor): A tensor of shape (n, 4) representing the bounding boxes of the detected objects.
                        Each row contains the coordinates of a bounding box in the format [x_min, y_min, x_max, y_max].
                - logits (torch.Tensor): A tensor of shape (n,) containing the confidence scores corresponding to each detected object.
                - phrases (torch.Tensor): A tensor of shape (n,) containing the category labels of the detected objects.
                - masks (torch.Tensor): A tensor of shape (n, 1, height, width) containing a binary mask for each predicted box.
                        Each mask corresponds to a detected object and has the same height and width as the input image.
        """

        # local_image_path = "/home/atharv21027/cse344-project/open-world-3D-det/datasets/nuScenes-mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"
        image_source, image = load_image(image_path)

        text_prompt = ". ".join(categories)
        boxes, logits, phrases = predict(
            model=self.groundingdino_model, 
            image=image, 
            caption=text_prompt, 
            box_threshold=box_threshold, 
            text_threshold=text_threshold,
            device=DEVICE,
            # token_spans=get_phrase_wise_token_span(TEXT_PROMPT)
        )

        # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # annotated_frame = annotated_frame[...,::-1] # BGR to RGB

        self.sam_predictor.set_image(image_source)

        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(DEVICE)

        # Perform segmentation
        if len(boxes) > 0:
            masks, _, _ = self.sam_predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes,
                        multimask_output = False,
                    )
        else:
            masks = None

        return (boxes_xyxy, logits, phrases, masks) 


def get_phrase_wise_token_span(prompt):
    """ only periods are considered as delimiters """
    span = []
    flag = True
    span_start = 0
    span_end = 0
    phrase = []
    for i in range(len(prompt)):
        if prompt[i] == '.':
            flag = False
            span.append(phrase)
            phrase = []
        
        elif prompt[i] == ' ':
            if flag == True:
                span_end = i
                phrase.append([span_start, span_end])
                flag = False
        
        else:
            if flag == False:
                span_start = i
                flag = True
            if flag == True:
                span_end = i
    
    print(span)
    return span


def plot_masks_and_boxes(image: np.ndarray, preds: dict, color_palette: dict = None, no_labels: bool = False, no_boxes: bool = False, no_masks: bool = False) -> PIL.Image:

    """
    Plot masks and bounding boxes on the image.
    
    Args:
        image (numpy.ndarray): Input image.
        preds (dict): Dictionary containing predictions with keys 'boxes', 'masks', and 'labels'.
                      'boxes': Array of bounding boxes with shape (num_boxes, 4).
                      'masks': Array of binary masks with shape (num_masks, height, width).
                      'labels': Array of class labels corresponding to each prediction.
        random_color (dict): Color palette for visualizing masks by class. Defaults to generating a random palette.
        no_labels (bool): If set to True, does not plot category labels on the image. Defaults to False.
        no_boxes (bool): If set to True, does not plot bounding boxes on the image. Defaults to False.
        no_masks (bool): If set to True, does not plot masks on the image. Defaults to False.
    
    Returns:
        PIL.Image: Annotated image with masks and bounding boxes plotted.
    """
    # Convert image to PIL format
    annotated_image = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    if color_palette is None:
        # Generate a random color palette based on the unique classes in preds['labels']
        num_categories = len(np.unique(preds['labels']))
        np.random.seed(42)
        color_palette = np.random.randint(0, 256, size=(num_categories, 3), dtype=np.uint8)
        color_palette = {label: color for label, color in zip(np.unique(preds['labels']), color_palette)}
    
    # Iterate over each prediction
    for box, mask, label in zip(preds['boxes'], preds['masks'], preds['labels']):
        # if random_color:
        #     color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        # else:
        #     color = np.array([30/255, 144/255, 255/255, 0.6])

        # Plot bounding boxes
        color = color_palette[label]
        if not no_boxes:
            draw.rectangle(box.tolist(), outline=tuple(color))
        
        # Plot masks
        if not no_masks:
            mask = mask.cpu().numpy().astype(np.float32)
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * np.append(color, 255 * 0.6).reshape(1, 1, -1)
            mask_image_pil = Image.fromarray(mask_image.astype(np.uint8), mode="RGBA")
            annotated_image.paste(mask_image_pil, (0, 0), mask_image_pil)
        # annotated_image.alpha_composite(mask_image_pil, (0, 0))

        # Print the label category name along with the box
        # draw.text((box[0], box[1]), str(label), fill=tuple(color), align="center")

        if not no_labels:
            x0, y0 = box[0], box[1]
            if hasattr(font, "getbbox"):
                text_bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                text_bbox = (x0, y0, w + x0, y0 + h)
            draw.rectangle(text_bbox, fill=tuple(color))
            draw.text((x0, y0), str(label), fill="white")

    return annotated_image



if __name__ == "__main__":
    # categories = ["car", "person", "traffic light"]
    categories = ['pedestrian', 'animal', 'car', 'motorcycle', 'bicycle', 'bus', 'truck', 'construction', 'emergency', 'trailer', 'barrier', 'trafficcone', 'pushable_pullable', 'debris', 'bicycle_rack']
    image_path = "/home/atharv21027/cse344-project/open-world-3D-det/datasets/nuScenes-mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg"

    gSAM = GroundingSAM()
    boxes, scores, labels, masks = gSAM(image_path, categories)
    image_source = cv2.imread(image_path)

    # Plot the annotated image
    ann_img = plot_masks_and_boxes(image=image_source, preds={"boxes": boxes, "masks": masks, "labels": labels})
    ann_img.convert("RGB").save("output3.jpeg")

    # plt.imshow(ann_img)
    # plt.axis('off')
    # plt.show()
    # plt.savefig("output2.png", dpi=400)
