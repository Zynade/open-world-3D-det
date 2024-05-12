# Introduction
We present an open world 3D object detector that does not require 3D annotations. 

# Documentation

## Preliminaries
We provide our conda environment to facilitate ease of reproducibility of our results. To build the required dependencies, run
```bash
conda env create --file environment.yml
conda activate zs3d
```

## Datasets
Store your datasets in the `datasets/` directory. To benchmark on nuScenes, you may extract the nuScenes minival data as `datasets/nuScenes-mini/`. Note that if you already have the data downloaded on another directory, you do not need to create a copy. You may create a symlink:
```bash
cd datasets
ln -s <path/to/dataset> nuScenes-mini
```

## GroundingSAM
`grounding_sam.py` defines a pipeline for running inference on GroundingSAM. This pipeline is defined in the `GroundingSAM` class, and the inference functionality is exposed via a public method which can be directly called on the class object.

Example:
```py
gsam = GroundingSAM()
box, score, label, mask = gsam("path/to/image", ["categories", "to", "detect"])
```

Additionally, `grounding_sam.py` offers a function for visualizing object bounding boxes, instance segmentation masks and category labels in `plot_masks_and_boxes()`.

## Benchmarking on nuScenes 
`infer_nuscenes.py` runs GroundingSAM on each of the validation sequences in nuScenes mini. We store the model predictions in a json format similar to the COCO format for 2D object detection, defined as follows:
```
[
    {
        "sample_data_token" : str       // nuScenes sample data token
        "category"          : str       // category label 
        "mask"              : RLE       // binary mask encoded in RLE format, similar to COCO
        "score"             : float     // detector confidence
    },

    .
    .
    .
]
```
The sample data token uniquely identify a sample, i.e., a keyframe in a nuScenes sequence along with the appropriate modality. This serves as an analog to image_id used in the COCO prediction format.

Instead of storing category_id, we store the raw category string owing to the open-world problem setting.

By default, the predictions are stored in `outputs/nuScenes-mini/mask_results_preds.json`.

Inferring the GroundingSAM pipeline on nuScenes minival takes 12 minutes on one NVIDIA A100 GPU.

To backproject LiDAR onto the masks produced by GroundingSAM and create 3D bboxes, run:
```bash
python load_lidar_nuscenes.py
```
Change the variables at the top of `load_lidar_nuscenes.py` according to your environment.


To evaluate the predictions, run:
```bash
python eval_custom.py outputs/nuScenes-mini/predictions_naive.json --eval_set mini_val --version v1.0-trainval --dataroot /data2/mehark/nuScenes/nuScenes/ --verbose 10
```



# Acknowledgement
- [SAM](https://github.com/facebookresearch/segment-anything)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)

# Contributors
Atharv Goel | [contact](mailto:atharv21027@iiitd.ac.in) 

Mehar Khurana | [contact](mailto:mehar21541@iiitd.ac.in) 

Prakhar Gupta | [contact](mailto:prakhar21550@iiitd.ac.in) 