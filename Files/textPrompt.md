# Image Segmentation with Text Prompt using Grounding Dino and SAM

![image](https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/4607fd17-a6d8-47eb-a0c8-c3001e011ffb)


## Grounding Dino

Grounding DINO aims to merge concepts found in the [**DINO**](https://arxiv.org/pdf/2203.03605.pdf?ref=blog.roboflow.com) and [**GLIP**](https://arxiv.org/pdf/2112.03857.pdf?ref=blog.roboflow.com) papers. DINO, a transformer-based detection method, **offers state-of-the-art object detection performance** and end-to-end optimization, eliminating the need for handcrafted modules like NMS (Non-Maximum Suppression).

On the other hand, GLIP focuses on **phrase grounding**. This task involves associating phrases or words from a given text with corresponding visual elements in an image or video, effectively linking textual descriptions to their respective visual representations.

## Segment Anything Model (SAM)

The [Segment Anything Model](https://github.com/facebookresearch/segment-anything?) (SAM) is an instance segmentation model developed by Meta Research and released in April, 2023. Segment Anything was trained on [**11 million images and 1.1 billion segmentation masks**](https://github.com/facebookresearch/segment-anything?ref=blog.roboflow.com).

## Grounding DNO to Generate Bounding Boxes

To initiate the annotation process, begin by preparing the desired image. Subsequently, utilize the Grounding DINO model to generate bounding boxes around the objects depicted in the image. These initial bounding boxes will serve as the initial reference for the subsequent instance segmentation procedure.

![image](https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/5e1a6ac6-234b-4d09-a21e-c42bd259088f)

## SAM to convert Bounding Boxes to Instance segmentation

Once the bounding boxes have been established, the SAM model can be employed to convert them into instance segmentation masks. The SAM model utilizes the input of bounding box data and produces accurate segmentation masks for individual objects within the image.

![image](https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/c01061d8-c15e-47c2-8716-59983044bff3)

