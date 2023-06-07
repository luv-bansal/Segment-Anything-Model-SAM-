# Image Segmentation with Text Prompt using Grounding Dino and SAM

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/1e6dece009c468d44bd0e3b414235a42acfcf123c69f219f.png)

## Grounding Dino

Grounding DINO aims to merge concepts found in the [**DINO**](https://arxiv.org/pdf/2203.03605.pdf?ref=blog.roboflow.com) and [**GLIP**](https://arxiv.org/pdf/2112.03857.pdf?ref=blog.roboflow.com) papers. DINO, a transformer-based detection method, **offers state-of-the-art object detection performance** and end-to-end optimization, eliminating the need for handcrafted modules like NMS (Non-Maximum Suppression).

On the other hand, GLIP focuses on **phrase grounding**. This task involves associating phrases or words from a given text with corresponding visual elements in an image or video, effectively linking textual descriptions to their respective visual representations.

## Segment Anything Model (SAM)

The [Segment Anything Model](https://github.com/facebookresearch/segment-anything?) (SAM) is an instance segmentation model developed by Meta Research and released in April, 2023. Segment Anything was trained on [**11 million images and 1.1 billion segmentation masks**](https://github.com/facebookresearch/segment-anything?ref=blog.roboflow.com).

## Grounding DNO to Generate Bounding Boxes

To initiate the annotation process, begin by preparing the desired image. Subsequently, utilize the Grounding DINO model to generate bounding boxes around the objects depicted in the image. These initial bounding boxes will serve as the initial reference for the subsequent instance segmentation procedure.

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/2cb87b64266632a35d7a627d99e0d08367a6544ee0f5f461.png)

## SAM to convert Bounding Boxes to Instance segmentation

Once the bounding boxes have been established, the SAM model can be employed to convert them into instance segmentation masks. The SAM model utilizes the input of bounding box data and produces accurate segmentation masks for individual objects within the image.

![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/7f2f70f97828f789dfd9393ceb0e967d1186b9d5d045e662.png)