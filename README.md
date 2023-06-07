# Image Segmentation using Segment Anything Model (SAM)

Meta's new prompt-based AI model allows zero-shot generalization for any segmentation tasks without the need for additional training. 

<!-- ![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/animations/cf9c238ae7e0726e0cb383e844e2919f86d8f865e8dd8953.gif)  ![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/animations/501a97d189380e5a5ffbb3b7f9cd6d45c84ffffb8abe4c22.gif) -->





<p align='center'>
<img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/0eebc0a0-c243-4c79-897e-c929ee321e13" width="400" height="400" style="padding-right:30" >
<img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/53991037-a588-4e8b-b1ea-defb4a05cd1c" width="400" height="400" >
</p>

## 1\. Overview

Foundation Models are powerful models that are trained over a massive amount of unlabeled data, which can be used for different applications with minimal fine-tuning.  This eliminates the need to generate and organize huge datasets and train the AI models to recognize everything for every specific use case. They are already popular in the field of NLP and now, **Meta's Segment Anything Project** has developed a solid base to introduce these foundation models in computer vision as well. Segment Anything Model can extract any object from an image using a variety of prompts as an input.

From here on, the discussion will focus on the two vital components of the project- 

* The largest segmentation dataset ever
* The Segment Anything Model (SAM) - a promptable foundation model for image segmentation

## 2\. The Segment Anything Model (SAM)

As discussed, the earlier models, while useful, take an extensive amount of resources and time. Each specific task requires a specially curated, well-organised dataset which needs a lot of time for data collection and manual annotation itself, along with the long hours of intensive training. Furthermore, in making changes to the dataset, considerable hours of retraining is also necessary. 

SAM is a transformer-based deep learning model which seeks to resolve these problems by allowing a very high-powered generalization of a range of downstream segmentation problems by pre-training it on an extensive dataset comprising of images and masks. This image segmentation model uses prompt engineering and otherwise, needs very minimal human involvement. 

### 2.1 Zero-shot generalization in SAM

Foundation models like SAM learn a general notion of what objects are,such that unfamiliar objects and images can be dealt with, without requiring additional training. This is called Zero-shot generalization.

On careful assessment, it is realized that SAM has a very effective zero-shot performance – at-par with or even superior to earlier completely supervised models .


<p align='center'>
<img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/168e91a3-598b-4680-8894-08a0e15d75c5" width="32%" height="300">
<img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/e4a9fc41-0802-4b05-90ca-f446232d2451" width="32%" height="300">
<img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/a306b2e4-78f2-447b-8f3f-0135d1d3f736" width="32%" height="300">
</p>

### 2.2 SAM as a Promptable Image Segmentation Model

<p align='center' float="left">
<img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/ce2a17ff-45ef-47ec-ac1a-e86a98ed0197"  >
</p>


SAM can take prompts from users about which area to segment out precisely. The following prompts can be provided to SAM as inputs: 

* By clicking on a point
* By drawing a bounding box
* By drawing a rough mask on an object
* By giving text prompts (not released)

Since the text prompt for SAM has not been released by meta, to read more about image segmentation with text prompt using SAM and Grounding Dino, you can refer to my article [here](https://onlinemarkdowneditor.dev/collaboration/#doce8829bf55f).


<p align='center' float="left">
<img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/e3085f17-6589-450e-aa3c-f9ccdbc0af71" width="400" height="400">
<img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/e343d3dd-9de7-4dbb-adcd-b5c5b09e441d" width="400" height="400">
</p>


### 2.3 Model Architecture

<p align='center'>
<img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/ad870e6d-79b8-4829-8bf6-08465183420b" width="75%" height="600">

</p>
<p align='center' float="left">
  <img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/402ebe03-4f7f-4985-a76e-1f837f034726">
</p>
Any image given as an input, first passes through an encoder which produces a one-time embedding for the input. 

There is also a prompt encoder for **points**, **boxes**, or **text as prompts**. For points, the x & y coordinates, along with the foreground and background information, become input to the encoder. For boxes, the bounding box coordinates become the input to the encoder, and as for the text (not released at the time of writing this), the tokens become the input.

In case we provide a mask as input, it directly goes through a downsampling stage. The downsampling happens using 2D convolutional layers. Then the model concatenates it with the image embedding to get the final vector. 

Any vector that the model gets from the prompt vector + image embedding passes through a **lightweight decoder** that creates the final segmentation mask. 

We get possible valid masks along with a confidence score as the output.

* **SAM Image Encoder :** The image encoder is one of the most powerful and essential components of SAM. It is built upon an **MAE pre-trained** [**Vision Transformer**](https://learnopencv.com/the-future-of-image-recognition-is-here-pytorch-vision-transformer/) model. The image encoder runs once per image and can be applied prior to prompting the model.
* **Prompt Encoder :** For the prompt encoder, points, boxes, and text act as sparse inputs, and masks act as dense inputs. The creators of SAM represent points and bounding boxes using positional encodings and sum it with learned embeddings. For text prompts, SAM uses the text encoder from CLIP. For masks as prompts, after downsampling happens through convolutional layers, the embedding is summed element-wise with the input image embedding.
* **Mask Decoder:** The mask decoder efficiently maps the image embedding, prompt embeddings, and an output token to a mask.

## 3\. The Segment Anything Dataset

The foundation of any groundbreaking deep learning model is the dataset it’s trained on. And it’s no different for the Segmentation Anything Model as well.

The Segment Anything Dataset contains more than **11 million images** and **1.1 billion masks**. The final dataset is called **SA-1B** dataset.

After annotating enough masks with SAM’s help, we were able to leverage SAM’s sophisticated ambiguity-aware design to annotate new images fully automatically. To do this, we present SAM with a grid of points on an image and ask SAM to segment everything at each point. Our final dataset includes more than 1.1 billion segmentation masks collected on ~11 million licensed and privacy-preserving images.

[Explore the dataset](https://segment-anything.com/dataset/index.html)    [Download the dataset](https://ai.facebook.com/datasets/segment-anything/)

## 4\. Data Engine in SAM

Meta developed a data engine to collect a dataset, **SA-1B** of 1.1 billion segmentation masks, as these masks are not abundant on the internet.

To enhance SAM's capabilities, researchers employed a model-in-the-loop data engine, using SAM to interactively annotate images and update the model. This iterative process of using SAM for annotation and training improved both the model and the dataset.


<p align="center">
  <img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/4e93e176-679b-47e5-9d94-216337d54768" width="400" height="400" >
</p>


The data engine has three stages:

* Assisted-Manual Stage
* Semi-Automatic Stage
* Fully Automatic Stage

### 5\. Code Implementation

Here is the [code implementation](https://colab.research.google.com/drive/1QmqOcXNT4h5SBbB1bsEjaFW4hqXwdXR2?usp=sharing) of Automated Mask Generation: Generate Segmentation with Bounding Box and Generate Segmentation with a text prompt.

## 6\. Evaluation Metric and Quantitative Results of SAM

* **Intersection-Over-Union:** IoU is the area of overlap between the predicted segmentation and the ground truth divided by the area of union between the predicted segmentation and the ground truth
* **Average Mask Rating:** Metric for mask quality on image segmentation. mask quality on a scale of 1 (nonsense) to 10 (pixel-perfect).
* **Efficiency.** The overall model design is largely motivated by efficiency. Given a precomputed image embedding, the prompt encoder and mask decoder runs in a web browser, on CPU, in ∼50ms. This runtime performance enables seamless, real-time interactive prompting of our model.
* 
<p align='center' float="left">
  <img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/09bbc5be-5c62-46d7-a11a-755a8a6a83f7">
  <img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/def7ba92-c936-47f7-82de-302bc8dbe200">
</p>


The results show that SAM outperforms the strong RITM baseline on 16 of the 23 datasets in terms of automatic evaluation using mIoU. With an oracle to resolve ambiguity, SAM outperforms RITM on all datasets.

## 7\. Applications

* **Image editing:** SAM can be used to quickly and easily remove objects from images, such as people, cars, or buildings. This can be useful for privacy reasons, or to create more creative and interesting images. For example, a photographer could use SAM to remove a distracting person from the background of a portrait, or a graphic designer could use SAM to remove a logo from a product image.

<p align='center' float="left">
  <img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/c9842fa0-7cdb-4028-a396-5cee9b0ced56">
</p>
<!-- ![](https://33333.cdn.cke-cs.com/kSW7V9NHUXugvhoQeFaf/images/13ff19911d67ae3c70915d5e4060d4d774fbc376027aa2a9.png) -->

* **Data Annotation:** Data annotation using the Segment Anything model offers versatile applications in the field of computer vision. This model enables precise annotation and segmentation of a wide range of objects, irrespective of their complexity or category. It can be applied in areas such as autonomous driving, where accurately annotating different road elements like lanes, traffic signs, and pedestrians is crucial for developing robust self-driving systems.

<p align='center' float="left">
  <img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/e0d644a3-284f-4919-b018-c1ef06306fba" width="70%" height="375" >
</p>


* **Medical image analysis:** SAM can be used to segment organs and tissues in medical images, such as MRI scans and CT scans. This can be useful for diagnosis and treatment planning. For example, a doctor could use SAM to identify a tumor in a patient's brain, or a surgeon could use SAM to plan the best way to remove a tumor.

<p align='center' float="left">
  <img src="https://github.com/luv-bansal/Segment-Anything-Model-SAM-/assets/70321430/a988f2cd-d51a-4993-b4a8-eb1e00a92381">
</p>

## 8\. Limitations

• **Require strong prior knowledge.** During the usage of SAM, we observe that for complex scenes, e.g., crop segmentation and fundus image segmentation, more manual prompts with prior knowledge are required, which could potentially result in a suboptimal user experience. Additionally, we notice that SAM tends to favor selecting the foreground mask. When applying the SAM model to shadow detection task, even with a large number of click prompts, its performance remains poor. This may be due to the strong foreground bias in its pre-training dataset, which hinders its ability to handle certain scenarios effectively. 

• **Less effective in low-contrast applications.** Segmenting objects with similar surrounding elements is considered challenging scenarios, especially when dealing with transparent or camouflaged objects that are “seamlessly” embedded in their surroundings. Experiments reveal that there is considerable room for exploring and enhancing SAM’s robustness in complex scenes with low-contrast elements. 

• **Limited understanding of professional data**. We apply SAM to real-world medical and industrial scenarios and discover that it produces unsatisfactory results for professional data, particularly when using box mode and everything mode. This reveals SAM’s limitations in understanding these practical scenarios. Moreover, even with click mode, both the user and the model are required to possess certain domain-specific knowledge and understanding of the task at hand. 

• **Smaller and irregular objects can pose challenges for SAM.** Remote sensing and agriculture present additional challenges, such as irregular buildings and small-sized streets captured from the aerial imaging sensors. These complexities make it challenging for SAM to produce complete segmentation. How to design effective strategies for SAM in such cases is still an open issue.

## References

1. [https://arxiv.org/pdf/2304.02643.pdf](https://arxiv.org/pdf/2304.02643.pdf)
2. [https://segment-anything.com/](https://segment-anything.com/)
3. [https://learnopencv.com/segment-anything/](https://learnopencv.com/segment-anything/)
4. [https://blog.roboflow.com/grounding-dino-zero-shot-object-detection/](https://blog.roboflow.com/grounding-dino-zero-shot-object-detection/)
