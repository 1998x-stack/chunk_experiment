# Monocular depth estimation

Monocular depth estimation is a computer vision task that involves predicting the depth information of a scene from a single image. In other words, it is the process of estimating the distance of objects in a scene from a single camera viewpoint.

Monocular depth estimation has various applications, including 3D reconstruction, augmented reality, autonomous driving, and robotics. It is a challenging task as it requires the model to understand the complex relationships between objects in the scene and the corresponding depth information, which can be affected by factors such as lighting conditions, occlusion, and texture.

The task illustrated in this tutorial is supported by the following model architectures:

[DPT](../model_doc/dpt), [GLPN](../model_doc/glpn)

In this guide you’ll learn how to:

-   create a depth estimation pipeline
-   run depth estimation inference by hand

Before you begin, make sure you have all the necessary libraries installed:

```
pip install -q transformers
```

## Depth estimation pipeline

The simplest way to try out inference with a model supporting depth estimation is to use the corresponding [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline). Instantiate a pipeline from a [checkpoint on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads):

```
>>> from transformers import pipeline

>>> checkpoint = "vinvino02/glpn-nyu"
>>> depth_estimator = pipeline("depth-estimation", model=checkpoint)
```

Next, choose an image to analyze:

```
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/HwBAsSbPBDU/download?ixid=MnwxMjA3fDB8MXxzZWFyY2h8MzR8fGNhciUyMGluJTIwdGhlJTIwc3RyZWV0fGVufDB8MHx8fDE2Nzg5MDEwODg&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> image
```

![Photo of a busy street](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-estimation-example.jpg)

Pass the image to the pipeline.

```
>>> predictions = depth_estimator(image)
```

The pipeline returns a dictionary with two entries. The first one, called `predicted_depth`, is a tensor with the values being the depth expressed in meters for each pixel. The second one, `depth`, is a PIL image that visualizes the depth estimation result.

Let’s take a look at the visualized result:

![Depth estimation visualization](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization.png)

## Depth estimation inference by hand

Now that you’ve seen how to use the depth estimation pipeline, let’s see how we can replicate the same result by hand.

Start by loading the model and associated processor from a [checkpoint on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads). Here we’ll use the same checkpoint as before:

```
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation

>>> checkpoint = "vinvino02/glpn-nyu"

>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
```

Prepare the image input for the model using the `image_processor` that will take care of the necessary image transformations such as resizing and normalization:

```
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values
```

Pass the prepared inputs through the model:

```
>>> import torch

>>> with torch.no_grad():
...     outputs = model(pixel_values)
...     predicted_depth = outputs.predicted_depth
```

Visualize the results:

```
>>> import numpy as np

>>> 
>>> prediction = torch.nn.functional.interpolate(
...     predicted_depth.unsqueeze(1),
...     size=image.size[::-1],
...     mode="bicubic",
...     align_corners=False,
... ).squeeze()
>>> output = prediction.numpy()

>>> formatted = (output * 255 / np.max(output)).astype("uint8")
>>> depth = Image.fromarray(formatted)
>>> depth
```

![Depth estimation visualization](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization.png)