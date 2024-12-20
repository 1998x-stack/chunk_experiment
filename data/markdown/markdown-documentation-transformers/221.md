# MobileViTV2

## Overview

The MobileViTV2 model was proposed in [Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/abs/2206.02680) by Sachin Mehta and Mohammad Rastegari.

MobileViTV2 is the second version of MobileViT, constructed by replacing the multi-headed self-attention in MobileViT with separable self-attention.

The abstract from the paper is the following:

_Mobile vision transformers (MobileViT) can achieve state-of-the-art performance across several mobile vision tasks, including classification and detection. Though these models have fewer parameters, they have high latency as compared to convolutional neural network-based models. The main efficiency bottleneck in MobileViT is the multi-headed self-attention (MHA) in transformers, which requires O(k2) time complexity with respect to the number of tokens (or patches) k. Moreover, MHA requires costly operations (e.g., batch-wise matrix multiplication) for computing self-attention, impacting latency on resource-constrained devices. This paper introduces a separable self-attention method with linear complexity, i.e. O(k). A simple yet effective characteristic of the proposed method is that it uses element-wise operations for computing self-attention, making it a good choice for resource-constrained devices. The improved model, MobileViTV2, is state-of-the-art on several mobile vision tasks, including ImageNet object classification and MS-COCO object detection. With about three million parameters, MobileViTV2 achieves a top-1 accuracy of 75.6% on the ImageNet dataset, outperforming MobileViT by about 1% while running 3.2× faster on a mobile device._

Tips:

-   MobileViTV2 is more like a CNN than a Transformer model. It does not work on sequence data but on batches of images. Unlike ViT, there are no embeddings. The backbone model outputs a feature map.
-   One can use [MobileViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) to prepare images for the model. Note that if you do your own preprocessing, the pretrained checkpoints expect images to be in BGR pixel order (not RGB).
-   The available image classification checkpoints are pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes).
-   The segmentation model uses a [DeepLabV3](https://arxiv.org/abs/1706.05587) head. The available semantic segmentation checkpoints are pre-trained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).

This model was contributed by [shehan97](https://huggingface.co/shehan97). The original code can be found [here](https://github.com/apple/ml-cvnets).

## MobileViTV2Config

### class transformers.MobileViTV2Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevitv2/configuration_mobilevitv2.py#L34)

( num\_channels = 3image\_size = 256patch\_size = 2expand\_ratio = 2.0hidden\_act = 'swish'conv\_kernel\_size = 3output\_stride = 32classifier\_dropout\_prob = 0.1initializer\_range = 0.02layer\_norm\_eps = 1e-05aspp\_out\_channels = 512atrous\_rates = \[6, 12, 18\]aspp\_dropout\_prob = 0.1semantic\_loss\_ignore\_index = 255n\_attn\_blocks = \[2, 4, 3\]base\_attn\_unit\_dims = \[128, 192, 256\]width\_multiplier = 1.0ffn\_multiplier = 2attn\_dropout = 0.0ffn\_dropout = 0.0\*\*kwargs )

This is the configuration class to store the configuration of a [MobileViTV2Model](/docs/transformers/v4.34.0/en/model_doc/mobilevitv2#transformers.MobileViTV2Model). It is used to instantiate a MobileViTV2 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the MobileViTV2 [apple/mobilevitv2-1.0](https://huggingface.co/apple/mobilevitv2-1.0) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import MobileViTV2Config, MobileViTV2Model

>>> 
>>> configuration = MobileViTV2Config()

>>> 
>>> model = MobileViTV2Model(configuration)

>>> 
>>> configuration = model.config
```

## MobileViTV2Model

### class transformers.MobileViTV2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L665)

( config: MobileViTV2Configexpand\_output: bool = True )

Parameters

-   **config** ([MobileViTV2Config](/docs/transformers/v4.34.0/en/model_doc/mobilevitv2#transformers.MobileViTV2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare MobileViTV2 model outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L699)

( pixel\_values: typing.Optional\[torch.Tensor\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

The [MobileViTV2Model](/docs/transformers/v4.34.0/en/model_doc/mobilevitv2#transformers.MobileViTV2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, MobileViTV2Model
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
>>> model = MobileViTV2Model.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 512, 8, 8]
```

## MobileViTV2ForImageClassification

### class transformers.MobileViTV2ForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L756)

( config: MobileViTV2Config )

Parameters

-   **config** ([MobileViTV2Config](/docs/transformers/v4.34.0/en/model_doc/mobilevitv2#transformers.MobileViTV2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MobileViTV2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [MobileViTV2ForImageClassification](/docs/transformers/v4.34.0/en/model_doc/mobilevitv2#transformers.MobileViTV2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, MobileViTV2ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
>>> model = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## MobileViTV2ForSemanticSegmentation

### class transformers.MobileViTV2ForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevitv2/modeling_mobilevitv2.py#L956)

( config: MobileViTV2Config )

Parameters

-   **config** ([MobileViTV2Config](/docs/transformers/v4.34.0/en/model_doc/mobilevitv2#transformers.MobileViTV2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MobileViTV2 model with a semantic segmentation head on top, e.g. for Pascal VOC.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [MobileViTV2ForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevitv2#transformers.MobileViTV2ForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import requests
>>> import torch
>>> from PIL import Image
>>> from transformers import AutoImageProcessor, MobileViTV2ForSemanticSegmentation

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
>>> model = MobileViTV2ForSemanticSegmentation.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> 
>>> logits = outputs.logits
```