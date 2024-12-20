# GLPN

This is a recently introduced model so the API hasn’t been tested extensively. There may be some bugs or slight breaking changes to fix it in the future. If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title).

## Overview

The GLPN model was proposed in [Global-Local Path Networks for Monocular Depth Estimation with Vertical CutDepth](https://arxiv.org/abs/2201.07436) by Doyeon Kim, Woonghyun Ga, Pyungwhan Ahn, Donggyu Joo, Sehwan Chun, Junmo Kim. GLPN combines [SegFormer](segformer)’s hierarchical mix-Transformer with a lightweight decoder for monocular depth estimation. The proposed decoder shows better performance than the previously proposed decoders, with considerably less computational complexity.

The abstract from the paper is the following:

_Depth estimation from a single image is an important task that can be applied to various fields in computer vision, and has grown rapidly with the development of convolutional neural networks. In this paper, we propose a novel structure and training strategy for monocular depth estimation to further improve the prediction accuracy of the network. We deploy a hierarchical transformer encoder to capture and convey the global context, and design a lightweight yet powerful decoder to generate an estimated depth map while considering local connectivity. By constructing connected paths between multi-scale local features and the global decoding stream with our proposed selective feature fusion module, the network can integrate both representations and recover fine details. In addition, the proposed decoder shows better performance than the previously proposed decoders, with considerably less computational complexity. Furthermore, we improve the depth-specific augmentation method by utilizing an important observation in depth estimation to enhance the model. Our network achieves state-of-the-art performance over the challenging depth dataset NYU Depth V2. Extensive experiments have been conducted to validate and show the effectiveness of the proposed approach. Finally, our model shows better generalisation ability and robustness than other comparative models._

Tips:

-   One can use [GLPNImageProcessor](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNImageProcessor) to prepare images for the model.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/glpn_architecture.jpg) Summary of the approach. Taken from the [original paper](https://arxiv.org/abs/2201.07436).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/vinvino02/GLPDepth).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with GLPN.

-   Demo notebooks for [GLPNForDepthEstimation](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNForDepthEstimation) can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/GLPN).
-   [Monocular depth estimation task guide](../tasks/monocular_depth_estimation)

## GLPNConfig

### class transformers.GLPNConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/glpn/configuration_glpn.py#L29)

( num\_channels = 3num\_encoder\_blocks = 4depths = \[2, 2, 2, 2\]sr\_ratios = \[8, 4, 2, 1\]hidden\_sizes = \[32, 64, 160, 256\]patch\_sizes = \[7, 3, 3, 3\]strides = \[4, 2, 2, 2\]num\_attention\_heads = \[1, 2, 5, 8\]mlp\_ratios = \[4, 4, 4, 4\]hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02drop\_path\_rate = 0.1layer\_norm\_eps = 1e-06decoder\_hidden\_size = 64max\_depth = 10head\_in\_index = -1\*\*kwargs )

This is the configuration class to store the configuration of a [GLPNModel](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNModel). It is used to instantiate an GLPN model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the GLPN [vinvino02/glpn-kitti](https://huggingface.co/vinvino02/glpn-kitti) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import GLPNModel, GLPNConfig

>>> 
>>> configuration = GLPNConfig()

>>> 
>>> model = GLPNModel(configuration)

>>> 
>>> configuration = model.config
```

## GLPNFeatureExtractor

Preprocess an image or a batch of images.

## GLPNImageProcessor

### class transformers.GLPNImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/glpn/image_processing_glpn.py#L40)

( do\_resize: bool = Truesize\_divisor: int = 32resample = <Resampling.BILINEAR: 2>do\_rescale: bool = True\*\*kwargs )

Parameters

-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s (height, width) dimensions, rounding them down to the closest multiple of `size_divisor`. Can be overridden by `do_resize` in `preprocess`.
-   **size\_divisor** (`int`, _optional_, defaults to 32) — When `do_resize` is `True`, images are resized so their height and width are rounded down to the closest multiple of `size_divisor`. Can be overridden by `size_divisor` in `preprocess`.
-   **resample** (`PIL.Image` resampling filter, _optional_, defaults to `PILImageResampling.BILINEAR`) — Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.). Can be overridden by `do_rescale` in `preprocess`.

Constructs a GLPN image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/glpn/image_processing_glpn.py#L124)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), transformers.utils.generic.TensorType, typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[transformers.utils.generic.TensorType\]\]do\_resize: typing.Optional\[bool\] = Nonesize\_divisor: typing.Optional\[int\] = Noneresample = Nonedo\_rescale: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess the given images.

## GLPNModel

### class transformers.GLPNModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/glpn/modeling_glpn.py#L483)

( config )

Parameters

-   **config** ([GLPNConfig](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare GLPN encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [GLPNModel](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, GLPNModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
>>> model = GLPNModel.from_pretrained("vinvino02/glpn-kitti")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 512, 15, 20]
```

## GLPNForDepthEstimation

### class transformers.GLPNForDepthEstimation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/glpn/modeling_glpn.py#L684)

( config )

Parameters

-   **config** ([GLPNConfig](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

GLPN Model transformer with a lightweight depth estimation head on top e.g. for KITTI, NYUv2. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/glpn/modeling_glpn.py#L695)

( pixel\_values: FloatTensorlabels: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

The [GLPNForDepthEstimation](/docs/transformers/v4.34.0/en/model_doc/glpn#transformers.GLPNForDepthEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, GLPNForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
>>> model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-kitti")

>>> 
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)
...     predicted_depth = outputs.predicted_depth

>>> 
>>> prediction = torch.nn.functional.interpolate(
...     predicted_depth.unsqueeze(1),
...     size=image.size[::-1],
...     mode="bicubic",
...     align_corners=False,
... )

>>> 
>>> output = prediction.squeeze().cpu().numpy()
>>> formatted = (output * 255 / np.max(output)).astype("uint8")
>>> depth = Image.fromarray(formatted)
```