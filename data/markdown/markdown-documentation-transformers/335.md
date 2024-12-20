# DPT

## Overview

The DPT model was proposed in [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413) by René Ranftl, Alexey Bochkovskiy, Vladlen Koltun. DPT is a model that leverages the [Vision Transformer (ViT)](vit) as backbone for dense prediction tasks like semantic segmentation and depth estimation.

The abstract from the paper is the following:

_We introduce dense vision transformers, an architecture that leverages vision transformers in place of convolutional networks as a backbone for dense prediction tasks. We assemble tokens from various stages of the vision transformer into image-like representations at various resolutions and progressively combine them into full-resolution predictions using a convolutional decoder. The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent predictions when compared to fully-convolutional networks. Our experiments show that this architecture yields substantial improvements on dense prediction tasks, especially when a large amount of training data is available. For monocular depth estimation, we observe an improvement of up to 28% in relative performance when compared to a state-of-the-art fully-convolutional network. When applied to semantic segmentation, dense vision transformers set a new state of the art on ADE20K with 49.02% mIoU. We further show that the architecture can be fine-tuned on smaller datasets such as NYUv2, KITTI, and Pascal Context where it also sets the new state of the art._

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dpt_architecture.jpg) DPT architecture. Taken from the [original paper](https://arxiv.org/abs/2103.13413).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/isl-org/DPT).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DPT.

-   Demo notebooks for [DPTForDepthEstimation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForDepthEstimation) can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DPT).
    
-   [Semantic segmentation task guide](../tasks/semantic_segmentation)
    
-   [Monocular depth estimation task guide](../tasks/monocular_depth_estimation)
    

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DPTConfig

### class transformers.DPTConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/configuration_dpt.py#L32)

( hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-12image\_size = 384patch\_size = 16num\_channels = 3is\_hybrid = Falseqkv\_bias = Truebackbone\_out\_indices = \[2, 5, 8, 11\]readout\_type = 'project'reassemble\_factors = \[4, 2, 1, 0.5\]neck\_hidden\_sizes = \[96, 192, 384, 768\]fusion\_hidden\_size = 256head\_in\_index = -1use\_batch\_norm\_in\_fusion\_residual = Falseuse\_auxiliary\_head = Trueauxiliary\_loss\_weight = 0.4semantic\_loss\_ignore\_index = 255semantic\_classifier\_dropout = 0.1backbone\_featmap\_shape = \[1, 1024, 24, 24\]neck\_ignore\_stages = \[0, 1\]backbone\_config = None\*\*kwargs )

This is the configuration class to store the configuration of a [DPTModel](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTModel). It is used to instantiate an DPT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the DPT [Intel/dpt-large](https://huggingface.co/Intel/dpt-large) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import DPTModel, DPTConfig

>>> 
>>> configuration = DPTConfig()

>>> 
>>> model = DPTModel(configuration)

>>> 
>>> configuration = model.config
```

Serializes this instance to a Python dictionary. Override the default [to\_dict()](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig.to_dict). Returns: `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,

## DPTFeatureExtractor

Preprocess an image or a batch of images.

( outputstarget\_sizes: typing.List\[typing.Tuple\] = None ) → semantic\_segmentation

Parameters

-   **outputs** ([DPTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForSemanticSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple]` of length `batch_size`, _optional_) — List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

`List[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [DPTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## DPTImageProcessor

### class transformers.DPTImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/image_processing_dpt.py#L94)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>keep\_aspect\_ratio: bool = Falseensure\_multiple\_of: int = 1do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None\*\*kwargs )

Constructs a DPT image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/image_processing_dpt.py#L211)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: bool = Nonesize: int = Nonekeep\_aspect\_ratio: bool = Noneensure\_multiple\_of: int = Noneresample: Resampling = Nonedo\_rescale: bool = Nonerescale\_factor: float = Nonedo\_normalize: bool = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

#### post\_process\_semantic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/image_processing_dpt.py#L346)

( outputstarget\_sizes: typing.List\[typing.Tuple\] = None ) → semantic\_segmentation

Parameters

-   **outputs** ([DPTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForSemanticSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple]` of length `batch_size`, _optional_) — List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

Returns

semantic\_segmentation

`List[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [DPTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## DPTModel

### class transformers.DPTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/modeling_dpt.py#L864)

( configadd\_pooling\_layer = True )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DPT Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/modeling_dpt.py#L896)

( pixel\_values: FloatTensorhead\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.dpt.modeling_dpt.BaseModelOutputWithPoolingAndIntermediateActivations` or `tuple(torch.FloatTensor)`

The [DPTModel](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, DPTModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
>>> model = DPTModel.from_pretrained("Intel/dpt-large")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 577, 1024]
```

## DPTForDepthEstimation

### class transformers.DPTForDepthEstimation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/modeling_dpt.py#L1052)

( config )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/modeling_dpt.py#L1067)

( pixel\_values: FloatTensorhead\_mask: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.DepthEstimatorOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.DepthEstimatorOutput) or `tuple(torch.FloatTensor)`

The [DPTForDepthEstimation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForDepthEstimation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DPTForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
>>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

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

## DPTForSemanticSegmentation

### class transformers.DPTForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/modeling_dpt.py#L1220)

( config )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DPT Model with a semantic segmentation head on top e.g. for ADE20k, CityScapes.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dpt/modeling_dpt.py#L1236)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

The [DPTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/dpt#transformers.DPTForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DPTForSemanticSegmentation
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large-ade")
>>> model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```