# ViTMatte

## Overview

The ViTMatte model was proposed in [Boosting Image Matting with Pretrained Plain Vision Transformers](https://arxiv.org/abs/2305.15272) by Jingfeng Yao, Xinggang Wang, Shusheng Yang, Baoyuan Wang. ViTMatte leverages plain [Vision Transformers](vit) for the task of image matting, which is the process of accurately estimating the foreground object in images and videos.

The abstract from the paper is the following:

_Recently, plain vision Transformers (ViTs) have shown impressive performance on various computer vision tasks, thanks to their strong modeling capacity and large-scale pretraining. However, they have not yet conquered the problem of image matting. We hypothesize that image matting could also be boosted by ViTs and present a new efficient and robust ViT-based matting system, named ViTMatte. Our method utilizes (i) a hybrid attention mechanism combined with a convolution neck to help ViTs achieve an excellent performance-computation trade-off in matting tasks. (ii) Additionally, we introduce the detail capture module, which just consists of simple lightweight convolutions to complement the detailed information required by matting. To the best of our knowledge, ViTMatte is the first work to unleash the potential of ViT on image matting with concise adaptation. It inherits many superior properties from ViT to matting, including various pretraining strategies, concise architecture design, and flexible inference strategies. We evaluate ViTMatte on Composition-1k and Distinctions-646, the most commonly used benchmark for image matting, our method achieves state-of-the-art performance and outperforms prior matting works by a large margin._

Tips:

-   The model expects both the image and trimap (concatenated) as input. One can use `ViTMatteImageProcessor` for this purpose.

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/hustvl/ViTMatte).

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vitmatte_architecture.png) ViTMatte high-level overview. Taken from the [original paper.](https://arxiv.org/abs/2305.15272)

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViTMatte.

-   A demo notebook regarding inference with [VitMatteForImageMatting](/docs/transformers/v4.34.0/en/model_doc/vitmatte#transformers.VitMatteForImageMatting), including background replacement, can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ViTMatte).

## VitMatteConfig

### class transformers.VitMatteConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vitmatte/configuration_vitmatte.py#L32)

( backbone\_config: PretrainedConfig = Nonehidden\_size: int = 384batch\_norm\_eps: float = 1e-05initializer\_range: float = 0.02convstream\_hidden\_sizes: typing.List\[int\] = \[48, 96, 192\]fusion\_hidden\_sizes: typing.List\[int\] = \[256, 128, 64, 32\]\*\*kwargs )

Parameters

-   **backbone\_config** (`PretrainedConfig` or `dict`, _optional_, defaults to `VitDetConfig()`) — The configuration of the backbone model.
-   **hidden\_size** (`int`, _optional_, defaults to 384) — The number of input channels of the decoder.
-   **batch\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the batch norm layers.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **convstream\_hidden\_sizes** (`List[int]`, _optional_, defaults to `[48, 96, 192]`) — The output channels of the ConvStream module.
-   **fusion\_hidden\_sizes** (`List[int]`, _optional_, defaults to `[256, 128, 64, 32]`) — The output channels of the Fusion blocks.

This is the configuration class to store the configuration of [VitMatteForImageMatting](/docs/transformers/v4.34.0/en/model_doc/vitmatte#transformers.VitMatteForImageMatting). It is used to instantiate a ViTMatte model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ViTMatte [hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import VitMatteConfig, VitMatteForImageMatting

>>> 
>>> configuration = VitMatteConfig()

>>> 
>>> model = VitMatteForImageMatting(configuration)

>>> 
>>> configuration = model.config
```

Serializes this instance to a Python dictionary. Override the default [to\_dict()](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig.to_dict). Returns: `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,

## VitMatteImageProcessor

### class transformers.VitMatteImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vitmatte/image_processing_vitmatte.py#L41)

( do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: bool = Truesize\_divisibility: int = 32\*\*kwargs )

Constructs a ViTMatte image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vitmatte/image_processing_vitmatte.py#L131)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]trimaps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: typing.Optional\[bool\] = Nonesize\_divisibility: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## VitMatteForImageMatting

### class transformers.VitMatteForImageMatting

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vitmatte/modeling_vitmatte.py#L262)

( config )

Parameters

-   **This** model is a PyTorch \[torch.nn.Module\](https —//pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
-   **it** as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and — behavior. — config ([UperNetConfig](/docs/transformers/v4.34.0/en/model_doc/upernet#transformers.UperNetConfig)): Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViTMatte framework leveraging any vision backbone e.g. for ADE20k, CityScapes.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vitmatte/modeling_vitmatte.py#L273)

( pixel\_values: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.vitmatte.modeling_vitmatte.ImageMattingOutput` or `tuple(torch.FloatTensor)`

The [VitMatteForImageMatting](/docs/transformers/v4.34.0/en/model_doc/vitmatte#transformers.VitMatteForImageMatting) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import VitMatteImageProcessor, VitMatteForImageMatting
>>> import torch
>>> from PIL import Image
>>> from huggingface_hub import hf_hub_download

>>> processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
>>> model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

>>> filepath = hf_hub_download(
...     repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
... )
>>> image = Image.open(filepath).convert("RGB")
>>> filepath = hf_hub_download(
...     repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
... )
>>> trimap = Image.open(filepath).convert("L")

>>> 
>>> inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

>>> with torch.no_grad():
...     alphas = model(**inputs).alphas
>>> print(alphas.shape)
torch.Size([1, 1, 640, 960])
```