# Swin2SR

## Overview

The Swin2SR model was proposed in [Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration](https://arxiv.org/abs/2209.11345) by Marcos V. Conde, Ui-Jin Choi, Maxime Burchi, Radu Timofte. Swin2R improves the [SwinIR](https://github.com/JingyunLiang/SwinIR/) model by incorporating [Swin Transformer v2](swinv2) layers which mitigates issues such as training instability, resolution gaps between pre-training and fine-tuning, and hunger on data.

The abstract from the paper is the following:

_Compression plays an important role on the efficient transmission and storage of images and videos through band-limited systems such as streaming services, virtual reality or videogames. However, compression unavoidably leads to artifacts and the loss of the original information, which may severely degrade the visual quality. For these reasons, quality enhancement of compressed images has become a popular research topic. While most state-of-the-art image restoration methods are based on convolutional neural networks, other transformers-based methods such as SwinIR, show impressive performance on these tasks. In this paper, we explore the novel Swin Transformer V2, to improve SwinIR for image super-resolution, and in particular, the compressed input scenario. Using this method we can tackle the major issues in training transformer vision models, such as training instability, resolution gaps between pre-training and fine-tuning, and hunger on data. We conduct experiments on three representative tasks: JPEG compression artifacts removal, image super-resolution (classical and lightweight), and compressed image super-resolution. Experimental results demonstrate that our method, Swin2SR, can improve the training convergence and performance of SwinIR, and is a top-5 solution at the “AIM 2022 Challenge on Super-Resolution of Compressed Image and Video”._

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/swin2sr_architecture.png) Swin2SR architecture. Taken from the [original paper.](https://arxiv.org/abs/2209.11345)

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/mv-lab/swin2sr).

## Resources

Demo notebooks for Swin2SR can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Swin2SR).

A demo Space for image super-resolution with SwinSR can be found [here](https://huggingface.co/spaces/jjourney1125/swin2sr).

## Swin2SRImageProcessor

### class transformers.Swin2SRImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin2sr/image_processing_swin2sr.py#L38)

( do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_pad: bool = Truepad\_size: int = 8\*\*kwargs )

Parameters

-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.

Constructs a Swin2SR image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin2sr/image_processing_swin2sr.py#L109)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_pad: typing.Optional\[bool\] = Nonepad\_size: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## Swin2SRConfig

### class transformers.Swin2SRConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin2sr/configuration_swin2sr.py#L30)

( image\_size = 64patch\_size = 1num\_channels = 3embed\_dim = 180depths = \[6, 6, 6, 6, 6, 6\]num\_heads = \[6, 6, 6, 6, 6, 6\]window\_size = 8mlp\_ratio = 2.0qkv\_bias = Truehidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0drop\_path\_rate = 0.1hidden\_act = 'gelu'use\_absolute\_embeddings = Falseinitializer\_range = 0.02layer\_norm\_eps = 1e-05upscale = 2img\_range = 1.0resi\_connection = '1conv'upsampler = 'pixelshuffle'\*\*kwargs )

This is the configuration class to store the configuration of a [Swin2SRModel](/docs/transformers/v4.34.0/en/model_doc/swin2sr#transformers.Swin2SRModel). It is used to instantiate a Swin Transformer v2 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Swin Transformer v2 [caidas/swin2sr-classicalsr-x2-64](https://huggingface.co/caidas/swin2sr-classicalsr-x2-64) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Swin2SRConfig, Swin2SRModel

>>> 
>>> configuration = Swin2SRConfig()

>>> 
>>> model = Swin2SRModel(configuration)

>>> 
>>> configuration = model.config
```

## Swin2SRModel

### class transformers.Swin2SRModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin2sr/modeling_swin2sr.py#L847)

( config )

Parameters

-   **config** ([Swin2SRConfig](/docs/transformers/v4.34.0/en/model_doc/swin2sr#transformers.Swin2SRConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Swin2SR Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin2sr/modeling_swin2sr.py#L896)

( pixel\_values: FloatTensorhead\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

The [Swin2SRModel](/docs/transformers/v4.34.0/en/model_doc/swin2sr#transformers.Swin2SRModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, Swin2SRModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
>>> model = Swin2SRModel.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 180, 488, 648]
```

## Swin2SRForImageSuperResolution

### class transformers.Swin2SRForImageSuperResolution

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin2sr/modeling_swin2sr.py#L1101)

( config )

Parameters

-   **config** ([Swin2SRConfig](/docs/transformers/v4.34.0/en/model_doc/swin2sr#transformers.Swin2SRConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swin2SR Model transformer with an upsampler head on top for image super resolution and restoration.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin2sr/modeling_swin2sr.py#L1128)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.ImageSuperResolutionOutput` or `tuple(torch.FloatTensor)`

The [Swin2SRForImageSuperResolution](/docs/transformers/v4.34.0/en/model_doc/swin2sr#transformers.Swin2SRForImageSuperResolution) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

>>> processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
>>> model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

>>> url = "https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> 
>>> inputs = processor(image, return_tensors="pt")

>>> 
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
>>> output = np.moveaxis(output, source=0, destination=-1)
>>> output = (output * 255.0).round().astype(np.uint8)  
>>> 
```