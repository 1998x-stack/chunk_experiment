# PoolFormer

## Overview

The PoolFormer model was proposed in [MetaFormer is Actually What You Need for Vision](https://arxiv.org/abs/2111.11418) by Sea AI Labs. Instead of designing complicated token mixer to achieve SOTA performance, the target of this work is to demonstrate the competence of transformer models largely stem from the general architecture MetaFormer.

The abstract from the paper is the following:

_Transformers have shown great potential in computer vision tasks. A common belief is their attention-based token mixer module contributes most to their competence. However, recent works show the attention-based module in transformers can be replaced by spatial MLPs and the resulted models still perform quite well. Based on this observation, we hypothesize that the general architecture of the transformers, instead of the specific token mixer module, is more essential to the model’s performance. To verify this, we deliberately replace the attention module in transformers with an embarrassingly simple spatial pooling operator to conduct only the most basic token mixing. Surprisingly, we observe that the derived model, termed as PoolFormer, achieves competitive performance on multiple computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves 82.1% top-1 accuracy, surpassing well-tuned vision transformer/MLP-like baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer parameters and 48%/60% fewer MACs. The effectiveness of PoolFormer verifies our hypothesis and urges us to initiate the concept of “MetaFormer”, a general architecture abstracted from transformers without specifying the token mixer. Based on the extensive experiments, we argue that MetaFormer is the key player in achieving superior results for recent transformer and MLP-like models on vision tasks. This work calls for more future research dedicated to improving MetaFormer instead of focusing on the token mixer modules. Additionally, our proposed PoolFormer could serve as a starting baseline for future MetaFormer architecture design._

The figure below illustrates the architecture of PoolFormer. Taken from the [original paper](https://arxiv.org/abs/2111.11418).

![](https://user-images.githubusercontent.com/15921929/142746124-1ab7635d-2536-4a0e-ad43-b4fe2c5a525d.png)

Tips:

-   PoolFormer has a hierarchical architecture, where instead of Attention, a simple Average Pooling layer is present. All checkpoints of the model can be found on the [hub](https://huggingface.co/models?other=poolformer).
-   One can use [PoolFormerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/poolformer#transformers.PoolFormerImageProcessor) to prepare images for the model.
-   As most models, PoolFormer comes in different sizes, the details of which can be found in the table below.

| **Model variant** | **Depths** | **Hidden sizes** | **Params (M)** | **ImageNet-1k Top 1** |
| --- | --- | --- | --- | --- |
| s12 | \[2, 2, 6, 2\] | \[64, 128, 320, 512\] | 12 | 77.2 |
| s24 | \[4, 4, 12, 4\] | \[64, 128, 320, 512\] | 21 | 80.3 |
| s36 | \[6, 6, 18, 6\] | \[64, 128, 320, 512\] | 31 | 81.4 |
| m36 | \[6, 6, 18, 6\] | \[96, 192, 384, 768\] | 56 | 82.1 |
| m48 | \[8, 8, 24, 8\] | \[96, 192, 384, 768\] | 73 | 82.5 |

This model was contributed by [heytanay](https://huggingface.co/heytanay). The original code can be found [here](https://github.com/sail-sg/poolformer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with PoolFormer.

-   [PoolFormerForImageClassification](/docs/transformers/v4.34.0/en/model_doc/poolformer#transformers.PoolFormerForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## PoolFormerConfig

### class transformers.PoolFormerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/poolformer/configuration_poolformer.py#L34)

( num\_channels = 3patch\_size = 16stride = 16pool\_size = 3mlp\_ratio = 4.0depths = \[2, 2, 6, 2\]hidden\_sizes = \[64, 128, 320, 512\]patch\_sizes = \[7, 3, 3, 3\]strides = \[4, 2, 2, 2\]padding = \[2, 1, 1, 1\]num\_encoder\_blocks = 4drop\_path\_rate = 0.0hidden\_act = 'gelu'use\_layer\_scale = Truelayer\_scale\_init\_value = 1e-05initializer\_range = 0.02\*\*kwargs )

This is the configuration class to store the configuration of [PoolFormerModel](/docs/transformers/v4.34.0/en/model_doc/poolformer#transformers.PoolFormerModel). It is used to instantiate a PoolFormer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the PoolFormer [sail/poolformer\_s12](https://huggingface.co/sail/poolformer_s12) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import PoolFormerConfig, PoolFormerModel

>>> 
>>> configuration = PoolFormerConfig()

>>> 
>>> model = PoolFormerModel(configuration)

>>> 
>>> configuration = model.config
```

## PoolFormerFeatureExtractor

Preprocess an image or a batch of images.

## PoolFormerImageProcessor

### class transformers.PoolFormerImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/poolformer/image_processing_poolformer.py#L49)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Nonecrop\_pct: int = 0.9resample: Resampling = <Resampling.BICUBIC: 3>do\_center\_crop: bool = Truecrop\_size: typing.Dict\[str, int\] = Nonerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_rescale: bool = Truedo\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None\*\*kwargs )

Constructs a PoolFormer image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/poolformer/image_processing_poolformer.py#L211)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: bool = Nonesize: typing.Dict\[str, int\] = Nonecrop\_pct: int = Noneresample: Resampling = Nonedo\_center\_crop: bool = Nonecrop\_size: typing.Dict\[str, int\] = Nonedo\_rescale: bool = Nonerescale\_factor: float = Nonedo\_normalize: bool = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## PoolFormerModel

### class transformers.PoolFormerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/poolformer/modeling_poolformer.py#L313)

( config )

Parameters

-   **config** ([PoolFormerConfig](/docs/transformers/v4.34.0/en/model_doc/poolformer#transformers.PoolFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare PoolFormer Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/poolformer/modeling_poolformer.py#L326)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.BaseModelOutputWithNoAttention` or `tuple(torch.FloatTensor)`

The [PoolFormerModel](/docs/transformers/v4.34.0/en/model_doc/poolformer#transformers.PoolFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, PoolFormerModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("sail/poolformer_s12")
>>> model = PoolFormerModel.from_pretrained("sail/poolformer_s12")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 512, 7, 7]
```

## PoolFormerForImageClassification

### class transformers.PoolFormerForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/poolformer/modeling_poolformer.py#L380)

( config )

Parameters

-   **config** ([PoolFormerConfig](/docs/transformers/v4.34.0/en/model_doc/poolformer#transformers.PoolFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

PoolFormer Model transformer with an image classification head on top

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [PoolFormerForImageClassification](/docs/transformers/v4.34.0/en/model_doc/poolformer#transformers.PoolFormerForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, PoolFormerForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("sail/poolformer_s12")
>>> model = PoolFormerForImageClassification.from_pretrained("sail/poolformer_s12")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```