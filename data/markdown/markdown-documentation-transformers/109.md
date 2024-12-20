# Big Transfer (BiT)

## Overview

The BiT model was proposed in [Big Transfer (BiT): General Visual Representation Learning](https://arxiv.org/abs/1912.11370) by Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby. BiT is a simple recipe for scaling up pre-training of [ResNet](resnet)\-like architectures (specifically, ResNetv2). The method results in significant improvements for transfer learning.

The abstract from the paper is the following:

_Transfer of pre-trained representations improves sample efficiency and simplifies hyperparameter tuning when training deep neural networks for vision. We revisit the paradigm of pre-training on large supervised datasets and fine-tuning the model on a target task. We scale up pre-training, and propose a simple recipe that we call Big Transfer (BiT). By combining a few carefully selected components, and transferring using a simple heuristic, we achieve strong performance on over 20 datasets. BiT performs well across a surprisingly wide range of data regimes — from 1 example per class to 1M total examples. BiT achieves 87.5% top-1 accuracy on ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark (VTAB). On small datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class, and 97.0% on CIFAR-10 with 10 examples per class. We conduct detailed analysis of the main components that lead to high transfer performance._

Tips:

-   BiT models are equivalent to ResNetv2 in terms of architecture, except that: 1) all batch normalization layers are replaced by [group normalization](https://arxiv.org/abs/1803.08494), 2) [weight standardization](https://arxiv.org/abs/1903.10520) is used for convolutional layers. The authors show that the combination of both is useful for training with large batch sizes, and has a significant impact on transfer learning.

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/google-research/big_transfer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BiT.

-   [BitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/bit#transformers.BitForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## BitConfig

### class transformers.BitConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bit/configuration_bit.py#L29)

( num\_channels = 3embedding\_size = 64hidden\_sizes = \[256, 512, 1024, 2048\]depths = \[3, 4, 6, 3\]layer\_type = 'preactivation'hidden\_act = 'relu'global\_padding = Nonenum\_groups = 32drop\_path\_rate = 0.0embedding\_dynamic\_padding = Falseoutput\_stride = 32width\_factor = 1out\_features = Noneout\_indices = None\*\*kwargs )

This is the configuration class to store the configuration of a [BitModel](/docs/transformers/v4.34.0/en/model_doc/bit#transformers.BitModel). It is used to instantiate an BiT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BiT [google/bit-50](https://huggingface.co/google/bit-50) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BitConfig, BitModel

>>> 
>>> configuration = BitConfig()

>>> 
>>> model = BitModel(configuration)

>>> 
>>> configuration = model.config
```

## BitImageProcessor

### class transformers.BitImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bit/image_processing_bit.py#L50)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BICUBIC: 3>do\_center\_crop: bool = Truecrop\_size: typing.Dict\[str, int\] = Nonedo\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_convert\_rgb: bool = True\*\*kwargs )

Constructs a BiT image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bit/image_processing_bit.py#L165)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: bool = Nonesize: typing.Dict\[str, int\] = Noneresample: Resampling = Nonedo\_center\_crop: bool = Nonecrop\_size: int = Nonedo\_rescale: bool = Nonerescale\_factor: float = Nonedo\_normalize: bool = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_convert\_rgb: bool = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Optional\[transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## BitModel

### class transformers.BitModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bit/modeling_bit.py#L706)

( config )

Parameters

-   **config** ([BitConfig](/docs/transformers/v4.34.0/en/model_doc/bit#transformers.BitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare BiT model outputting raw features without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bit/modeling_bit.py#L724)

( pixel\_values: Tensoroutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

The [BitModel](/docs/transformers/v4.34.0/en/model_doc/bit#transformers.BitModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, BitModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/bit-50")
>>> model = BitModel.from_pretrained("google/bit-50")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 2048, 7, 7]
```

## BitForImageClassification

### class transformers.BitForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bit/modeling_bit.py#L769)

( config )

Parameters

-   **config** ([BitConfig](/docs/transformers/v4.34.0/en/model_doc/bit#transformers.BitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BiT Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [BitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/bit#transformers.BitForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, BitForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/bit-50")
>>> model = BitForImageClassification.from_pretrained("google/bit-50")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tiger cat
```