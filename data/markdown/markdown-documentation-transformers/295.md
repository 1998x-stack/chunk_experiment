# Hybrid Vision Transformer (ViT Hybrid)

## Overview

The hybrid Vision Transformer (ViT) model was proposed in [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. It’s the first paper that successfully trains a Transformer encoder on ImageNet, attaining very good results compared to familiar convolutional architectures. ViT hybrid is a slight variant of the [plain Vision Transformer](vit), by leveraging a convolutional backbone (specifically, [BiT](bit)) whose features are used as initial “tokens” for the Transformer.

The abstract from the paper is the following:

_While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train._

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code (written in JAX) can be found [here](https://github.com/google-research/vision_transformer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViT Hybrid.

-   [ViTHybridForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit_hybrid#transformers.ViTHybridForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ViTHybridConfig

### class transformers.ViTHybridConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_hybrid/configuration_vit_hybrid.py#L32)

( backbone\_config = Nonehidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-12image\_size = 224patch\_size = 1num\_channels = 3backbone\_featmap\_shape = \[1, 1024, 24, 24\]qkv\_bias = True\*\*kwargs )

This is the configuration class to store the configuration of a [ViTHybridModel](/docs/transformers/v4.34.0/en/model_doc/vit_hybrid#transformers.ViTHybridModel). It is used to instantiate a ViT Hybrid model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ViT Hybrid [google/vit-hybrid-base-bit-384](https://huggingface.co/google/vit-hybrid-base-bit-384) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ViTHybridConfig, ViTHybridModel

>>> 
>>> configuration = ViTHybridConfig()

>>> 
>>> model = ViTHybridModel(configuration)

>>> 
>>> configuration = model.config
```

## ViTHybridImageProcessor

### class transformers.ViTHybridImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_hybrid/image_processing_vit_hybrid.py#L50)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BICUBIC: 3>do\_center\_crop: bool = Truecrop\_size: typing.Dict\[str, int\] = Nonedo\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_convert\_rgb: bool = True\*\*kwargs )

Constructs a ViT Hybrid image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_hybrid/image_processing_vit_hybrid.py#L165)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: bool = Nonesize: typing.Dict\[str, int\] = Noneresample: Resampling = Nonedo\_center\_crop: bool = Nonecrop\_size: int = Nonedo\_rescale: bool = Nonerescale\_factor: float = Nonedo\_normalize: bool = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_convert\_rgb: bool = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Optional\[transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## ViTHybridModel

### class transformers.ViTHybridModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_hybrid/modeling_vit_hybrid.py#L533)

( config: ViTHybridConfigadd\_pooling\_layer: bool = Trueuse\_mask\_token: bool = False )

Parameters

-   **config** ([ViTHybridConfig](/docs/transformers/v4.34.0/en/model_doc/vit_hybrid#transformers.ViTHybridConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare ViT Hybrid Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_hybrid/modeling_vit_hybrid.py#L558)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneinterpolate\_pos\_encoding: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [ViTHybridModel](/docs/transformers/v4.34.0/en/model_doc/vit_hybrid#transformers.ViTHybridModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, ViTHybridModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-hybrid-base-bit-384")
>>> model = ViTHybridModel.from_pretrained("google/vit-hybrid-base-bit-384")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 197, 768]
```

## ViTHybridForImageClassification

### class transformers.ViTHybridForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_hybrid/modeling_vit_hybrid.py#L652)

( config: ViTHybridConfig )

Parameters

-   **config** ([ViTHybridConfig](/docs/transformers/v4.34.0/en/model_doc/vit_hybrid#transformers.ViTHybridConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViT Hybrid Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_hybrid/modeling_vit_hybrid.py#L665)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneinterpolate\_pos\_encoding: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

The [ViTHybridForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit_hybrid#transformers.ViTHybridForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, ViTHybridForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-hybrid-base-bit-384")
>>> model = ViTHybridForImageClassification.from_pretrained("google/vit-hybrid-base-bit-384")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```