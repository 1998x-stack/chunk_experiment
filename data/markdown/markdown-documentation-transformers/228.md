# Neighborhood Attention Transformer

## Overview

NAT was proposed in [Neighborhood Attention Transformer](https://arxiv.org/abs/2204.07143) by Ali Hassani, Steven Walton, Jiachen Li, Shen Li, and Humphrey Shi.

It is a hierarchical vision transformer based on Neighborhood Attention, a sliding-window self attention pattern.

The abstract from the paper is the following:

_We present Neighborhood Attention (NA), the first efficient and scalable sliding-window attention mechanism for vision. NA is a pixel-wise operation, localizing self attention (SA) to the nearest neighboring pixels, and therefore enjoys a linear time and space complexity compared to the quadratic complexity of SA. The sliding-window pattern allows NA’s receptive field to grow without needing extra pixel shifts, and preserves translational equivariance, unlike Swin Transformer’s Window Self Attention (WSA). We develop NATTEN (Neighborhood Attention Extension), a Python package with efficient C++ and CUDA kernels, which allows NA to run up to 40% faster than Swin’s WSA while using up to 25% less memory. We further present Neighborhood Attention Transformer (NAT), a new hierarchical transformer design based on NA that boosts image classification and downstream vision performance. Experimental results on NAT are competitive; NAT-Tiny reaches 83.2% top-1 accuracy on ImageNet, 51.4% mAP on MS-COCO and 48.4% mIoU on ADE20K, which is 1.9% ImageNet accuracy, 1.0% COCO mAP, and 2.6% ADE20K mIoU improvement over a Swin model with similar size._

Tips:

-   One can use the [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) API to prepare images for the model.
-   NAT can be used as a _backbone_. When `output_hidden_states = True`, it will output both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, height, width, num_channels)`.

Notes:

-   NAT depends on [NATTEN](https://github.com/SHI-Labs/NATTEN/)’s implementation of Neighborhood Attention. You can install it with pre-built wheels for Linux by referring to [shi-labs.com/natten](https://shi-labs.com/natten), or build on your system by running `pip install natten`. Note that the latter will likely take time to compile. NATTEN does not support Windows devices yet.
-   Patch size of 4 is only supported at the moment.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/neighborhood-attention-pattern.jpg) Neighborhood Attention compared to other attention patterns. Taken from the [original paper](https://arxiv.org/abs/2204.07143).

This model was contributed by [Ali Hassani](https://huggingface.co/alihassanijr). The original code can be found [here](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with NAT.

-   [NatForImageClassification](/docs/transformers/v4.34.0/en/model_doc/nat#transformers.NatForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## NatConfig

### class transformers.NatConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nat/configuration_nat.py#L30)

( patch\_size = 4num\_channels = 3embed\_dim = 64depths = \[3, 4, 6, 5\]num\_heads = \[2, 4, 8, 16\]kernel\_size = 7mlp\_ratio = 3.0qkv\_bias = Truehidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0drop\_path\_rate = 0.1hidden\_act = 'gelu'initializer\_range = 0.02layer\_norm\_eps = 1e-05layer\_scale\_init\_value = 0.0out\_features = Noneout\_indices = None\*\*kwargs )

This is the configuration class to store the configuration of a [NatModel](/docs/transformers/v4.34.0/en/model_doc/nat#transformers.NatModel). It is used to instantiate a Nat model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Nat [shi-labs/nat-mini-in1k-224](https://huggingface.co/shi-labs/nat-mini-in1k-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import NatConfig, NatModel

>>> 
>>> configuration = NatConfig()

>>> 
>>> model = NatModel(configuration)

>>> 
>>> configuration = model.config
```

## NatModel

### class transformers.NatModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nat/modeling_nat.py#L678)

( configadd\_pooling\_layer = True )

Parameters

-   **config** ([NatConfig](/docs/transformers/v4.34.0/en/model_doc/nat#transformers.NatConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Nat Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nat/modeling_nat.py#L708)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.nat.modeling_nat.NatModelOutput` or `tuple(torch.FloatTensor)`

The [NatModel](/docs/transformers/v4.34.0/en/model_doc/nat#transformers.NatModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, NatModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
>>> model = NatModel.from_pretrained("shi-labs/nat-mini-in1k-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 7, 7, 512]
```

## NatForImageClassification

### class transformers.NatForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nat/modeling_nat.py#L770)

( config )

Parameters

-   **config** ([NatConfig](/docs/transformers/v4.34.0/en/model_doc/nat#transformers.NatConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Nat Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nat/modeling_nat.py#L787)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.nat.modeling_nat.NatImageClassifierOutput` or `tuple(torch.FloatTensor)`

The [NatForImageClassification](/docs/transformers/v4.34.0/en/model_doc/nat#transformers.NatForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, NatForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
>>> model = NatForImageClassification.from_pretrained("shi-labs/nat-mini-in1k-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tiger cat
```