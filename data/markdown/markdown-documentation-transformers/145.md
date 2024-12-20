# Dilated Neighborhood Attention Transformer

## Overview

DiNAT was proposed in [Dilated Neighborhood Attention Transformer](https://arxiv.org/abs/2209.15001) by Ali Hassani and Humphrey Shi.

It extends [NAT](nat) by adding a Dilated Neighborhood Attention pattern to capture global context, and shows significant performance improvements over it.

The abstract from the paper is the following:

_Transformers are quickly becoming one of the most heavily applied deep learning architectures across modalities, domains, and tasks. In vision, on top of ongoing efforts into plain transformers, hierarchical transformers have also gained significant attention, thanks to their performance and easy integration into existing frameworks. These models typically employ localized attention mechanisms, such as the sliding-window Neighborhood Attention (NA) or Swin Transformer’s Shifted Window Self Attention. While effective at reducing self attention’s quadratic complexity, local attention weakens two of the most desirable properties of self attention: long range inter-dependency modeling, and global receptive field. In this paper, we introduce Dilated Neighborhood Attention (DiNA), a natural, flexible and efficient extension to NA that can capture more global context and expand receptive fields exponentially at no additional cost. NA’s local attention and DiNA’s sparse global attention complement each other, and therefore we introduce Dilated Neighborhood Attention Transformer (DiNAT), a new hierarchical vision transformer built upon both. DiNAT variants enjoy significant improvements over strong baselines such as NAT, Swin, and ConvNeXt. Our large model is faster and ahead of its Swin counterpart by 1.5% box AP in COCO object detection, 1.3% mask AP in COCO instance segmentation, and 1.1% mIoU in ADE20K semantic segmentation. Paired with new frameworks, our large variant is the new state of the art panoptic segmentation model on COCO (58.2 PQ) and ADE20K (48.5 PQ), and instance segmentation model on Cityscapes (44.5 AP) and ADE20K (35.4 AP) (no extra data). It also matches the state of the art specialized semantic segmentation models on ADE20K (58.2 mIoU), and ranks second on Cityscapes (84.5 mIoU) (no extra data)._

Tips:

-   One can use the [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) API to prepare images for the model.
-   DiNAT can be used as a _backbone_. When `output_hidden_states = True`, it will output both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, height, width, num_channels)`.

Notes:

-   DiNAT depends on [NATTEN](https://github.com/SHI-Labs/NATTEN/)’s implementation of Neighborhood Attention and Dilated Neighborhood Attention. You can install it with pre-built wheels for Linux by referring to [shi-labs.com/natten](https://shi-labs.com/natten), or build on your system by running `pip install natten`. Note that the latter will likely take time to compile. NATTEN does not support Windows devices yet.
-   Patch size of 4 is only supported at the moment.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dilated-neighborhood-attention-pattern.jpg) Neighborhood Attention with different dilation values. Taken from the [original paper](https://arxiv.org/abs/2209.15001).

This model was contributed by [Ali Hassani](https://huggingface.co/alihassanijr). The original code can be found [here](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DiNAT.

-   [DinatForImageClassification](/docs/transformers/v4.34.0/en/model_doc/dinat#transformers.DinatForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DinatConfig

### class transformers.DinatConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinat/configuration_dinat.py#L30)

( patch\_size = 4num\_channels = 3embed\_dim = 64depths = \[3, 4, 6, 5\]num\_heads = \[2, 4, 8, 16\]kernel\_size = 7dilations = \[\[1, 8, 1\], \[1, 4, 1, 4\], \[1, 2, 1, 2, 1, 2\], \[1, 1, 1, 1, 1\]\]mlp\_ratio = 3.0qkv\_bias = Truehidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0drop\_path\_rate = 0.1hidden\_act = 'gelu'initializer\_range = 0.02layer\_norm\_eps = 1e-05layer\_scale\_init\_value = 0.0out\_features = Noneout\_indices = None\*\*kwargs )

This is the configuration class to store the configuration of a [DinatModel](/docs/transformers/v4.34.0/en/model_doc/dinat#transformers.DinatModel). It is used to instantiate a Dinat model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Dinat [shi-labs/dinat-mini-in1k-224](https://huggingface.co/shi-labs/dinat-mini-in1k-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import DinatConfig, DinatModel

>>> 
>>> configuration = DinatConfig()

>>> 
>>> model = DinatModel(configuration)

>>> 
>>> configuration = model.config
```

## DinatModel

### class transformers.DinatModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinat/modeling_dinat.py#L700)

( configadd\_pooling\_layer = True )

Parameters

-   **config** ([DinatConfig](/docs/transformers/v4.34.0/en/model_doc/dinat#transformers.DinatConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dinat Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinat/modeling_dinat.py#L730)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.dinat.modeling_dinat.DinatModelOutput` or `tuple(torch.FloatTensor)`

The [DinatModel](/docs/transformers/v4.34.0/en/model_doc/dinat#transformers.DinatModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, DinatModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("shi-labs/dinat-mini-in1k-224")
>>> model = DinatModel.from_pretrained("shi-labs/dinat-mini-in1k-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 7, 7, 512]
```

## DinatForImageClassification

### class transformers.DinatForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinat/modeling_dinat.py#L792)

( config )

Parameters

-   **config** ([DinatConfig](/docs/transformers/v4.34.0/en/model_doc/dinat#transformers.DinatConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Dinat Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinat/modeling_dinat.py#L809)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.dinat.modeling_dinat.DinatImageClassifierOutput` or `tuple(torch.FloatTensor)`

The [DinatForImageClassification](/docs/transformers/v4.34.0/en/model_doc/dinat#transformers.DinatForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, DinatForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("shi-labs/dinat-mini-in1k-224")
>>> model = DinatForImageClassification.from_pretrained("shi-labs/dinat-mini-in1k-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```