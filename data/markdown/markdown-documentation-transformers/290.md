# UPerNet

## Overview

The UPerNet model was proposed in [Unified Perceptual Parsing for Scene Understanding](https://arxiv.org/abs/1807.10221) by Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, Jian Sun. UPerNet is a general framework to effectively segment a wide range of concepts from images, leveraging any vision backbone like [ConvNeXt](convnext) or [Swin](swin).

The abstract from the paper is the following:

_Humans recognize the visual world at multiple levels: we effortlessly categorize scenes and detect objects inside, while also identifying the textures and surfaces of the objects along with their different compositional parts. In this paper, we study a new task called Unified Perceptual Parsing, which requires the machine vision systems to recognize as many visual concepts as possible from a given image. A multi-task framework called UPerNet and a training strategy are developed to learn from heterogeneous image annotations. We benchmark our framework on Unified Perceptual Parsing and show that it is able to effectively segment a wide range of concepts from images. The trained networks are further applied to discover visual knowledge in natural scenes._

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/upernet_architecture.jpg) UPerNet framework. Taken from the [original paper](https://arxiv.org/abs/1807.10221).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code is based on OpenMMLab’s mmsegmentation [here](https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/uper_head.py).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with UPerNet.

-   Demo notebooks for UPerNet can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/UPerNet).
-   [UperNetForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/semantic_segmentation.ipynb).
-   See also: [Semantic segmentation task guide](../tasks/semantic_segmentation)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## Usage

UPerNet is a general framework for semantic segmentation. It can be used with any vision backbone, like so:

```
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation

backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)
```

To use another vision backbone, like [ConvNeXt](convnext), simply instantiate the model with the appropriate backbone:

```
from transformers import ConvNextConfig, UperNetConfig, UperNetForSemanticSegmentation

backbone_config = ConvNextConfig(out_features=["stage1", "stage2", "stage3", "stage4"])

config = UperNetConfig(backbone_config=backbone_config)
model = UperNetForSemanticSegmentation(config)
```

Note that this will randomly initialize all the weights of the model.

## UperNetConfig

### class transformers.UperNetConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/upernet/configuration_upernet.py#L26)

( backbone\_config = Nonehidden\_size = 512initializer\_range = 0.02pool\_scales = \[1, 2, 3, 6\]use\_auxiliary\_head = Trueauxiliary\_loss\_weight = 0.4auxiliary\_in\_channels = 384auxiliary\_channels = 256auxiliary\_num\_convs = 1auxiliary\_concat\_input = Falseloss\_ignore\_index = 255\*\*kwargs )

This is the configuration class to store the configuration of an [UperNetForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation). It is used to instantiate an UperNet model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the UperNet [openmmlab/upernet-convnext-tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import UperNetConfig, UperNetForSemanticSegmentation

>>> 
>>> configuration = UperNetConfig()

>>> 
>>> model = UperNetForSemanticSegmentation(configuration)

>>> 
>>> configuration = model.config
```

## UperNetForSemanticSegmentation

### class transformers.UperNetForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/upernet/modeling_upernet.py#L353)

( config )

Parameters

-   **This** model is a PyTorch \[torch.nn.Module\](https —//pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
-   **it** as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and — behavior. — config ([UperNetConfig](/docs/transformers/v4.34.0/en/model_doc/upernet#transformers.UperNetConfig)): Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

UperNet framework leveraging any vision backbone e.g. for ADE20k, CityScapes.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/upernet/modeling_upernet.py#L366)

( pixel\_values: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

The [UperNetForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/upernet#transformers.UperNetForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
>>> from PIL import Image
>>> from huggingface_hub import hf_hub_download

>>> image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
>>> model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

>>> filepath = hf_hub_download(
...     repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
... )
>>> image = Image.open(filepath).convert("RGB")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)

>>> logits = outputs.logits  
>>> list(logits.shape)
[1, 150, 512, 512]
```