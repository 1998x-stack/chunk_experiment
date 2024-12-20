# DINOv2

## Overview

The DINOv2 model was proposed in [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) by Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski. DINOv2 is an upgrade of [DINO](https://arxiv.org/abs/2104.14294), a self-supervised method applied on [Vision Transformers](vit). This method enables all-purpose visual features, i.e., features that work across image distributions and tasks without finetuning.

The abstract from the paper is the following:

_The recent breakthroughs in natural language processing for model pretraining on large quantities of data have opened the way for similar foundation models in computer vision. These models could greatly simplify the use of images in any system by producing all-purpose visual features, i.e., features that work across image distributions and tasks without finetuning. This work shows that existing pretraining methods, especially self-supervised methods, can produce such features if trained on enough curated data from diverse sources. We revisit existing approaches and combine different techniques to scale our pretraining in terms of data and model size. Most of the technical contributions aim at accelerating and stabilizing the training at scale. In terms of data, we propose an automatic pipeline to build a dedicated, diverse, and curated image dataset instead of uncurated data, as typically done in the self-supervised literature. In terms of models, we train a ViT model (Dosovitskiy et al., 2020) with 1B parameters and distill it into a series of smaller models that surpass the best available all-purpose features, OpenCLIP (Ilharco et al., 2021) on most of the benchmarks at image and pixel levels._

Tips:

-   One can use [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) class to prepare images for the model.

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/facebookresearch/dinov2).

## Dinov2Config

### class transformers.Dinov2Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinov2/configuration_dinov2.py#L35)

( hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12mlp\_ratio = 4hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-06image\_size = 224patch\_size = 16num\_channels = 3qkv\_bias = Truelayerscale\_value = 1.0drop\_path\_rate = 0.0use\_swiglu\_ffn = Falseout\_features = Noneout\_indices = Noneapply\_layernorm = Truereshape\_hidden\_states = True\*\*kwargs )

This is the configuration class to store the configuration of a [Dinov2Model](/docs/transformers/v4.34.0/en/model_doc/dinov2#transformers.Dinov2Model). It is used to instantiate an Dinov2 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Dinov2 [google/dinov2-base-patch16-224](https://huggingface.co/google/dinov2-base-patch16-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Dinov2Config, Dinov2Model

>>> 
>>> configuration = Dinov2Config()

>>> 
>>> model = Dinov2Model(configuration)

>>> 
>>> configuration = model.config
```

## Dinov2Model

### class transformers.Dinov2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinov2/modeling_dinov2.py#L587)

( config: Dinov2Config )

Parameters

-   **config** ([Dinov2Config](/docs/transformers/v4.34.0/en/model_doc/dinov2#transformers.Dinov2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DINOv2 Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinov2/modeling_dinov2.py#L611)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [Dinov2Model](/docs/transformers/v4.34.0/en/model_doc/dinov2#transformers.Dinov2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, Dinov2Model
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
>>> model = Dinov2Model.from_pretrained("facebook/dinov2-base")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 257, 768]
```

## Dinov2ForImageClassification

### class transformers.Dinov2ForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinov2/modeling_dinov2.py#L676)

( config: Dinov2Config )

Parameters

-   **config** ([Dinov2Config](/docs/transformers/v4.34.0/en/model_doc/dinov2#transformers.Dinov2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Dinov2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/dinov2/modeling_dinov2.py#L691)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

The [Dinov2ForImageClassification](/docs/transformers/v4.34.0/en/model_doc/dinov2#transformers.Dinov2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, Dinov2ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
>>> model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-base")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
```