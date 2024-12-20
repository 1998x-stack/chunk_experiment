# Swin Transformer V2

## Overview

The Swin Transformer V2 model was proposed in [Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/abs/2111.09883) by Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo.

The abstract from the paper is the following:

_Large-scale NLP models have been shown to significantly improve the performance on language tasks with no signs of saturation. They also demonstrate amazing few-shot capabilities like that of human beings. This paper aims to explore large-scale models in computer vision. We tackle three major issues in training and application of large vision models, including training instability, resolution gaps between pre-training and fine-tuning, and hunger on labelled data. Three main techniques are proposed: 1) a residual-post-norm method combined with cosine attention to improve training stability; 2) A log-spaced continuous position bias method to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs; 3) A self-supervised pre-training method, SimMIM, to reduce the needs of vast labeled images. Through these techniques, this paper successfully trained a 3 billion-parameter Swin Transformer V2 model, which is the largest dense vision model to date, and makes it capable of training with images of up to 1,536×1,536 resolution. It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification. Also note our training is much more efficient than that in Google’s billion-level visual models, which consumes 40 times less labelled data and 40 times less training time._

Tips:

-   One can use the [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) API to prepare images for the model.

This model was contributed by [nandwalritik](https://huggingface.co/nandwalritik). The original code can be found [here](https://github.com/microsoft/Swin-Transformer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Swin Transformer v2.

-   [Swinv2ForImageClassification](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2ForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

Besides that:

-   [Swinv2ForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2ForMaskedImageModeling) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## Swinv2Config

### class transformers.Swinv2Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swinv2/configuration_swinv2.py#L30)

( image\_size = 224patch\_size = 4num\_channels = 3embed\_dim = 96depths = \[2, 2, 6, 2\]num\_heads = \[3, 6, 12, 24\]window\_size = 7mlp\_ratio = 4.0qkv\_bias = Truehidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0drop\_path\_rate = 0.1hidden\_act = 'gelu'use\_absolute\_embeddings = Falseinitializer\_range = 0.02layer\_norm\_eps = 1e-05encoder\_stride = 32\*\*kwargs )

This is the configuration class to store the configuration of a [Swinv2Model](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2Model). It is used to instantiate a Swin Transformer v2 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Swin Transformer v2 [microsoft/swinv2-tiny-patch4-window8-256](https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Swinv2Config, Swinv2Model

>>> 
>>> configuration = Swinv2Config()

>>> 
>>> model = Swinv2Model(configuration)

>>> 
>>> configuration = model.config
```

## Swinv2Model

### class transformers.Swinv2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swinv2/modeling_swinv2.py#L1024)

( configadd\_pooling\_layer = Trueuse\_mask\_token = False )

Parameters

-   **config** ([Swinv2Config](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Swinv2 Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swinv2/modeling_swinv2.py#L1051)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.swinv2.modeling_swinv2.Swinv2ModelOutput` or `tuple(torch.FloatTensor)`

The [Swinv2Model](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, Swinv2Model
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
>>> model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 64, 768]
```

## Swinv2ForMaskedImageModeling

### class transformers.Swinv2ForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swinv2/modeling_swinv2.py#L1135)

( config )

Parameters

-   **config** ([Swinv2Config](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swinv2 Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

Note that we provide a script to pre-train this model on custom data in our [examples directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swinv2/modeling_swinv2.py#L1152)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.swinv2.modeling_swinv2.Swinv2MaskedImageModelingOutput` or `tuple(torch.FloatTensor)`

The [Swinv2ForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2ForMaskedImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, Swinv2ForMaskedImageModeling
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
>>> model = Swinv2ForMaskedImageModeling.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
>>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
>>> 
>>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
>>> list(reconstructed_pixel_values.shape)
[1, 3, 256, 256]
```

## Swinv2ForImageClassification

### class transformers.Swinv2ForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swinv2/modeling_swinv2.py#L1247)

( config )

Parameters

-   **config** ([Swinv2Config](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swinv2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swinv2/modeling_swinv2.py#L1262)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.swinv2.modeling_swinv2.Swinv2ImageClassifierOutput` or `tuple(torch.FloatTensor)`

The [Swinv2ForImageClassification](/docs/transformers/v4.34.0/en/model_doc/swinv2#transformers.Swinv2ForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, Swinv2ForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
>>> model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
Egyptian cat
```