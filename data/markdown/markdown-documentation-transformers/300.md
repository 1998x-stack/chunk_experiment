# ViTMSN

## Overview

The ViTMSN model was proposed in [Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141) by Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Michael Rabbat, Nicolas Ballas. The paper presents a joint-embedding architecture to match the prototypes of masked patches with that of the unmasked patches. With this setup, their method yields excellent performance in the low-shot and extreme low-shot regimes.

The abstract from the paper is the following:

_We propose Masked Siamese Networks (MSN), a self-supervised learning framework for learning image representations. Our approach matches the representation of an image view containing randomly masked patches to the representation of the original unmasked image. This self-supervised pre-training strategy is particularly scalable when applied to Vision Transformers since only the unmasked patches are processed by the network. As a result, MSNs improve the scalability of joint-embedding architectures, while producing representations of a high semantic level that perform competitively on low-shot image classification. For instance, on ImageNet-1K, with only 5,000 annotated images, our base MSN model achieves 72.4% top-1 accuracy, and with 1% of ImageNet-1K labels, we achieve 75.7% top-1 accuracy, setting a new state-of-the-art for self-supervised learning on this benchmark._

Tips:

-   MSN (masked siamese networks) is a method for self-supervised pre-training of Vision Transformers (ViTs). The pre-training objective is to match the prototypes assigned to the unmasked views of the images to that of the masked views of the same images.
-   The authors have only released pre-trained weights of the backbone (ImageNet-1k pre-training). So, to use that on your own image classification dataset, use the [ViTMSNForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit_msn#transformers.ViTMSNForImageClassification) class which is initialized from [ViTMSNModel](/docs/transformers/v4.34.0/en/model_doc/vit_msn#transformers.ViTMSNModel). Follow [this notebook](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb) for a detailed tutorial on fine-tuning.
-   MSN is particularly useful in the low-shot and extreme low-shot regimes. Notably, it achieves 75.7% top-1 accuracy with only 1% of ImageNet-1K labels when fine-tuned.

![drawing](https://i.ibb.co/W6PQMdC/Screenshot-2022-09-13-at-9-08-40-AM.png) MSN architecture. Taken from the [original paper.](https://arxiv.org/abs/2204.07141)

This model was contributed by [sayakpaul](https://huggingface.co/sayakpaul). The original code can be found [here](https://github.com/facebookresearch/msn).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViT MSN.

-   [ViTMSNForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit_msn#transformers.ViTMSNForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ViTMSNConfig

### class transformers.ViTMSNConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_msn/configuration_vit_msn.py#L30)

( hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-06image\_size = 224patch\_size = 16num\_channels = 3qkv\_bias = True\*\*kwargs )

This is the configuration class to store the configuration of a [ViTMSNModel](/docs/transformers/v4.34.0/en/model_doc/vit_msn#transformers.ViTMSNModel). It is used to instantiate an ViT MSN model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ViT [facebook/vit\_msn\_base](https://huggingface.co/facebook/vit_msn_base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ViTMSNModel, ViTMSNConfig

>>> 
>>> configuration = ViTConfig()

>>> 
>>> model = ViTMSNModel(configuration)

>>> 
>>> configuration = model.config
```

## ViTMSNModel

### class transformers.ViTMSNModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_msn/modeling_vit_msn.py#L492)

( config: ViTMSNConfiguse\_mask\_token: bool = False )

Parameters

-   **config** ([ViTMSNConfig](/docs/transformers/v4.34.0/en/model_doc/vit_msn#transformers.ViTMSNConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare ViTMSN Model outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_msn/modeling_vit_msn.py#L516)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneinterpolate\_pos\_encoding: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

The [ViTMSNModel](/docs/transformers/v4.34.0/en/model_doc/vit_msn#transformers.ViTMSNModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, ViTMSNModel
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
>>> model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")
>>> inputs = image_processor(images=image, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## ViTMSNForImageClassification

### class transformers.ViTMSNForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_msn/modeling_vit_msn.py#L601)

( config: ViTMSNConfig )

Parameters

-   **config** ([ViTMSNConfig](/docs/transformers/v4.34.0/en/model_doc/vit_msn#transformers.ViTMSNConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViTMSN Model with an image classification head on top e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_msn/modeling_vit_msn.py#L614)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneinterpolate\_pos\_encoding: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

The [ViTMSNForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit_msn#transformers.ViTMSNForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, ViTMSNForImageClassification
>>> import torch
>>> from PIL import Image
>>> import requests

>>> torch.manual_seed(2)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
>>> model = ViTMSNForImageClassification.from_pretrained("facebook/vit-msn-small")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
Kerry blue terrier
```