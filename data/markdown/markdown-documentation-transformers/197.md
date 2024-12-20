# LeViT

## Overview

The LeViT model was proposed in [LeViT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2104.01136) by Ben Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze. LeViT improves the [Vision Transformer (ViT)](vit) in performance and efficiency by a few architectural differences such as activation maps with decreasing resolutions in Transformers and the introduction of an attention bias to integrate positional information.

The abstract from the paper is the following:

_We design a family of image classification architectures that optimize the trade-off between accuracy and efficiency in a high-speed regime. Our work exploits recent findings in attention-based architectures, which are competitive on highly parallel processing hardware. We revisit principles from the extensive literature on convolutional neural networks to apply them to transformers, in particular activation maps with decreasing resolutions. We also introduce the attention bias, a new way to integrate positional information in vision transformers. As a result, we propose LeVIT: a hybrid neural network for fast inference image classification. We consider different measures of efficiency on different hardware platforms, so as to best reflect a wide range of application scenarios. Our extensive experiments empirically validate our technical choices and show they are suitable to most architectures. Overall, LeViT significantly outperforms existing convnets and vision transformers with respect to the speed/accuracy tradeoff. For example, at 80% ImageNet top-1 accuracy, LeViT is 5 times faster than EfficientNet on CPU._

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/levit_architecture.png) LeViT Architecture. Taken from the [original paper](https://arxiv.org/abs/2104.01136).

Tips:

-   Compared to ViT, LeViT models use an additional distillation head to effectively learn from a teacher (which, in the LeViT paper, is a ResNet like-model). The distillation head is learned through backpropagation under supervision of a ResNet like-model. They also draw inspiration from convolution neural networks to use activation maps with decreasing resolutions to increase the efficiency.
-   There are 2 ways to fine-tune distilled models, either (1) in a classic way, by only placing a prediction head on top of the final hidden state and not using the distillation head, or (2) by placing both a prediction head and distillation head on top of the final hidden state. In that case, the prediction head is trained using regular cross-entropy between the prediction of the head and the ground-truth label, while the distillation prediction head is trained using hard distillation (cross-entropy between the prediction of the distillation head and the label predicted by the teacher). At inference time, one takes the average prediction between both heads as final prediction. (2) is also called “fine-tuning with distillation”, because one relies on a teacher that has already been fine-tuned on the downstream dataset. In terms of models, (1) corresponds to [LevitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitForImageClassification) and (2) corresponds to [LevitForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher).
-   All released checkpoints were pre-trained and fine-tuned on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes). only. No external data was used. This is in contrast with the original ViT model, which used external data like the JFT-300M dataset/Imagenet-21k for pre-training.
-   The authors of LeViT released 5 trained LeViT models, which you can directly plug into [LevitModel](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitModel) or [LevitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitForImageClassification). Techniques like data augmentation, optimization, and regularization were used in order to simulate training on a much larger dataset (while only using ImageNet-1k for pre-training). The 5 variants available are (all trained on images of size 224x224): _facebook/levit-128S_, _facebook/levit-128_, _facebook/levit-192_, _facebook/levit-256_ and _facebook/levit-384_. Note that one should use [LevitImageProcessor](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitImageProcessor) in order to prepare images for the model.
-   [LevitForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher) currently supports only inference and not training or fine-tuning.
-   You can check out demo notebooks regarding inference as well as fine-tuning on custom data [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) (you can just replace [ViTFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTFeatureExtractor) by [LevitImageProcessor](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitImageProcessor) and [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification) by [LevitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitForImageClassification) or [LevitForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher)).

This model was contributed by [anugunj](https://huggingface.co/anugunj). The original code can be found [here](https://github.com/facebookresearch/LeViT).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LeViT.

-   [LevitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## LevitConfig

### class transformers.LevitConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/levit/configuration_levit.py#L35)

( image\_size = 224num\_channels = 3kernel\_size = 3stride = 2padding = 1patch\_size = 16hidden\_sizes = \[128, 256, 384\]num\_attention\_heads = \[4, 8, 12\]depths = \[4, 4, 4\]key\_dim = \[16, 16, 16\]drop\_path\_rate = 0mlp\_ratio = \[2, 2, 2\]attention\_ratio = \[2, 2, 2\]initializer\_range = 0.02\*\*kwargs )

This is the configuration class to store the configuration of a [LevitModel](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitModel). It is used to instantiate a LeViT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the LeViT [facebook/levit-128S](https://huggingface.co/facebook/levit-128S) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import LevitConfig, LevitModel

>>> 
>>> configuration = LevitConfig()

>>> 
>>> model = LevitModel(configuration)

>>> 
>>> configuration = model.config
```

## LevitFeatureExtractor

Preprocess an image or a batch of images.

## LevitImageProcessor

### class transformers.LevitImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/levit/image_processing_levit.py#L45)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BICUBIC: 3>do\_center\_crop: bool = Truecrop\_size: typing.Dict\[str, int\] = Nonedo\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.Iterable\[float\], NoneType\] = \[0.485, 0.456, 0.406\]image\_std: typing.Union\[float, typing.Iterable\[float\], NoneType\] = \[0.229, 0.224, 0.225\]\*\*kwargs )

Constructs a LeViT image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/levit/image_processing_levit.py#L173)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample: Resampling = Nonedo\_center\_crop: typing.Optional\[bool\] = Nonecrop\_size: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.Iterable\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.Iterable\[float\], NoneType\] = Nonereturn\_tensors: typing.Optional\[transformers.utils.generic.TensorType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images to be used as input to a LeViT model.

## LevitModel

### class transformers.LevitModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/levit/modeling_levit.py#L544)

( config )

Parameters

-   **config** ([LevitConfig](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Levit model outputting raw features without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/levit/modeling_levit.py#L553)

( pixel\_values: FloatTensor = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

The [LevitModel](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, LevitModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/levit-128S")
>>> model = LevitModel.from_pretrained("facebook/levit-128S")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 16, 384]
```

## LevitForImageClassification

### class transformers.LevitForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/levit/modeling_levit.py#L604)

( config )

Parameters

-   **config** ([LevitConfig](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Levit Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [LevitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, LevitForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/levit-128S")
>>> model = LevitForImageClassification.from_pretrained("facebook/levit-128S")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## LevitForImageClassificationWithTeacher

### class transformers.LevitForImageClassificationWithTeacher

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/levit/modeling_levit.py#L691)

( config )

Parameters

-   **config** ([LevitConfig](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LeViT Model transformer with image classification heads on top (a linear layer on top of the final hidden state and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet. .. warning:: This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet supported.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/levit/modeling_levit.py#L713)

( pixel\_values: FloatTensor = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.levit.modeling_levit.LevitForImageClassificationWithTeacherOutput` or `tuple(torch.FloatTensor)`

The [LevitForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/levit#transformers.LevitForImageClassificationWithTeacher) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, LevitForImageClassificationWithTeacher
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/levit-128S")
>>> model = LevitForImageClassificationWithTeacher.from_pretrained("facebook/levit-128S")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```