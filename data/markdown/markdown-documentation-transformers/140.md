# DeiT

This is a recently introduced model so the API hasn’t been tested extensively. There may be some bugs or slight breaking changes to fix it in the future. If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title).

## Overview

The DeiT model was proposed in [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) by Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou. The [Vision Transformer (ViT)](vit) introduced in [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929) has shown that one can match or even outperform existing convolutional neural networks using a Transformer encoder (BERT-like). However, the ViT models introduced in that paper required training on expensive infrastructure for multiple weeks, using external data. DeiT (data-efficient image transformers) are more efficiently trained transformers for image classification, requiring far less data and far less computing resources compared to the original ViT models.

The abstract from the paper is the following:

_Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. However, these visual transformers are pre-trained with hundreds of millions of images using an expensive infrastructure, thereby limiting their adoption. In this work, we produce a competitive convolution-free transformer by training on Imagenet only. We train them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop evaluation) on ImageNet with no external data. More importantly, we introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both Imagenet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and models._

Tips:

-   Compared to ViT, DeiT models use a so-called distillation token to effectively learn from a teacher (which, in the DeiT paper, is a ResNet like-model). The distillation token is learned through backpropagation, by interacting with the class (\[CLS\]) and patch tokens through the self-attention layers.
-   There are 2 ways to fine-tune distilled models, either (1) in a classic way, by only placing a prediction head on top of the final hidden state of the class token and not using the distillation signal, or (2) by placing both a prediction head on top of the class token and on top of the distillation token. In that case, the \[CLS\] prediction head is trained using regular cross-entropy between the prediction of the head and the ground-truth label, while the distillation prediction head is trained using hard distillation (cross-entropy between the prediction of the distillation head and the label predicted by the teacher). At inference time, one takes the average prediction between both heads as final prediction. (2) is also called “fine-tuning with distillation”, because one relies on a teacher that has already been fine-tuned on the downstream dataset. In terms of models, (1) corresponds to [DeiTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTForImageClassification) and (2) corresponds to [DeiTForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTForImageClassificationWithTeacher).
-   Note that the authors also did try soft distillation for (2) (in which case the distillation prediction head is trained using KL divergence to match the softmax output of the teacher), but hard distillation gave the best results.
-   All released checkpoints were pre-trained and fine-tuned on ImageNet-1k only. No external data was used. This is in contrast with the original ViT model, which used external data like the JFT-300M dataset/Imagenet-21k for pre-training.
-   The authors of DeiT also released more efficiently trained ViT models, which you can directly plug into [ViTModel](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTModel) or [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification). Techniques like data augmentation, optimization, and regularization were used in order to simulate training on a much larger dataset (while only using ImageNet-1k for pre-training). There are 4 variants available (in 3 different sizes): _facebook/deit-tiny-patch16-224_, _facebook/deit-small-patch16-224_, _facebook/deit-base-patch16-224_ and _facebook/deit-base-patch16-384_. Note that one should use [DeiTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTImageProcessor) in order to prepare images for the model.

This model was contributed by [nielsr](https://huggingface.co/nielsr). The TensorFlow version of this model was added by [amyeroberts](https://huggingface.co/amyeroberts).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DeiT.

-   [DeiTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

Besides that:

-   [DeiTForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTForMaskedImageModeling) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DeiTConfig

### class transformers.DeiTConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/configuration_deit.py#L37)

( hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-12image\_size = 224patch\_size = 16num\_channels = 3qkv\_bias = Trueencoder\_stride = 16\*\*kwargs )

This is the configuration class to store the configuration of a [DeiTModel](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTModel). It is used to instantiate an DeiT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the DeiT [facebook/deit-base-distilled-patch16-224](https://huggingface.co/facebook/deit-base-distilled-patch16-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import DeiTConfig, DeiTModel

>>> 
>>> configuration = DeiTConfig()

>>> 
>>> model = DeiTModel(configuration)

>>> 
>>> configuration = model.config
```

## DeiTFeatureExtractor

Preprocess an image or a batch of images.

## DeiTImageProcessor

### class transformers.DeiTImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/image_processing_deit.py#L45)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = 3do\_center\_crop: bool = Truecrop\_size: typing.Dict\[str, int\] = Nonerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_rescale: bool = Truedo\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None\*\*kwargs )

Constructs a DeiT image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/image_processing_deit.py#L161)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: bool = Nonesize: typing.Dict\[str, int\] = Noneresample = Nonedo\_center\_crop: bool = Nonecrop\_size: typing.Dict\[str, int\] = Nonedo\_rescale: bool = Nonerescale\_factor: float = Nonedo\_normalize: bool = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## DeiTModel

### class transformers.DeiTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_deit.py#L461)

( config: DeiTConfigadd\_pooling\_layer: bool = Trueuse\_mask\_token: bool = False )

Parameters

-   **config** ([DeiTConfig](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DeiT Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_deit.py#L486)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [DeiTModel](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, DeiTModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
>>> model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 198, 768]
```

## DeiTForMaskedImageModeling

### class transformers.DeiTForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_deit.py#L581)

( config: DeiTConfig )

Parameters

-   **config** ([DeiTConfig](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DeiT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

Note that we provide a script to pre-train this model on custom data in our [examples directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_deit.py#L599)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.MaskedImageModelingOutput` or `tuple(torch.FloatTensor)`

The [DeiTForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTForMaskedImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DeiTForMaskedImageModeling
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
>>> model = DeiTForMaskedImageModeling.from_pretrained("facebook/deit-base-distilled-patch16-224")

>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
>>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
>>> 
>>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
>>> list(reconstructed_pixel_values.shape)
[1, 3, 224, 224]
```

## DeiTForImageClassification

### class transformers.DeiTForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_deit.py#L693)

( config: DeiTConfig )

Parameters

-   **config** ([DeiTConfig](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DeiT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_deit.py#L706)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

The [DeiTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DeiTForImageClassification
>>> import torch
>>> from PIL import Image
>>> import requests

>>> torch.manual_seed(3)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> 
>>> 
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
>>> model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> 
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: magpie
```

## DeiTForImageClassificationWithTeacher

### class transformers.DeiTForImageClassificationWithTeacher

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_deit.py#L843)

( config: DeiTConfig )

Parameters

-   **config** ([DeiTConfig](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of the \[CLS\] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

.. warning::

This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet supported.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_deit.py#L861)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.deit.modeling_deit.DeiTForImageClassificationWithTeacherOutput` or `tuple(torch.FloatTensor)`

The [DeiTForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTForImageClassificationWithTeacher) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, DeiTForImageClassificationWithTeacher
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
>>> model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## TFDeiTModel

### class transformers.TFDeiTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_tf_deit.py#L610)

( \*args\*\*kwargs )

Parameters

-   **config** ([DeiTConfig](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DeiT Model transformer outputting raw hidden-states without any specific head on top. This model is a TensorFlow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.

The [TFDeiTModel](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.TFDeiTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFDeiTModel
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
>>> model = TFDeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

>>> inputs = image_processor(image, return_tensors="tf")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 198, 768]
```

## TFDeiTForMaskedImageModeling

### class transformers.TFDeiTForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_tf_deit.py#L718)

( \*args\*\*kwargs )

Parameters

-   **config** ([DeiTConfig](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DeiT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886). This model is a TensorFlow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_tf_deit.py#L725)

( pixel\_values: tf.Tensor | None = Nonebool\_masked\_pos: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.modeling_tf_outputs.TFMaskedImageModelingOutput` or `tuple(tf.Tensor)`

The [TFDeiTForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.TFDeiTForMaskedImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TFDeiTForMaskedImageModeling
>>> import tensorflow as tf
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
>>> model = TFDeiTForMaskedImageModeling.from_pretrained("facebook/deit-base-distilled-patch16-224")

>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
>>> pixel_values = image_processor(images=image, return_tensors="tf").pixel_values
>>> 
>>> bool_masked_pos = tf.cast(tf.random.uniform((1, num_patches), minval=0, maxval=2, dtype=tf.int32), tf.bool)

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
>>> list(reconstructed_pixel_values.shape)
[1, 3, 224, 224]
```

## TFDeiTForImageClassification

### class transformers.TFDeiTForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_tf_deit.py#L833)

( \*args\*\*kwargs )

Parameters

-   **config** ([DeiTConfig](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DeiT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a TensorFlow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_tf_deit.py#L847)

( pixel\_values: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Nonelabels: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.modeling_tf_outputs.TFImageClassifierOutput` or `tuple(tf.Tensor)`

The [TFDeiTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.TFDeiTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TFDeiTForImageClassification
>>> import tensorflow as tf
>>> from PIL import Image
>>> import requests

>>> tf.keras.utils.set_random_seed(3)
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> 
>>> 
>>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
>>> model = TFDeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

>>> inputs = image_processor(images=image, return_tensors="tf")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> 
>>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
>>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
Predicted class: little blue heron, Egretta caerulea
```

## TFDeiTForImageClassificationWithTeacher

### class transformers.TFDeiTForImageClassificationWithTeacher

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_tf_deit.py#L935)

( \*args\*\*kwargs )

Parameters

-   **config** ([DeiTConfig](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of the \[CLS\] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

.. warning::

This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet supported.

This model is a TensorFlow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/deit/modeling_tf_deit.py#L954)

( pixel\_values: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.models.deit.modeling_tf_deit.TFDeiTForImageClassificationWithTeacherOutput` or `tuple(tf.Tensor)`

The [TFDeiTForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.TFDeiTForImageClassificationWithTeacher) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFDeiTForImageClassificationWithTeacher
>>> import tensorflow as tf
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
>>> model = TFDeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")

>>> inputs = image_processor(image, return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> predicted_label = int(tf.math.argmax(logits, axis=-1))
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```