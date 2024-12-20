# Swin Transformer

## Overview

The Swin Transformer was proposed in [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo.

The abstract from the paper is the following:

_This paper presents a new vision Transformer, called Swin Transformer, that capably serves as a general-purpose backbone for computer vision. Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text. To address these differences, we propose a hierarchical Transformer whose representation is computed with \\bold{S}hifted \\bold{win}dows. The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection. This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size. These qualities of Swin Transformer make it compatible with a broad range of vision tasks, including image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as object detection (58.7 box AP and 51.1 mask AP on COCO test-dev) and semantic segmentation (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the-art by a large margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the potential of Transformer-based models as vision backbones. The hierarchical design and the shifted window approach also prove beneficial for all-MLP architectures._

Tips:

-   One can use the [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) API to prepare images for the model.
-   Swin pads the inputs supporting any input height and width (if divisible by `32`).
-   Swin can be used as a _backbone_. When `output_hidden_states = True`, it will output both `hidden_states` and `reshaped_hidden_states`. The `reshaped_hidden_states` have a shape of `(batch, num_channels, height, width)` rather than `(batch_size, sequence_length, num_channels)`.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/swin_transformer_architecture.png) Swin Transformer architecture. Taken from the [original paper](https://arxiv.org/abs/2102.03334).

This model was contributed by [novice03](https://huggingface.co/novice03). The Tensorflow version of this model was contributed by [amyeroberts](https://huggingface.co/amyeroberts). The original code can be found [here](https://github.com/microsoft/Swin-Transformer).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Swin Transformer.

-   [SwinForImageClassification](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

Besides that:

-   [SwinForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinForMaskedImageModeling) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## SwinConfig

### class transformers.SwinConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/configuration_swin.py#L38)

( image\_size = 224patch\_size = 4num\_channels = 3embed\_dim = 96depths = \[2, 2, 6, 2\]num\_heads = \[3, 6, 12, 24\]window\_size = 7mlp\_ratio = 4.0qkv\_bias = Truehidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0drop\_path\_rate = 0.1hidden\_act = 'gelu'use\_absolute\_embeddings = Falseinitializer\_range = 0.02layer\_norm\_eps = 1e-05encoder\_stride = 32out\_features = Noneout\_indices = None\*\*kwargs )

This is the configuration class to store the configuration of a [SwinModel](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinModel). It is used to instantiate a Swin model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Swin [microsoft/swin-tiny-patch4-window7-224](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import SwinConfig, SwinModel

>>> 
>>> configuration = SwinConfig()

>>> 
>>> model = SwinModel(configuration)

>>> 
>>> configuration = model.config
```

## SwinModel

### class transformers.SwinModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_swin.py#L946)

( configadd\_pooling\_layer = Trueuse\_mask\_token = False )

Parameters

-   **config** ([SwinConfig](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Swin Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_swin.py#L973)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.swin.modeling_swin.SwinModelOutput` or `tuple(torch.FloatTensor)`

The [SwinModel](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, SwinModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
>>> model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 49, 768]
```

## SwinForMaskedImageModeling

### class transformers.SwinForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_swin.py#L1055)

( config )

Parameters

-   **config** ([SwinConfig](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swin Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

Note that we provide a script to pre-train this model on custom data in our [examples directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_swin.py#L1072)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.swin.modeling_swin.SwinMaskedImageModelingOutput` or `tuple(torch.FloatTensor)`

The [SwinForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinForMaskedImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, SwinForMaskedImageModeling
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-simmim-window6-192")
>>> model = SwinForMaskedImageModeling.from_pretrained("microsoft/swin-base-simmim-window6-192")

>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
>>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
>>> 
>>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
>>> list(reconstructed_pixel_values.shape)
[1, 3, 192, 192]
```

## SwinForImageClassification

### class transformers.SwinForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_swin.py#L1166)

( config )

Parameters

-   **config** ([SwinConfig](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_swin.py#L1181)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.swin.modeling_swin.SwinImageClassifierOutput` or `tuple(torch.FloatTensor)`

The [SwinForImageClassification](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, SwinForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
>>> model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## TFSwinModel

### class transformers.TFSwinModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_tf_swin.py#L1173)

( \*args\*\*kwargs )

Parameters

-   **config** ([SwinConfig](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Swin Model transformer outputting raw hidden-states without any specific head on top. This model is a Tensorflow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_tf_swin.py#L1181)

( pixel\_values: tf.Tensor | None = Nonebool\_masked\_pos: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.models.swin.modeling_tf_swin.TFSwinModelOutput` or `tuple(tf.Tensor)`

The [TFSwinModel](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.TFSwinModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFSwinModel
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
>>> model = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

>>> inputs = image_processor(image, return_tensors="tf")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 49, 768]
```

## TFSwinForMaskedImageModeling

### class transformers.TFSwinForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_tf_swin.py#L1276)

( \*args\*\*kwargs )

Parameters

-   **config** ([SwinConfig](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swin Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886). This model is a Tensorflow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_tf_swin.py#L1284)

( pixel\_values: tf.Tensor | None = Nonebool\_masked\_pos: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.models.swin.modeling_tf_swin.TFSwinMaskedImageModelingOutput` or `tuple(tf.Tensor)`

The [TFSwinForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.TFSwinForMaskedImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TFSwinForMaskedImageModeling
>>> import tensorflow as tf
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
>>> model = TFSwinForMaskedImageModeling.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
>>> pixel_values = image_processor(images=image, return_tensors="tf").pixel_values
>>> 
>>> bool_masked_pos = tf.random.uniform((1, num_patches)) >= 0.5

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
>>> list(reconstructed_pixel_values.shape)
[1, 3, 224, 224]
```

## TFSwinForImageClassification

### class transformers.TFSwinForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_tf_swin.py#L1388)

( \*args\*\*kwargs )

Parameters

-   **config** ([SwinConfig](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.SwinConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Swin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a Tensorflow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) sub-class. Use it as a regular Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/swin/modeling_tf_swin.py#L1402)

( pixel\_values: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Nonelabels: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.models.swin.modeling_tf_swin.TFSwinImageClassifierOutput` or `tuple(tf.Tensor)`

The [TFSwinForImageClassification](/docs/transformers/v4.34.0/en/model_doc/swin#transformers.TFSwinForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFSwinForImageClassification
>>> import tensorflow as tf
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
>>> model = TFSwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

>>> inputs = image_processor(image, return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> predicted_label = int(tf.math.argmax(logits, axis=-1))
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```