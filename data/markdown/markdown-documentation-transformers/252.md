# RegNet

## Overview

The RegNet model was proposed in [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678) by Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár.

The authors design search spaces to perform Neural Architecture Search (NAS). They first start from a high dimensional search space and iteratively reduce the search space by empirically applying constraints based on the best-performing models sampled by the current search space.

The abstract from the paper is the following:

_In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The overall process is analogous to classic manual design of networks, but elevated to the design space level. Using our methodology we explore the structure aspect of network design and arrive at a low-dimensional design space consisting of simple, regular networks that we call RegNet. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function. We analyze the RegNet design space and arrive at interesting findings that do not match the current practice of network design. The RegNet design space provides simple and fast networks that work well across a wide range of flop regimes. Under comparable training settings and flops, the RegNet models outperform the popular EfficientNet models while being up to 5x faster on GPUs._

Tips:

-   One can use [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) to prepare images for the model.
-   The huge 10B model from [Self-supervised Pretraining of Visual Features in the Wild](https://arxiv.org/abs/2103.01988), trained on one billion Instagram images, is available on the [hub](https://huggingface.co/facebook/regnet-y-10b-seer)

This model was contributed by [Francesco](https://huggingface.co/Francesco). The TensorFlow version of the model was contributed by [sayakpaul](https://huggingface.com/sayakpaul) and [ariG23498](https://huggingface.com/ariG23498). The original code can be found [here](https://github.com/facebookresearch/pycls).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with RegNet.

-   [RegNetForImageClassification](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## RegNetConfig

### class transformers.RegNetConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/configuration_regnet.py#L28)

( num\_channels = 3embedding\_size = 32hidden\_sizes = \[128, 192, 512, 1088\]depths = \[2, 6, 12, 2\]groups\_width = 64layer\_type = 'y'hidden\_act = 'relu'\*\*kwargs )

Parameters

-   **num\_channels** (`int`, _optional_, defaults to 3) — The number of input channels.
-   **embedding\_size** (`int`, _optional_, defaults to 64) — Dimensionality (hidden size) for the embedding layer.
-   **hidden\_sizes** (`List[int]`, _optional_, defaults to `[256, 512, 1024, 2048]`) — Dimensionality (hidden size) at each stage.
-   **depths** (`List[int]`, _optional_, defaults to `[3, 4, 6, 3]`) — Depth (number of layers) for each stage.
-   **layer\_type** (`str`, _optional_, defaults to `"y"`) — The layer to use, it can be either `"x" or` “y”`. An` x`layer is a ResNet's BottleNeck layer with`reduction`fixed to`1`. While a` y`layer is a`x\` but with squeeze and excitation. Please refer to the paper for a detailed explanation of how these layers were constructed.
-   **hidden\_act** (`str`, _optional_, defaults to `"relu"`) — The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
-   **downsample\_in\_first\_stage** (`bool`, _optional_, defaults to `False`) — If `True`, the first stage will downsample the inputs using a `stride` of 2.

This is the configuration class to store the configuration of a [RegNetModel](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetModel). It is used to instantiate a RegNet model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the RegNet [facebook/regnet-y-040](https://huggingface.co/facebook/regnet-y-040) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import RegNetConfig, RegNetModel

>>> 
>>> configuration = RegNetConfig()
>>> 
>>> model = RegNetModel(configuration)
>>> 
>>> configuration = model.config
```

## RegNetModel

### class transformers.RegNetModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_regnet.py#L331)

( config )

Parameters

-   **config** ([RegNetConfig](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare RegNet model outputting raw features without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_regnet.py#L341)

( pixel\_values: Tensoroutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

The [RegNetModel](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, RegNetModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
>>> model = RegNetModel.from_pretrained("facebook/regnet-y-040")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 1088, 7, 7]
```

## RegNetForImageClassification

### class transformers.RegNetForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_regnet.py#L385)

( config )

Parameters

-   **config** ([RegNetConfig](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [RegNetForImageClassification](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, RegNetForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
>>> model = RegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## TFRegNetModel

### class transformers.TFRegNetModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_tf_regnet.py#L380)

( \*args\*\*kwargs )

Parameters

-   **This** model is a Tensorflow — \[tf.keras.layers.Layer\](https —//[www.tensorflow.org/api\_docs/python/tf/keras/layers/Layer](http://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)) sub-class. Use it as a
-   **regular** Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and — behavior. — config ([RegNetConfig](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetConfig)): Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare RegNet model outputting raw features without any specific head on top.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_tf_regnet.py#L385)

( pixel\_values: Tensoroutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetraining: bool = False ) → `transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndNoAttention` or `tuple(tf.Tensor)`

The [TFRegNetModel](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.TFRegNetModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFRegNetModel
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
>>> model = TFRegNetModel.from_pretrained("facebook/regnet-y-040")

>>> inputs = image_processor(image, return_tensors="tf")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 1088, 7, 7]
```

## TFRegNetForImageClassification

### class transformers.TFRegNetForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_tf_regnet.py#L429)

( \*args\*\*kwargs )

Parameters

-   **This** model is a Tensorflow — \[tf.keras.layers.Layer\](https —//[www.tensorflow.org/api\_docs/python/tf/keras/layers/Layer](http://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)) sub-class. Use it as a
-   **regular** Tensorflow Module and refer to the Tensorflow documentation for all matter related to general usage and — behavior. — config ([RegNetConfig](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetConfig)): Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_tf_regnet.py#L440)

( pixel\_values: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Nonelabels: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or `tuple(tf.Tensor)`

The [TFRegNetForImageClassification](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.TFRegNetForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFRegNetForImageClassification
>>> import tensorflow as tf
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
>>> model = TFRegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

>>> inputs = image_processor(image, return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> predicted_label = int(tf.math.argmax(logits, axis=-1))
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## FlaxRegNetModel

### class transformers.FlaxRegNetModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_flax_regnet.py#L689)

( config: RegNetConfiginput\_shape = (1, 224, 224, 3)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([RegNetConfig](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare RegNet model outputting raw features without any specific head on top.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

The `FlaxRegNetPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, FlaxRegNetModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
>>> model = FlaxRegNetModel.from_pretrained("facebook/regnet-y-040")

>>> inputs = image_processor(images=image, return_tensors="np")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## FlaxRegNetForImageClassification

### class transformers.FlaxRegNetForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_flax_regnet.py#L782)

( config: RegNetConfiginput\_shape = (1, 224, 224, 3)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([RegNetConfig](/docs/transformers/v4.34.0/en/model_doc/regnet#transformers.RegNetConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/regnet/modeling_flax_regnet.py#L596)

( pixel\_valuesparams: dict = Nonetrain: bool = Falseoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_flax_outputs.FlaxImageClassifierOutputWithNoAttention` or `tuple(torch.FloatTensor)`

The `FlaxRegNetPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, FlaxRegNetForImageClassification
>>> from PIL import Image
>>> import jax
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/regnet-y-040")
>>> model = FlaxRegNetForImageClassification.from_pretrained("facebook/regnet-y-040")

>>> inputs = image_processor(images=image, return_tensors="np")
>>> outputs = model(**inputs)
>>> logits = outputs.logits

>>> 
>>> predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
>>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
```