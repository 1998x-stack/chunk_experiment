# Convolutional Vision Transformer (CvT)

## Overview

The CvT model was proposed in [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808) by Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan and Lei Zhang. The Convolutional vision Transformer (CvT) improves the [Vision Transformer (ViT)](vit) in performance and efficiency by introducing convolutions into ViT to yield the best of both designs.

The abstract from the paper is the following:

_We present in this paper a new architecture, named Convolutional vision Transformer (CvT), that improves Vision Transformer (ViT) in performance and efficiency by introducing convolutions into ViT to yield the best of both designs. This is accomplished through two primary modifications: a hierarchy of Transformers containing a new convolutional token embedding, and a convolutional Transformer block leveraging a convolutional projection. These changes introduce desirable properties of convolutional neural networks (CNNs) to the ViT architecture (\\ie shift, scale, and distortion invariance) while maintaining the merits of Transformers (\\ie dynamic attention, global context, and better generalization). We validate CvT by conducting extensive experiments, showing that this approach achieves state-of-the-art performance over other Vision Transformers and ResNets on ImageNet-1k, with fewer parameters and lower FLOPs. In addition, performance gains are maintained when pretrained on larger datasets (\\eg ImageNet-22k) and fine-tuned to downstream tasks. Pre-trained on ImageNet-22k, our CvT-W24 obtains a top-1 accuracy of 87.7\\% on the ImageNet-1k val set. Finally, our results show that the positional encoding, a crucial component in existing Vision Transformers, can be safely removed in our model, simplifying the design for higher resolution vision tasks._

Tips:

-   CvT models are regular Vision Transformers, but trained with convolutions. They outperform the [original model (ViT)](vit) when fine-tuned on ImageNet-1K and CIFAR-100.
-   You can check out demo notebooks regarding inference as well as fine-tuning on custom data [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) (you can just replace [ViTFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTFeatureExtractor) by [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) and [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification) by [CvtForImageClassification](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtForImageClassification)).
-   The available checkpoints are either (1) pre-trained on [ImageNet-22k](http://www.image-net.org/) (a collection of 14 million images and 22k classes) only, (2) also fine-tuned on ImageNet-22k or (3) also fine-tuned on [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes).

This model was contributed by [anugunj](https://huggingface.co/anugunj). The original code can be found [here](https://github.com/microsoft/CvT).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CvT.

-   [CvtForImageClassification](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## CvtConfig

### class transformers.CvtConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cvt/configuration_cvt.py#L29)

( num\_channels = 3 patch\_sizes = \[7, 3, 3\] patch\_stride = \[4, 2, 2\] patch\_padding = \[2, 1, 1\] embed\_dim = \[64, 192, 384\] num\_heads = \[1, 3, 6\] depth = \[1, 2, 10\] mlp\_ratio = \[4.0, 4.0, 4.0\] attention\_drop\_rate = \[0.0, 0.0, 0.0\] drop\_rate = \[0.0, 0.0, 0.0\] drop\_path\_rate = \[0.0, 0.0, 0.1\] qkv\_bias = \[True, True, True\] cls\_token = \[False, False, True\] qkv\_projection\_method = \['dw\_bn', 'dw\_bn', 'dw\_bn'\] kernel\_qkv = \[3, 3, 3\] padding\_kv = \[1, 1, 1\] stride\_kv = \[2, 2, 2\] padding\_q = \[1, 1, 1\] stride\_q = \[1, 1, 1\] initializer\_range = 0.02 layer\_norm\_eps = 1e-12 \*\*kwargs )

Parameters

-   **num\_channels** (`int`, _optional_, defaults to 3) — The number of input channels.
-   **patch\_sizes** (`List[int]`, _optional_, defaults to `[7, 3, 3]`) — The kernel size of each encoder’s patch embedding.
-   **patch\_stride** (`List[int]`, _optional_, defaults to `[4, 2, 2]`) — The stride size of each encoder’s patch embedding.
-   **patch\_padding** (`List[int]`, _optional_, defaults to `[2, 1, 1]`) — The padding size of each encoder’s patch embedding.
-   **embed\_dim** (`List[int]`, _optional_, defaults to `[64, 192, 384]`) — Dimension of each of the encoder blocks.
-   **num\_heads** (`List[int]`, _optional_, defaults to `[1, 3, 6]`) — Number of attention heads for each attention layer in each block of the Transformer encoder.
-   **depth** (`List[int]`, _optional_, defaults to `[1, 2, 10]`) — The number of layers in each encoder block.
-   **mlp\_ratios** (`List[float]`, _optional_, defaults to `[4.0, 4.0, 4.0, 4.0]`) — Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the encoder blocks.
-   **attention\_drop\_rate** (`List[float]`, _optional_, defaults to `[0.0, 0.0, 0.0]`) — The dropout ratio for the attention probabilities.
-   **drop\_rate** (`List[float]`, _optional_, defaults to `[0.0, 0.0, 0.0]`) — The dropout ratio for the patch embeddings probabilities.
-   **drop\_path\_rate** (`List[float]`, _optional_, defaults to `[0.0, 0.0, 0.1]`) — The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
-   **qkv\_bias** (`List[bool]`, _optional_, defaults to `[True, True, True]`) — The bias bool for query, key and value in attentions
-   **cls\_token** (`List[bool]`, _optional_, defaults to `[False, False, True]`) — Whether or not to add a classification token to the output of each of the last 3 stages.
-   **qkv\_projection\_method** (`List[string]`, _optional_, defaults to \[“dw\_bn”, “dw\_bn”, “dw\_bn”\]\`) — The projection method for query, key and value Default is depth-wise convolutions with batch norm. For Linear projection use “avg”.
-   **kernel\_qkv** (`List[int]`, _optional_, defaults to `[3, 3, 3]`) — The kernel size for query, key and value in attention layer
-   **padding\_kv** (`List[int]`, _optional_, defaults to `[1, 1, 1]`) — The padding size for key and value in attention layer
-   **stride\_kv** (`List[int]`, _optional_, defaults to `[2, 2, 2]`) — The stride size for key and value in attention layer
-   **padding\_q** (`List[int]`, _optional_, defaults to `[1, 1, 1]`) — The padding size for query in attention layer
-   **stride\_q** (`List[int]`, _optional_, defaults to `[1, 1, 1]`) — The stride size for query in attention layer
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-6) — The epsilon used by the layer normalization layers.

This is the configuration class to store the configuration of a [CvtModel](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtModel). It is used to instantiate a CvT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the CvT [microsoft/cvt-13](https://huggingface.co/microsoft/cvt-13) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CvtConfig, CvtModel

>>> 
>>> configuration = CvtConfig()

>>> 
>>> model = CvtModel(configuration)

>>> 
>>> configuration = model.config
```

## CvtModel

### class transformers.CvtModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cvt/modeling_cvt.py#L590)

( config add\_pooling\_layer = True )

Parameters

-   **config** ([CvtConfig](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Cvt Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cvt/modeling_cvt.py#L605)

( pixel\_values: typing.Optional\[torch.Tensor\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See `CvtImageProcessor.__call__` for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or `tuple(torch.FloatTensor)`

A `transformers.models.cvt.modeling_cvt.BaseModelOutputWithCLSToken` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CvtConfig](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **cls\_token\_value** (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`) — Classification token at the output of the last layer of the model.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.

The [CvtModel](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, CvtModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
>>> model = CvtModel.from_pretrained("microsoft/cvt-13")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 384, 14, 14]
```

## CvtForImageClassification

### class transformers.CvtForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cvt/modeling_cvt.py#L651)

( config )

Parameters

-   **config** ([CvtConfig](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cvt/modeling_cvt.py#L666)

( pixel\_values: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See `CvtImageProcessor.__call__` for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CvtConfig](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at the output of each stage.

The [CvtForImageClassification](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, CvtForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
>>> model = CvtForImageClassification.from_pretrained("microsoft/cvt-13")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## TFCvtModel

### class transformers.TFCvtModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cvt/modeling_tf_cvt.py#L762)

( \*args \*\*kwargs )

Parameters

-   **config** ([CvtConfig](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare Cvt Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TF 2.0 models accepts two formats as inputs:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional arguments.

This second option is useful when using `tf.keras.Model.fit` method which currently requires having all the tensors in the first argument of the model call function: `model(inputs)`.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cvt/modeling_tf_cvt.py#L768)

( pixel\_values: tf.Tensor | None = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → `transformers.models.cvt.modeling_tf_cvt.TFBaseModelOutputWithCLSToken` or `tuple(tf.Tensor)`

Parameters

-   **pixel\_values** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` \``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See `CvtImageProcessor.__call__` for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to \`False“) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

Returns

`transformers.models.cvt.modeling_tf_cvt.TFBaseModelOutputWithCLSToken` or `tuple(tf.Tensor)`

A `transformers.models.cvt.modeling_tf_cvt.TFBaseModelOutputWithCLSToken` or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CvtConfig](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **cls\_token\_value** (`tf.Tensor` of shape `(batch_size, 1, hidden_size)`) — Classification token at the output of the last layer of the model.
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.

The [TFCvtModel](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.TFCvtModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TFCvtModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
>>> model = TFCvtModel.from_pretrained("microsoft/cvt-13")

>>> inputs = image_processor(images=image, return_tensors="tf")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## TFCvtForImageClassification

### class transformers.TFCvtForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cvt/modeling_tf_cvt.py#L826)

( \*args \*\*kwargs )

Parameters

-   **config** ([CvtConfig](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

Cvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TF 2.0 models accepts two formats as inputs:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional arguments.

This second option is useful when using `tf.keras.Model.fit` method which currently requires having all the tensors in the first argument of the model call function: `model(inputs)`.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cvt/modeling_tf_cvt.py#L844)

( pixel\_values: tf.Tensor | None = None labels: tf.Tensor | None = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → `transformers.modeling_tf_outputs.TFImageClassifierOutputWithNoAttention` or `tuple(tf.Tensor)`

Parameters

-   **pixel\_values** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` \``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See `CvtImageProcessor.__call__` for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to \`False“) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

`transformers.modeling_tf_outputs.TFImageClassifierOutputWithNoAttention` or `tuple(tf.Tensor)`

A `transformers.modeling_tf_outputs.TFImageClassifierOutputWithNoAttention` or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CvtConfig](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.CvtConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at the output of each stage.

The [TFCvtForImageClassification](/docs/transformers/v4.34.0/en/model_doc/cvt#transformers.TFCvtForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TFCvtForImageClassification
>>> import tensorflow as tf
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/cvt-13")
>>> model = TFCvtForImageClassification.from_pretrained("microsoft/cvt-13")

>>> inputs = image_processor(images=image, return_tensors="tf")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> 
>>> predicted_class_idx = tf.math.argmax(logits, axis=-1)[0]
>>> print("Predicted class:", model.config.id2label[int(predicted_class_idx)])
```