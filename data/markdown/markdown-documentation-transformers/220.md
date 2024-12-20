# MobileViT

## Overview

The MobileViT model was proposed in [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer](https://arxiv.org/abs/2110.02178) by Sachin Mehta and Mohammad Rastegari. MobileViT introduces a new layer that replaces local processing in convolutions with global processing using transformers.

The abstract from the paper is the following:

_Light-weight convolutional neural networks (CNNs) are the de-facto for mobile vision tasks. Their spatial inductive biases allow them to learn representations with fewer parameters across different vision tasks. However, these networks are spatially local. To learn global representations, self-attention-based vision trans-formers (ViTs) have been adopted. Unlike CNNs, ViTs are heavy-weight. In this paper, we ask the following question: is it possible to combine the strengths of CNNs and ViTs to build a light-weight and low latency network for mobile vision tasks? Towards this end, we introduce MobileViT, a light-weight and general-purpose vision transformer for mobile devices. MobileViT presents a different perspective for the global processing of information with transformers, i.e., transformers as convolutions. Our results show that MobileViT significantly outperforms CNN- and ViT-based networks across different tasks and datasets. On the ImageNet-1k dataset, MobileViT achieves top-1 accuracy of 78.4% with about 6 million parameters, which is 3.2% and 6.2% more accurate than MobileNetv3 (CNN-based) and DeIT (ViT-based) for a similar number of parameters. On the MS-COCO object detection task, MobileViT is 5.7% more accurate than MobileNetv3 for a similar number of parameters._

Tips:

-   MobileViT is more like a CNN than a Transformer model. It does not work on sequence data but on batches of images. Unlike ViT, there are no embeddings. The backbone model outputs a feature map. You can follow [this tutorial](https://keras.io/examples/vision/mobilevit) for a lightweight introduction.
    
-   One can use [MobileViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTImageProcessor) to prepare images for the model. Note that if you do your own preprocessing, the pretrained checkpoints expect images to be in BGR pixel order (not RGB).
    
-   The available image classification checkpoints are pre-trained on [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes).
    
-   The segmentation model uses a [DeepLabV3](https://arxiv.org/abs/1706.05587) head. The available semantic segmentation checkpoints are pre-trained on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).
    
-   As the name suggests MobileViT was designed to be performant and efficient on mobile phones. The TensorFlow versions of the MobileViT models are fully compatible with [TensorFlow Lite](https://www.tensorflow.org/lite).
    
    You can use the following code to convert a MobileViT checkpoint (be it image classification or semantic segmentation) to generate a TensorFlow Lite model:
    

```
from transformers import TFMobileViTForImageClassification
import tensorflow as tf


model_ckpt = "apple/mobilevit-xx-small"
model = TFMobileViTForImageClassification.from_pretrained(model_ckpt)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()
tflite_filename = model_ckpt.split("/")[-1] + ".tflite"
with open(tflite_filename, "wb") as f:
    f.write(tflite_model)
```

The resulting model will be just **about an MB** making it a good fit for mobile applications where resources and network bandwidth can be constrained.

This model was contributed by [matthijs](https://huggingface.co/Matthijs). The TensorFlow version of the model was contributed by [sayakpaul](https://huggingface.co/sayakpaul). The original code and weights can be found [here](https://github.com/apple/ml-cvnets).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with MobileViT.

-   [MobileViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

**Semantic segmentation**

-   [Semantic segmentation task guide](../tasks/semantic_segmentation)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## MobileViTConfig

### class transformers.MobileViTConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/configuration_mobilevit.py#L46)

( num\_channels = 3 image\_size = 256 patch\_size = 2 hidden\_sizes = \[144, 192, 240\] neck\_hidden\_sizes = \[16, 32, 64, 96, 128, 160, 640\] num\_attention\_heads = 4 mlp\_ratio = 2.0 expand\_ratio = 4.0 hidden\_act = 'silu' conv\_kernel\_size = 3 output\_stride = 32 hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.0 classifier\_dropout\_prob = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-05 qkv\_bias = True aspp\_out\_channels = 256 atrous\_rates = \[6, 12, 18\] aspp\_dropout\_prob = 0.1 semantic\_loss\_ignore\_index = 255 \*\*kwargs )

Parameters

-   **num\_channels** (`int`, _optional_, defaults to 3) — The number of input channels.
-   **image\_size** (`int`, _optional_, defaults to 256) — The size (resolution) of each image.
-   **patch\_size** (`int`, _optional_, defaults to 2) — The size (resolution) of each patch.
-   **hidden\_sizes** (`List[int]`, _optional_, defaults to `[144, 192, 240]`) — Dimensionality (hidden size) of the Transformer encoders at each stage.
-   **neck\_hidden\_sizes** (`List[int]`, _optional_, defaults to `[16, 32, 64, 96, 128, 160, 640]`) — The number of channels for the feature maps of the backbone.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 4) — Number of attention heads for each attention layer in the Transformer encoder.
-   **mlp\_ratio** (`float`, _optional_, defaults to 2.0) — The ratio of the number of channels in the output of the MLP to the number of channels in the input.
-   **expand\_ratio** (`float`, _optional_, defaults to 4.0) — Expansion factor for the MobileNetv2 layers.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"silu"`) — The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
-   **conv\_kernel\_size** (`int`, _optional_, defaults to 3) — The size of the convolutional kernel in the MobileViT layer.
-   **output\_stride** (`int`, `optional`, defaults to 32) — The ratio of the spatial resolution of the output to the resolution of the input image.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probabilitiy for all fully connected layers in the Transformer encoder.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **classifier\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for attached classifiers.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization layers.
-   **qkv\_bias** (`bool`, _optional_, defaults to `True`) — Whether to add a bias to the queries, keys and values.
-   **aspp\_out\_channels** (`int`, `optional`, defaults to 256) — Number of output channels used in the ASPP layer for semantic segmentation.
-   **atrous\_rates** (`List[int]`, _optional_, defaults to `[6, 12, 18]`) — Dilation (atrous) factors used in the ASPP layer for semantic segmentation.
-   **aspp\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the ASPP layer for semantic segmentation.
-   **semantic\_loss\_ignore\_index** (`int`, _optional_, defaults to 255) — The index that is ignored by the loss function of the semantic segmentation model.

This is the configuration class to store the configuration of a [MobileViTModel](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTModel). It is used to instantiate a MobileViT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the MobileViT [apple/mobilevit-small](https://huggingface.co/apple/mobilevit-small) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import MobileViTConfig, MobileViTModel

>>> 
>>> configuration = MobileViTConfig()

>>> 
>>> model = MobileViTModel(configuration)

>>> 
>>> configuration = model.config
```

## MobileViTFeatureExtractor

Preprocess an image or a batch of images.

( outputs target\_sizes: typing.List\[typing.Tuple\] = None ) → semantic\_segmentation

Parameters

-   **outputs** ([MobileViTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple]` of length `batch_size`, _optional_) — List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

`List[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [MobileViTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## MobileViTImageProcessor

### class transformers.MobileViTImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/image_processing_mobilevit.py#L51)

( do\_resize: bool = True size: typing.Dict\[str, int\] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_rescale: bool = True rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098 do\_center\_crop: bool = True crop\_size: typing.Dict\[str, int\] = None do\_flip\_channel\_order: bool = True \*\*kwargs )

Parameters

-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
-   **size** (`Dict[str, int]` _optional_, defaults to `{"shortest_edge" -- 224}`): Controls the size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess` method.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BILINEAR`) — Defines the resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the `preprocess` method.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `True`) — Whether to crop the input at the center. If the input size is smaller than `crop_size` along any edge, the image is padded with 0’s and then center cropped. Can be overridden by the `do_center_crop` parameter in the `preprocess` method.
-   **crop\_size** (`Dict[str, int]`, _optional_, defaults to `{"height" -- 256, "width": 256}`): Desired output size `(size["height"], size["width"])` when applying center-cropping. Can be overridden by the `crop_size` parameter in the `preprocess` method.
-   **do\_flip\_channel\_order** (`bool`, _optional_, defaults to `True`) — Whether to flip the color channels from RGB to BGR. Can be overridden by the `do_flip_channel_order` parameter in the `preprocess` method.

Constructs a MobileViT image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/image_processing_mobilevit.py#L172)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] do\_resize: bool = None size: typing.Dict\[str, int\] = None resample: Resampling = None do\_rescale: bool = None rescale\_factor: float = None do\_center\_crop: bool = None crop\_size: typing.Dict\[str, int\] = None do\_flip\_channel\_order: bool = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
-   **do\_resize** (`bool`, _optional_, defaults to `self.do_resize`) — Whether to resize the image.
-   **size** (`Dict[str, int]`, _optional_, defaults to `self.size`) — Size of the image after resizing.
-   **resample** (`int`, _optional_, defaults to `self.resample`) — Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only has an effect if `do_resize` is set to `True`.
-   **do\_rescale** (`bool`, _optional_, defaults to `self.do_rescale`) — Whether to rescale the image by rescale factor.
-   **rescale\_factor** (`float`, _optional_, defaults to `self.rescale_factor`) — Rescale factor to rescale the image by if `do_rescale` is set to `True`.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `self.do_center_crop`) — Whether to center crop the image.
-   **crop\_size** (`Dict[str, int]`, _optional_, defaults to `self.crop_size`) — Size of the center crop if `do_center_crop` is set to `True`.
-   **do\_flip\_channel\_order** (`bool`, _optional_, defaults to `self.do_flip_channel_order`) — Whether to flip the channel order of the image.
-   **return\_tensors** (`str` or `TensorType`, _optional_) — The type of tensors to return. Can be one of:
    
    -   Unset: Return a list of `np.ndarray`.
    -   `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
    -   `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
    -   `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
    -   `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
    
-   **data\_format** (`ChannelDimension` or `str`, _optional_, defaults to `ChannelDimension.FIRST`) — The channel dimension format for the output image. Can be one of:
    
    -   `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    
-   **input\_data\_format** (`ChannelDimension` or `str`, _optional_) — The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    

Preprocess an image or batch of images.

#### post\_process\_semantic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/image_processing_mobilevit.py#L303)

( outputs target\_sizes: typing.List\[typing.Tuple\] = None ) → semantic\_segmentation

Parameters

-   **outputs** ([MobileViTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple]` of length `batch_size`, _optional_) — List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

Returns

semantic\_segmentation

`List[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [MobileViTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## MobileViTModel

### class transformers.MobileViTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_mobilevit.py#L708)

( config: MobileViTConfig expand\_output: bool = True )

Parameters

-   **config** ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare MobileViT model outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_mobilevit.py#L745)

( pixel\_values: typing.Optional\[torch.Tensor\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [MobileViTImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.BaseModelOutputWithPoolingAndNoAttention` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state after a pooling operation on the spatial dimensions.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, num_channels, height, width)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    

The [MobileViTModel](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, MobileViTModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
>>> model = MobileViTModel.from_pretrained("apple/mobilevit-small")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 640, 8, 8]
```

## MobileViTForImageClassification

### class transformers.MobileViTForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_mobilevit.py#L802)

( config: MobileViTConfig )

Parameters

-   **config** ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_mobilevit.py#L818)

( pixel\_values: typing.Optional\[torch.Tensor\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [MobileViTImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.ImageClassifierOutputWithNoAttention](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutputWithNoAttention) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at the output of each stage.

The [MobileViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, MobileViTForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
>>> model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## MobileViTForSemanticSegmentation

### class transformers.MobileViTForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_mobilevit.py#L997)

( config: MobileViTConfig )

Parameters

-   **config** ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_mobilevit.py#L1008)

( pixel\_values: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [MobileViTImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, height, width)`, _optional_) — Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) — Classification scores for each pixel.
    
    The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the original image size as post-processing. You should always check your logits shape and resize as needed.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [MobileViTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import requests
>>> import torch
>>> from PIL import Image
>>> from transformers import AutoImageProcessor, MobileViTForSemanticSegmentation

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
>>> model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> 
>>> logits = outputs.logits
```

## TFMobileViTModel

### class transformers.TFMobileViTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_tf_mobilevit.py#L800)

( \*args \*\*kwargs )

Parameters

-   **config** ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare MobileViT model outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_tf_mobilevit.py#L808)

( pixel\_values: tf.Tensor | None = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: bool = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or `tuple(tf.Tensor)`

Parameters

-   **pixel\_values** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [MobileViTImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.

A [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`tf.Tensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
    This output is usually _not_ a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFMobileViTModel](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.TFMobileViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFMobileViTModel
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
>>> model = TFMobileViTModel.from_pretrained("apple/mobilevit-small")

>>> inputs = image_processor(image, return_tensors="tf")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 640, 8, 8]
```

## TFMobileViTForImageClassification

### class transformers.TFMobileViTForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_tf_mobilevit.py#L835)

( \*args \*\*kwargs )

Parameters

-   **config** ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_tf_mobilevit.py#L848)

( pixel\_values: tf.Tensor | None = None output\_hidden\_states: Optional\[bool\] = None labels: tf.Tensor | None = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → `transformers.modeling_tf_outputs.TFImageClassifierOutputWithNoAttention` or `tuple(tf.Tensor)`

Parameters

-   **pixel\_values** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [MobileViTImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **labels** (`tf.Tensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

`transformers.modeling_tf_outputs.TFImageClassifierOutputWithNoAttention` or `tuple(tf.Tensor)`

A `transformers.modeling_tf_outputs.TFImageClassifierOutputWithNoAttention` or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the model at the output of each stage.

The [TFMobileViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.TFMobileViTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFMobileViTForImageClassification
>>> import tensorflow as tf
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
>>> model = TFMobileViTForImageClassification.from_pretrained("apple/mobilevit-small")

>>> inputs = image_processor(image, return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> predicted_label = int(tf.math.argmax(logits, axis=-1))
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## TFMobileViTForSemanticSegmentation

### class transformers.TFMobileViTForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_tf_mobilevit.py#L1011)

( \*args \*\*kwargs )

Parameters

-   **config** ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mobilevit/modeling_tf_mobilevit.py#L1039)

( pixel\_values: tf.Tensor | None = None labels: tf.Tensor | None = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: bool = False ) → `transformers.modeling_tf_outputs.TFSemanticSegmenterOutputWithNoAttention` or `tuple(tf.Tensor)`

Parameters

-   **pixel\_values** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]`, `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [MobileViTImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **labels** (`tf.Tensor` of shape `(batch_size, height, width)`, _optional_) — Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

Returns

`transformers.modeling_tf_outputs.TFSemanticSegmenterOutputWithNoAttention` or `tuple(tf.Tensor)`

A `transformers.modeling_tf_outputs.TFSemanticSegmenterOutputWithNoAttention` or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MobileViTConfig](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.MobileViTConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels, logits_height, logits_width)`) — Classification scores for each pixel.
    
    The logits returned do not necessarily have the same size as the `pixel_values` passed as inputs. This is to avoid doing two interpolations and lose some quality when a user needs to resize the logits to the original image size as post-processing. You should always check your logits shape and resize as needed.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, patch_size, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    

The [TFMobileViTForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/mobilevit#transformers.TFMobileViTForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TFMobileViTForSemanticSegmentation
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("apple/deeplabv3-mobilevit-small")
>>> model = TFMobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-small")

>>> inputs = image_processor(images=image, return_tensors="tf")

>>> outputs = model(**inputs)

>>> 
>>> logits = outputs.logits
```