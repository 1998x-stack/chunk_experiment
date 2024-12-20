# EfficientFormer

## Overview

The EfficientFormer model was proposed in [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191) by Yanyu Li, Geng Yuan, Yang Wen, Eric Hu, Georgios Evangelidis, Sergey Tulyakov, Yanzhi Wang, Jian Ren. EfficientFormer proposes a dimension-consistent pure transformer that can be run on mobile devices for dense prediction tasks like image classification, object detection and semantic segmentation.

The abstract from the paper is the following:

_Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks. However, due to the massive number of parameters and model design, e.g., attention mechanism, ViT-based models are generally times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance? To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs. Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as a design paradigm. Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer. Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices. Our fastest model, EfficientFormer-L1, achieves 79.2% top-1 accuracy on ImageNet-1K with only 1.6 ms inference latency on iPhone 12 (compiled with CoreML), which { runs as fast as MobileNetV2×1.4 (1.6 ms, 74.7% top-1),} and our largest model, EfficientFormer-L7, obtains 83.3% accuracy with only 7.0 ms latency. Our work proves that properly designed transformers can reach extremely low latency on mobile devices while maintaining high performance._

This model was contributed by [novice03](https://huggingface.co/novice03) and [Bearnardd](https://huggingface.co/Bearnardd). The original code can be found [here](https://github.com/snap-research/EfficientFormer). The TensorFlow version of this model was added by [D-Roberts](https://huggingface.co/D-Roberts).

## Documentation resources

-   [Image classification task guide](../tasks/image_classification)

## EfficientFormerConfig

### class transformers.EfficientFormerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/configuration_efficientformer.py#L32)

( depths: typing.List\[int\] = \[3, 2, 6, 4\] hidden\_sizes: typing.List\[int\] = \[48, 96, 224, 448\] downsamples: typing.List\[bool\] = \[True, True, True, True\] dim: int = 448 key\_dim: int = 32 attention\_ratio: int = 4 resolution: int = 7 num\_hidden\_layers: int = 5 num\_attention\_heads: int = 8 mlp\_expansion\_ratio: int = 4 hidden\_dropout\_prob: float = 0.0 patch\_size: int = 16 num\_channels: int = 3 pool\_size: int = 3 downsample\_patch\_size: int = 3 downsample\_stride: int = 2 downsample\_pad: int = 1 drop\_path\_rate: float = 0.0 num\_meta3d\_blocks: int = 1 distillation: bool = True use\_layer\_scale: bool = True layer\_scale\_init\_value: float = 1e-05 hidden\_act: str = 'gelu' initializer\_range: float = 0.02 layer\_norm\_eps: float = 1e-12 image\_size: int = 224 batch\_norm\_eps: float = 1e-05 \*\*kwargs )

Parameters

-   **depths** (`List(int)`, _optional_, defaults to `[3, 2, 6, 4]`) — Depth of each stage.
-   **hidden\_sizes** (`List(int)`, _optional_, defaults to `[48, 96, 224, 448]`) — Dimensionality of each stage.
-   **downsamples** (`List(bool)`, _optional_, defaults to `[True, True, True, True]`) — Whether or not to downsample inputs between two stages.
-   **dim** (`int`, _optional_, defaults to 448) — Number of channels in Meta3D layers
-   **key\_dim** (`int`, _optional_, defaults to 32) — The size of the key in meta3D block.
-   **attention\_ratio** (`int`, _optional_, defaults to 4) — Ratio of the dimension of the query and value to the dimension of the key in MSHA block
-   **resolution** (`int`, _optional_, defaults to 7) — Size of each patch
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 5) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 8) — Number of attention heads for each attention layer in the 3D MetaBlock.
-   **mlp\_expansion\_ratio** (`int`, _optional_, defaults to 4) — Ratio of size of the hidden dimensionality of an MLP to the dimensionality of its input.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings and encoder.
-   **patch\_size** (`int`, _optional_, defaults to 16) — The size (resolution) of each patch.
-   **num\_channels** (`int`, _optional_, defaults to 3) — The number of input channels.
-   **pool\_size** (`int`, _optional_, defaults to 3) — Kernel size of pooling layers.
-   **downsample\_patch\_size** (`int`, _optional_, defaults to 3) — The size of patches in downsampling layers.
-   **downsample\_stride** (`int`, _optional_, defaults to 2) — The stride of convolution kernels in downsampling layers.
-   **downsample\_pad** (`int`, _optional_, defaults to 1) — Padding in downsampling layers.
-   **drop\_path\_rate** (`int`, _optional_, defaults to 0) — Rate at which to increase dropout probability in DropPath.
-   **num\_meta3d\_blocks** (`int`, _optional_, defaults to 1) — The number of 3D MetaBlocks in the last stage.
-   **distillation** (`bool`, _optional_, defaults to `True`) — Whether to add a distillation head.
-   **use\_layer\_scale** (`bool`, _optional_, defaults to `True`) — Whether to scale outputs from token mixers.
-   **layer\_scale\_init\_value** (`float`, _optional_, defaults to 1e-5) — Factor by which outputs from token mixers are scaled.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **image\_size** (`int`, _optional_, defaults to `224`) — The size (resolution) of each image.

This is the configuration class to store the configuration of an [EfficientFormerModel](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerModel). It is used to instantiate an EfficientFormer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the EfficientFormer [snap-research/efficientformer-l1](https://huggingface.co/snap-research/efficientformer-l1) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import EfficientFormerConfig, EfficientFormerModel

>>> 
>>> configuration = EfficientFormerConfig()

>>> 
>>> model = EfficientFormerModel(configuration)

>>> 
>>> configuration = model.config
```

## EfficientFormerImageProcessor

### class transformers.EfficientFormerImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/image_processing_efficientformer.py#L45)

( do\_resize: bool = True size: typing.Union\[typing.Dict\[str, int\], NoneType\] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_center\_crop: bool = True do\_rescale: bool = True rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098 crop\_size: typing.Dict\[str, int\] = None do\_normalize: bool = True image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None \*\*kwargs )

Parameters

-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s (height, width) dimensions to the specified `(size["height"], size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
-   **size** (`dict`, _optional_, defaults to `{"height" -- 224, "width": 224}`): Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess` method.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BILINEAR`) — Resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the `preprocess` method.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `True`) — Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the `preprocess` method.
-   **crop\_size** (`Dict[str, int]` _optional_, defaults to 224) — Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess` method.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method. do\_normalize — Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_MEAN`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_STD`) — Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a EfficientFormer image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/image_processing_efficientformer.py#L160)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] do\_resize: typing.Optional\[bool\] = None size: typing.Dict\[str, int\] = None resample: Resampling = None do\_center\_crop: bool = None crop\_size: int = None do\_rescale: typing.Optional\[bool\] = None rescale\_factor: typing.Optional\[float\] = None do\_normalize: typing.Optional\[bool\] = None image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
-   **do\_resize** (`bool`, _optional_, defaults to `self.do_resize`) — Whether to resize the image.
-   **size** (`Dict[str, int]`, _optional_, defaults to `self.size`) — Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after resizing.
-   **resample** (`PILImageResampling` filter, _optional_, defaults to `self.resample`) — `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has an effect if `do_resize` is set to `True`.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `self.do_center_crop`) — Whether to center crop the image.
-   **do\_rescale** (`bool`, _optional_, defaults to `self.do_rescale`) — Whether to rescale the image values between \[0 - 1\].
-   **rescale\_factor** (`float`, _optional_, defaults to `self.rescale_factor`) — Rescale factor to rescale the image by if `do_rescale` is set to `True`.
-   **crop\_size** (`Dict[str, int]`, _optional_, defaults to `self.crop_size`) — Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
-   **do\_normalize** (`bool`, _optional_, defaults to `self.do_normalize`) — Whether to normalize the image.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `self.image_mean`) — Image mean to use if `do_normalize` is set to `True`.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `self.image_std`) — Image standard deviation to use if `do_normalize` is set to `True`.
-   **return\_tensors** (`str` or `TensorType`, _optional_) — The type of tensors to return. Can be one of:
    
    -   Unset: Return a list of `np.ndarray`.
    -   `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
    -   `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
    -   `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
    -   `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
    
-   **data\_format** (`ChannelDimension` or `str`, _optional_, defaults to `ChannelDimension.FIRST`) — The channel dimension format for the output image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   Unset: Use the channel dimension format of the input image.
    
-   **input\_data\_format** (`ChannelDimension` or `str`, _optional_) — The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    

Preprocess an image or batch of images.

## EfficientFormerModel

### class transformers.EfficientFormerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_efficientformer.py#L557)

( config: EfficientFormerConfig )

Parameters

-   **config** ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_efficientformer.py#L569)

( pixel\_values: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [ViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.preprocess()](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor.preprocess) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [EfficientFormerModel](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, EfficientFormerModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
>>> model = EfficientFormerModel.from_pretrained("snap-research/efficientformer-l1-300")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 49, 448]
```

## EfficientFormerForImageClassification

### class transformers.EfficientFormerForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_efficientformer.py#L619)

( config: EfficientFormerConfig )

Parameters

-   **config** ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

EfficientFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_efficientformer.py#L634)

( pixel\_values: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [ViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.preprocess()](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor.preprocess) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the model at the output of each stage.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [EfficientFormerForImageClassification](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, EfficientFormerForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
>>> model = EfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
Egyptian cat
```

## EfficientFormerForImageClassificationWithTeacher

### class transformers.EfficientFormerForImageClassificationWithTeacher

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_efficientformer.py#L749)

( config: EfficientFormerConfig )

Parameters

-   **config** ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden state of the \[CLS\] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet supported.

This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_efficientformer.py#L766)

( pixel\_values: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.efficientformer.modeling_efficientformer.EfficientFormerForImageClassificationWithTeacherOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [ViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor). See [ViTImageProcessor.preprocess()](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor.preprocess) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.efficientformer.modeling_efficientformer.EfficientFormerForImageClassificationWithTeacherOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.efficientformer.modeling_efficientformer.EfficientFormerForImageClassificationWithTeacherOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Prediction scores as the average of the cls\_logits and distillation logits.
-   **cls\_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the class token).
-   **distillation\_logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the distillation token).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The [EfficientFormerForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerForImageClassificationWithTeacher) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, EfficientFormerForImageClassificationWithTeacher
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
>>> model = EfficientFormerForImageClassificationWithTeacher.from_pretrained("snap-research/efficientformer-l1-300")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
Egyptian cat
```

## TFEfficientFormerModel

### class transformers.TFEfficientFormerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_tf_efficientformer.py#L775)

( \*args \*\*kwargs )

Parameters

-   **config** ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top. This model is a TensorFlow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_tf_efficientformer.py#L781)

( pixel\_values: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None training: bool = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or `tuple(tf.Tensor)`

Parameters

-   **pixel\_values** ((`tf.Tensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [EfficientFormerImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`tf.Tensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
    This output is usually _not_ a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFEfficientFormerModel](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.TFEfficientFormerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFEfficientFormerModel
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
>>> model = TFEfficientFormerModel.from_pretrained("snap-research/efficientformer-l1-300")

>>> inputs = image_processor(image, return_tensors="tf")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 49, 448]
```

## TFEfficientFormerForImageClassification

### class transformers.TFEfficientFormerForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_tf_efficientformer.py#L815)

( \*args \*\*kwargs )

Parameters

-   **config** ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

EfficientFormer Model transformer with an image classification head on top of pooled last hidden state, e.g. for ImageNet.

This model is a TensorFlow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_tf_efficientformer.py#L829)

( pixel\_values: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = None labels: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None training: bool = False ) → `transformers.modeling_tf_outputs.TFImageClassifierOutput` or `tuple(tf.Tensor)`

Parameters

-   **pixel\_values** ((`tf.Tensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [EfficientFormerImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`tf.Tensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

`transformers.modeling_tf_outputs.TFImageClassifierOutput` or `tuple(tf.Tensor)`

A `transformers.modeling_tf_outputs.TFImageClassifierOutput` or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the model at the output of each stage.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, patch_size, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFEfficientFormerForImageClassification](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.TFEfficientFormerForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFEfficientFormerForImageClassification
>>> import tensorflow as tf
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
>>> model = TFEfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")

>>> inputs = image_processor(image, return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> predicted_label = int(tf.math.argmax(logits, axis=-1))
>>> print(model.config.id2label[predicted_label])
LABEL_281
```

## TFEfficientFormerForImageClassificationWithTeacher

### class transformers.TFEfficientFormerForImageClassificationWithTeacher

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_tf_efficientformer.py#L920)

( \*args \*\*kwargs )

Parameters

-   **config** ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden state and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

.. warning:: This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet supported.

This model is a TensorFlow [tf.keras.layers.Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer). Use it as a regular TensorFlow Module and refer to the TensorFlow documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/efficientformer/modeling_tf_efficientformer.py#L939)

( pixel\_values: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None training: bool = False ) → `transformers.models.efficientformer.modeling_tf_efficientformer.TFEfficientFormerForImageClassificationWithTeacherOutput` or `tuple(tf.Tensor)`

Parameters

-   **pixel\_values** ((`tf.Tensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [EfficientFormerImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.efficientformer.modeling_tf_efficientformer.TFEfficientFormerForImageClassificationWithTeacherOutput` or `tuple(tf.Tensor)`

A `transformers.models.efficientformer.modeling_tf_efficientformer.TFEfficientFormerForImageClassificationWithTeacherOutput` or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([EfficientFormerConfig](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerConfig)) and inputs.

The [TFEfficientFormerForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.TFEfficientFormerForImageClassificationWithTeacher) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

-   **Output** type of [EfficientFormerForImageClassificationWithTeacher](/docs/transformers/v4.34.0/en/model_doc/efficientformer#transformers.EfficientFormerForImageClassificationWithTeacher). logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Prediction scores as the average of the cls\_logits and distillation logits. cls\_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the class token). distillation\_logits (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the distillation token). hidden\_states (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs. attentions (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Example:

```
>>> from transformers import AutoImageProcessor, TFEfficientFormerForImageClassificationWithTeacher
>>> import tensorflow as tf
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
>>> model = TFEfficientFormerForImageClassificationWithTeacher.from_pretrained("snap-research/efficientformer-l1-300")

>>> inputs = image_processor(image, return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> predicted_label = int(tf.math.argmax(logits, axis=-1))
>>> print(model.config.id2label[predicted_label])
LABEL_281
```