# FLAVA

## Overview

The FLAVA model was proposed in [FLAVA: A Foundational Language And Vision Alignment Model](https://arxiv.org/abs/2112.04482) by Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela and is accepted at CVPR 2022.

The paper aims at creating a single unified foundation model which can work across vision, language as well as vision-and-language multimodal tasks.

The abstract from the paper is the following:

_State-of-the-art vision and vision-and-language models rely on large-scale visio-linguistic pretraining for obtaining good performance on a variety of downstream tasks. Generally, such models are often either cross-modal (contrastive) or multi-modal (with earlier fusion) but not both; and they often only target specific modalities or tasks. A promising direction would be to use a single holistic universal model, as a “foundation”, that targets all modalities at once — a true vision and language foundation model should be good at vision tasks, language tasks, and cross- and multi-modal vision and language tasks. We introduce FLAVA as such a model and demonstrate impressive performance on a wide range of 35 tasks spanning these target modalities._

This model was contributed by [aps](https://huggingface.co/aps). The original code can be found [here](https://github.com/facebookresearch/multimodal/tree/main/examples/flava).

## FlavaConfig

### class transformers.FlavaConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/configuration_flava.py#L467)

( image\_config: typing.Dict\[str, typing.Any\] = Nonetext\_config: typing.Dict\[str, typing.Any\] = Nonemultimodal\_config: typing.Dict\[str, typing.Any\] = Noneimage\_codebook\_config: typing.Dict\[str, typing.Any\] = Nonehidden\_size: int = 768layer\_norm\_eps: float = 1e-12projection\_dim: int = 768init\_codebook: bool = Truelogit\_scale\_init\_value: float = 2.6592initializer\_range: float = 0.02ce\_ignore\_index: int = -100mim\_weight: float = 1.0mlm\_weight: float = 1.0global\_contrastive\_weight: float = 1.0itm\_weight: float = 1.0mmm\_image\_weight: float = 1.0mmm\_text\_weight: float = 1.0global\_backprop\_contrastive: bool = Trueskip\_unmasked\_multimodal\_encoder: bool = Truereturn\_loss: bool = True\*\*kwargs )

[FlavaConfig](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaConfig) is the configuration class to store the configuration of a [FlavaModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaModel). It is used to instantiate FLAVA model according to the specified arguments, defining the text model, image model, image codebook and multimodal model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import FlavaConfig, FlavaModel, FlavaForPreTraining

>>> 
>>> configuration = FlavaConfig()

>>> 
>>> model = FlavaModel(configuration)
>>> model_pre = FlavaForPreTraining(configuration)

>>> 
>>> configuration = model.config
>>> configuration_pre = model_pre.config
```

#### from\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/configuration_flava.py#L741)

( image\_config: FlavaImageConfigtext\_config: FlavaTextConfigmultimodal\_config: FlavaMultimodalConfigimage\_codebook\_config: FlavaImageCodebookConfig\*\*kwargs ) → [FlavaConfig](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaConfig)

An instance of a configuration object

Instantiate a [FlavaConfig](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaConfig) (or a derived class) from flava text model configuration, flava image model configuration, flava multimodal model and flava codebook model configuration.

## FlavaTextConfig

### class transformers.FlavaTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/configuration_flava.py#L150)

( vocab\_size: int = 30522type\_vocab\_size: int = 2max\_position\_embeddings: int = 512position\_embedding\_type: str = 'absolute'hidden\_size: int = 768num\_hidden\_layers: int = 12num\_attention\_heads: int = 12intermediate\_size: int = 3072hidden\_act: str = 'gelu'hidden\_dropout\_prob: float = 0.0attention\_probs\_dropout\_prob: float = 0.0initializer\_range: float = 0.02layer\_norm\_eps: float = 1e-12pad\_token\_id: int = 0qkv\_bias: bool = True\*\*kwargs )

This is the configuration class to store the configuration of a [FlavaTextModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaTextModel). It is used to instantiate an FLAVA model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import FlavaTextConfig, FlavaTextModel

>>> 
>>> configuration = FlavaTextConfig()

>>> 
>>> model = FlavaTextModel(configuration)

>>> 
>>> configuration = model.config
```

## FlavaImageConfig

### class transformers.FlavaImageConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/configuration_flava.py#L31)

( hidden\_size: int = 768num\_hidden\_layers: int = 12num\_attention\_heads: int = 12intermediate\_size: int = 3072hidden\_act: int = 'gelu'hidden\_dropout\_prob: float = 0.0attention\_probs\_dropout\_prob: float = 0.0initializer\_range: float = 0.02layer\_norm\_eps: float = 1e-12image\_size: int = 224patch\_size: int = 16num\_channels: int = 3qkv\_bias: bool = Truemask\_token: bool = Truevocab\_size: int = 8192\*\*kwargs )

This is the configuration class to store the configuration of a [FlavaImageModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaImageModel). It is used to instantiate an FLAVA model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import FlavaImageConfig, FlavaImageModel

>>> 
>>> configuration = FlavaImageConfig()

>>> 
>>> model = FlavaImageModel(configuration)

>>> 
>>> configuration = model.config
```

## FlavaMultimodalConfig

### class transformers.FlavaMultimodalConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/configuration_flava.py#L279)

( hidden\_size: int = 768num\_hidden\_layers: int = 6num\_attention\_heads: int = 12intermediate\_size: int = 3072hidden\_act: int = 'gelu'hidden\_dropout\_prob: int = 0.0attention\_probs\_dropout\_prob: int = 0.0initializer\_range: float = 0.02layer\_norm\_eps: float = 1e-12qkv\_bias: bool = Trueuse\_cls\_token: bool = True\*\*kwargs )

This is the configuration class to store the configuration of a [FlavaMultimodalModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaMultimodalModel). It is used to instantiate an FLAVA model according to the specified arguments, defining the model architecture.

Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA [facebook/flava-full](https://huggingface.co/facebook/flava-full) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import FlavaMultimodalConfig, FlavaMultimodalModel

>>> 
>>> configuration = FlavaMultimodalConfig()

>>> 
>>> model = FlavaMultimodalModel(configuration)

>>> 
>>> configuration = model.config
```

## FlavaImageCodebookConfig

### class transformers.FlavaImageCodebookConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/configuration_flava.py#L382)

( num\_groups: int = 4input\_channels: int = 3num\_blocks\_per\_group: int = 2hidden\_size: int = 256vocab\_size: int = 8192freeze: int = Trueinitializer\_range: float = 0.02\*\*kwargs )

## FlavaProcessor

### class transformers.FlavaProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/processing_flava.py#L28)

( image\_processor = Nonetokenizer = None\*\*kwargs )

Parameters

-   **image\_processor** ([FlavaImageProcessor](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaImageProcessor)) — The image processor is a required input.
-   **tokenizer** ([BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast)) — The tokenizer is a required input.

Constructs a FLAVA processor which wraps a FLAVA image processor and a FLAVA tokenizer into a single processor.

[FlavaProcessor](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaProcessor) offers all the functionalities of [FlavaImageProcessor](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaImageProcessor) and [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast). See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaProcessor.decode) for more information.

This method forwards all its arguments to BertTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to BertTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## FlavaFeatureExtractor

## FlavaImageProcessor

### class transformers.FlavaImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/image_processing_flava.py#L135)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BICUBIC: 3>do\_center\_crop: bool = Truecrop\_size: typing.Dict\[str, int\] = Nonedo\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.Iterable\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.Iterable\[float\], NoneType\] = Nonereturn\_image\_mask: bool = Falseinput\_size\_patches: int = 14total\_mask\_patches: int = 75mask\_group\_min\_patches: int = 16mask\_group\_max\_patches: typing.Optional\[int\] = Nonemask\_group\_min\_aspect\_ratio: float = 0.3mask\_group\_max\_aspect\_ratio: typing.Optional\[float\] = Nonereturn\_codebook\_pixels: bool = Falsecodebook\_do\_resize: bool = Truecodebook\_size: bool = Nonecodebook\_resample: int = <Resampling.LANCZOS: 1>codebook\_do\_center\_crop: bool = Truecodebook\_crop\_size: int = Nonecodebook\_do\_rescale: bool = Truecodebook\_rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098codebook\_do\_map\_pixels: bool = Truecodebook\_do\_normalize: bool = Truecodebook\_image\_mean: typing.Union\[float, typing.Iterable\[float\], NoneType\] = Nonecodebook\_image\_std: typing.Union\[float, typing.Iterable\[float\], NoneType\] = None\*\*kwargs )

Constructs a Flava image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/image_processing_flava.py#L447)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: typing.Optional\[bool\] = Nonesize: typing.Dict\[str, int\] = Noneresample: Resampling = Nonedo\_center\_crop: typing.Optional\[bool\] = Nonecrop\_size: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonereturn\_image\_mask: typing.Optional\[bool\] = Noneinput\_size\_patches: typing.Optional\[int\] = Nonetotal\_mask\_patches: typing.Optional\[int\] = Nonemask\_group\_min\_patches: typing.Optional\[int\] = Nonemask\_group\_max\_patches: typing.Optional\[int\] = Nonemask\_group\_min\_aspect\_ratio: typing.Optional\[float\] = Nonemask\_group\_max\_aspect\_ratio: typing.Optional\[float\] = Nonereturn\_codebook\_pixels: typing.Optional\[bool\] = Nonecodebook\_do\_resize: typing.Optional\[bool\] = Nonecodebook\_size: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonecodebook\_resample: typing.Optional\[int\] = Nonecodebook\_do\_center\_crop: typing.Optional\[bool\] = Nonecodebook\_crop\_size: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonecodebook\_do\_rescale: typing.Optional\[bool\] = Nonecodebook\_rescale\_factor: typing.Optional\[float\] = Nonecodebook\_do\_map\_pixels: typing.Optional\[bool\] = Nonecodebook\_do\_normalize: typing.Optional\[bool\] = Nonecodebook\_image\_mean: typing.Optional\[typing.Iterable\[float\]\] = Nonecodebook\_image\_std: typing.Optional\[typing.Iterable\[float\]\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## FlavaForPreTraining

### class transformers.FlavaForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1727)

( config: FlavaConfigimage\_codebook: typing.Optional\[torch.nn.modules.module.Module\] = None )

Parameters

-   **config** ([FlavaConfig](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
-   **image\_codebook** (`nn.Module`) — If passed, the image codebook will be set to this. Otherwise. it will be initialized using the image\_codebook\_config defined in the config first as the first parameter.

The FLAVA model for pretraining which outputs losses, embeddings, logits and transformer outputs.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1771)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneinput\_ids\_masked: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Nonecodebook\_pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneimage\_attention\_mask: typing.Optional\[torch.Tensor\] = Noneskip\_unmasked\_multimodal\_encoder: bool = Nonemlm\_labels: typing.Optional\[torch.Tensor\] = Nonemim\_labels: typing.Optional\[torch.Tensor\] = Noneitm\_labels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: bool = Truereturn\_dict: typing.Optional\[bool\] = Nonereturn\_loss: typing.Optional\[bool\] = None ) → `transformers.models.flava.modeling_flava.FlavaForPreTrainingOutput` or `tuple(torch.FloatTensor)`

The [FlavaForPreTraining](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## FlavaModel

### class transformers.FlavaModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1193)

( config: FlavaConfig )

Parameters

-   **config** ([FlavaConfig](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare FLAVA Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1337)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneimage\_attention\_mask: typing.Optional\[torch.Tensor\] = Noneskip\_multimodal\_encoder: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: bool = Truereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.flava.modeling_flava.FlavaModelOutput` or `tuple(torch.FloatTensor)`

The [FlavaModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, FlavaModel

>>> model = FlavaModel.from_pretrained("facebook/flava-full")
>>> processor = AutoProcessor.from_pretrained("facebook/flava-full")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.contrastive_logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1239)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [FlavaModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1285)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Noneinterpolate\_pos\_encoding: typing.Optional\[bool\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

The [FlavaModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## FlavaImageCodebook

### class transformers.FlavaImageCodebook

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1511)

( config: FlavaImageCodebookConfig\*\*kwargs: typing.Any )

Parameters

-   **config** ([FlavaImageCodebookConfig](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaImageCodebookConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The FLAVA’s image codebook model inspired from DALL-E’s original encoder. Outputs raw hidden states and can be used to generate image tokens for an image based on DALL-E’s vocab. Used to generate labels for MIM. Use `get_codebook_indices` to get image tokens for an image.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### get\_codebook\_indices

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1561)

( pixel\_values: Tensor )

#### get\_codebook\_probs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1591)

( pixel\_values: Tensor )

## FlavaTextModel

### class transformers.FlavaTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L990)

( config: FlavaTextConfigadd\_pooling\_layer: bool = True )

Parameters

-   **config** ([FlavaTextConfig](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaTextConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare FLAVA Text Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1021)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [FlavaTextModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlavaTextModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
>>> model = FlavaTextModel.from_pretrained("facebook/flava-full")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## FlavaImageModel

### class transformers.FlavaImageModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L891)

( config: FlavaImageConfigadd\_pooling\_layer: bool = True )

Parameters

-   **config** ([FlavaImageConfig](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaImageConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare FLAVA Image Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L924)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Noneinterpolate\_pos\_encoding: typing.Optional\[bool\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [FlavaImageModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaImageModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, FlavaImageModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/flava-full")
>>> model = FlavaImageModel.from_pretrained("facebook/flava-full")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 197, 768]
```

## FlavaMultimodalModel

### class transformers.FlavaMultimodalModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1095)

( config: FlavaMultimodalConfigadd\_pooling\_layer = True )

Parameters

-   **config** ([FlavaMultimodalConfig](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaMultimodalConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare FLAVA Multimodal Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/flava/modeling_flava.py#L1123)

( hidden\_states: Tensorattention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [FlavaMultimodalModel](/docs/transformers/v4.34.0/en/model_doc/flava#transformers.FlavaMultimodalModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlavaMultimodalModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/flava-full")
>>> model = FlavaMultimodalModel.from_pretrained("facebook/flava-full")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```