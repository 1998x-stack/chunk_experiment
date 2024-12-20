# Chinese-CLIP

## Overview

The Chinese-CLIP model was proposed in [Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese](https://arxiv.org/abs/2211.01335) by An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, Chang Zhou. Chinese-CLIP is an implementation of CLIP (Radford et al., 2021) on a large-scale dataset of Chinese image-text pairs. It is capable of performing cross-modal retrieval and also playing as a vision backbone for vision tasks like zero-shot image classification, open-domain object detection, etc. The original Chinese-CLIP code is released [at this link](https://github.com/OFA-Sys/Chinese-CLIP).

The abstract from the paper is the following:

_The tremendous success of CLIP (Radford et al., 2021) has promoted the research and application of contrastive learning for vision-language pretraining. In this work, we construct a large-scale dataset of image-text pairs in Chinese, where most data are retrieved from publicly available datasets, and we pretrain Chinese CLIP models on the new dataset. We develop 5 Chinese CLIP models of multiple sizes, spanning from 77 to 958 million parameters. Furthermore, we propose a two-stage pretraining method, where the model is first trained with the image encoder frozen and then trained with all parameters being optimized, to achieve enhanced model performance. Our comprehensive experiments demonstrate that Chinese CLIP can achieve the state-of-the-art performance on MUGE, Flickr30K-CN, and COCO-CN in the setups of zero-shot learning and finetuning, and it is able to achieve competitive performance in zero-shot image classification based on the evaluation on the ELEVATER benchmark (Li et al., 2022). Our codes, pretrained models, and demos have been released._

## Usage

The code snippet below shows how to compute image & text features and similarities:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import ChineseCLIPProcessor, ChineseCLIPModel

>>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> 
>>> texts = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

>>> 
>>> inputs = processor(images=image, return_tensors="pt")
>>> image_features = model.get_image_features(**inputs)
>>> image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  

>>> 
>>> inputs = processor(text=texts, padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
>>> text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  

>>> 
>>> inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

Currently, we release the following scales of pretrained Chinese-CLIP models at HF Model Hub:

-   [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
-   [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14)
-   [OFA-Sys/chinese-clip-vit-large-patch14-336px](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14-336px)
-   [OFA-Sys/chinese-clip-vit-huge-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-huge-patch14)

The Chinese-CLIP model was contributed by [OFA-Sys](https://huggingface.co/OFA-Sys).

## ChineseCLIPConfig

### class transformers.ChineseCLIPConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/configuration_chinese_clip.py#L268)

( text\_config = None vision\_config = None projection\_dim = 512 logit\_scale\_init\_value = 2.6592 \*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [ChineseCLIPTextConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [ChineseCLIPVisionConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPVisionConfig).
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Dimentionality of text and vision projection layers.
-   **logit\_scale\_init\_value** (`float`, _optional_, defaults to 2.6592) — The inital value of the _logit\_scale_ paramter. Default is used as per the original ChineseCLIP implementation.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[ChineseCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig) is the configuration class to store the configuration of a [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel). It is used to instantiate Chinese-CLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the Chinese-CLIP [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ChineseCLIPConfig, ChineseCLIPModel

>>> 
>>> configuration = ChineseCLIPConfig()

>>> 
>>> model = ChineseCLIPModel(configuration)

>>> 
>>> configuration = model.config

>>> 

>>> 
>>> config_text = ChineseCLIPTextConfig()
>>> config_vision = ChineseCLIPVisionConfig()

>>> config = ChineseCLIPConfig.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/configuration_chinese_clip.py#L406)

( text\_config: ChineseCLIPTextConfig vision\_config: ChineseCLIPVisionConfig \*\*kwargs )

Instantiate a [ChineseCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig) (or a derived class) from Chinese-CLIP text model configuration and Chinese-CLIP vision model configuration. Returns: [ChineseCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig): An instance of a configuration object

## ChineseCLIPTextConfig

### class transformers.ChineseCLIPTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/configuration_chinese_clip.py#L40)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 initializer\_factor = 1.0 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 position\_embedding\_type = 'absolute' use\_cache = True \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 30522) — Vocabulary size of the CHINESE\_CLIP model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel).
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
-   **hidden\_act** (`str` or `Callable`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 512) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **type\_vocab\_size** (`int`, _optional_, defaults to 2) — The vocabulary size of the `token_type_ids` passed when calling [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel).
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **position\_embedding\_type** (`str`, _optional_, defaults to `"absolute"`) — Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155). For more information on `"relative_key_query"`, please refer to _Method 4_ in [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.

This is the configuration class to store the configuration of a [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel). It is used to instantiate a Chinese CLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Chinese CLIP \[OFA-Sys/chinese-clip-vit-base-patch16\](https: //huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ChineseCLIPTextConfig, ChineseCLIPTextModel

>>> 
>>> configuration = ChineseCLIPTextConfig()

>>> 
>>> model = ChineseCLIPTextModel(configuration)

>>> 
>>> configuration = model.config
```

## ChineseCLIPVisionConfig

### class transformers.ChineseCLIPVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/configuration_chinese_clip.py#L163)

( hidden\_size = 768 intermediate\_size = 3072 projection\_dim = 512 num\_hidden\_layers = 12 num\_attention\_heads = 12 num\_channels = 3 image\_size = 224 patch\_size = 32 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 \*\*kwargs )

Parameters

-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **image\_size** (`int`, _optional_, defaults to 224) — The size (resolution) of each image.
-   **patch\_size** (`int`, _optional_, defaults to 32) — The size (resolution) of each patch.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"quick_gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` \``"quick_gelu"` are supported.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization layers.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **initializer\_factor** (\`float“, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

This is the configuration class to store the configuration of a [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel). It is used to instantiate an ChineseCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ChineseCLIP \[OFA-Sys/chinese-clip-vit-base-patch16\](https: //huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ChineseCLIPVisionConfig, ChineseCLIPVisionModel

>>> 
>>> configuration = ChineseCLIPVisionConfig()

>>> 
>>> model = ChineseCLIPVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## ChineseCLIPImageProcessor

### class transformers.ChineseCLIPImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/image_processing_chinese_clip.py#L50)

( do\_resize: bool = True size: typing.Dict\[str, int\] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_center\_crop: bool = True crop\_size: typing.Dict\[str, int\] = None do\_rescale: bool = True rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None do\_convert\_rgb: bool = True \*\*kwargs )

Parameters

-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by `do_resize` in the `preprocess` method.
-   **size** (`Dict[str, int]` _optional_, defaults to `{"shortest_edge" -- 224}`): Size of the image after resizing. The shortest edge of the image is resized to size\[“shortest\_edge”\], with the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess` method.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BICUBIC`) — Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `True`) — Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the `preprocess` method.
-   **crop\_size** (`Dict[str, int]` _optional_, defaults to 224) — Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess` method.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess` method. do\_normalize — Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_MEAN`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_STD`) — Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method. Can be overridden by the `image_std` parameter in the `preprocess` method.
-   **do\_convert\_rgb** (`bool`, _optional_, defaults to `True`) — Whether to convert the image to RGB.

Constructs a Chinese-CLIP image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/image_processing_chinese_clip.py#L163)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] do\_resize: bool = None size: typing.Dict\[str, int\] = None resample: Resampling = None do\_center\_crop: bool = None crop\_size: int = None do\_rescale: bool = None rescale\_factor: float = None do\_normalize: bool = None image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None do\_convert\_rgb: bool = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None data\_format: typing.Optional\[transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
-   **do\_resize** (`bool`, _optional_, defaults to `self.do_resize`) — Whether to resize the image.
-   **size** (`Dict[str, int]`, _optional_, defaults to `self.size`) — Size of the image after resizing. Shortest edge of the image is resized to size\[“shortest\_edge”\], with the longest edge resized to keep the input aspect ratio.
-   **resample** (`int`, _optional_, defaults to `self.resample`) — Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `self.do_center_crop`) — Whether to center crop the image.
-   **crop\_size** (`Dict[str, int]`, _optional_, defaults to `self.crop_size`) — Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
-   **do\_rescale** (`bool`, _optional_, defaults to `self.do_rescale`) — Whether to rescale the image.
-   **rescale\_factor** (`float`, _optional_, defaults to `self.rescale_factor`) — Rescale factor to rescale the image by if `do_rescale` is set to `True`.
-   **do\_normalize** (`bool`, _optional_, defaults to `self.do_normalize`) — Whether to normalize the image.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `self.image_mean`) — Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `self.image_std`) — Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.
-   **do\_convert\_rgb** (`bool`, _optional_, defaults to `self.do_convert_rgb`) — Whether to convert the image to RGB.
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

## ChineseCLIPFeatureExtractor

## ChineseCLIPProcessor

### class transformers.ChineseCLIPProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/processing_chinese_clip.py#L25)

( image\_processor = None tokenizer = None \*\*kwargs )

Parameters

-   **image\_processor** ([ChineseCLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPImageProcessor)) — The image processor is a required input.
-   **tokenizer** ([BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast)) — The tokenizer is a required input.

Constructs a Chinese-CLIP processor which wraps a Chinese-CLIP image processor and a Chinese-CLIP tokenizer into a single processor.

[ChineseCLIPProcessor](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPProcessor) offers all the functionalities of [ChineseCLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPImageProcessor) and [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast). See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPProcessor.decode) for more information.

This method forwards all its arguments to BertTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to BertTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## ChineseCLIPModel

### class transformers.ChineseCLIPModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1351)

( config: ChineseCLIPConfig )

Parameters

-   **config** ([ChineseCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1486)

( input\_ids: typing.Optional\[torch.LongTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None return\_loss: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.chinese_clip.modeling_chinese_clip.ChineseCLIPOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [ChineseCLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **return\_loss** (`bool`, _optional_) — Whether or not to return the contrastive loss.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.chinese_clip.modeling_chinese_clip.ChineseCLIPOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.chinese_clip.modeling_chinese_clip.ChineseCLIPOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.chinese_clip.configuration_chinese_clip.ChineseCLIPConfig'>`) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
-   **logits\_per\_image:(`torch.FloatTensor`** of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text similarity scores.
-   **logits\_per\_text:(`torch.FloatTensor`** of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image similarity scores.
-   **text\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [ChineseCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPTextModel).
-   **image\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [ChineseCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPVisionModel).
-   **text\_model\_output(`BaseModelOutputWithPoolingAndCrossAttentions`):** The output of the [ChineseCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPTextModel).
-   **vision\_model\_output(`BaseModelOutputWithPoolingAndCrossAttentions`):** The output of the [ChineseCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPVisionModel).

The [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, ChineseCLIPModel

>>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1386)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by applying the projection layer to the final \[CLS\] hidden state of Text-Transformer.

The [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, ChineseCLIPModel

>>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> inputs = tokenizer(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
>>> text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1436)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [ChineseCLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the final \[CLS\] hidden state of Vision-Transformer.

The [ChineseCLIPModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, ChineseCLIPModel

>>> model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> processor = AutoProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
>>> image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
```

## ChineseCLIPTextModel

### class transformers.ChineseCLIPTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1114)

( config add\_pooling\_layer = True )

Parameters

-   **config** ([ChineseCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text model from CHINESE\_CLIP without any head or projection on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1155)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None encoder\_hidden\_states: typing.Optional\[torch.Tensor\] = None encoder\_attention\_mask: typing.Optional\[torch.Tensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [ChineseCLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **return\_loss** (`bool`, _optional_) — Whether or not to return the contrastive loss.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.
-   **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) — Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

A [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([ChineseCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    

The [ChineseCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, ChineseCLIPTextModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> model = ChineseCLIPTextModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## ChineseCLIPVisionModel

### class transformers.ChineseCLIPVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1296)

( config: ChineseCLIPVisionConfig )

Parameters

-   **config** ([ChineseCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision model from CHINESE\_CLIP without any head or projection on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1309)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [ChineseCLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.chinese_clip.configuration_chinese_clip.ChineseCLIPVisionConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [ChineseCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/chinese_clip#transformers.ChineseCLIPVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import CLIPProcessor, ChineseCLIPVisionModel

>>> model = ChineseCLIPVisionModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
>>> processor = CLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")

>>> url = "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/pokemon.jpeg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```