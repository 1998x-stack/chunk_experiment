# BridgeTower

## Overview

The BridgeTower model was proposed in [BridgeTower: Building Bridges Between Encoders in Vision-Language Representative Learning](https://arxiv.org/abs/2206.08657) by Xiao Xu, Chenfei Wu, Shachar Rosenman, Vasudev Lal, Wanxiang Che, Nan Duan. The goal of this model is to build a bridge between each uni-modal encoder and the cross-modal encoder to enable comprehensive and detailed interaction at each layer of the cross-modal encoder thus achieving remarkable performance on various downstream tasks with almost negligible additional performance and computational costs.

This paper has been accepted to the [AAAI’23](https://aaai.org/Conferences/AAAI-23/) conference.

The abstract from the paper is the following:

_Vision-Language (VL) models with the TWO-TOWER architecture have dominated visual-language representation learning in recent years. Current VL models either use lightweight uni-modal encoders and learn to extract, align and fuse both modalities simultaneously in a deep cross-modal encoder, or feed the last-layer uni-modal representations from the deep pre-trained uni-modal encoders into the top cross-modal encoder. Both approaches potentially restrict vision-language representation learning and limit model performance. In this paper, we propose BRIDGETOWER, which introduces multiple bridge layers that build a connection between the top layers of uni-modal encoders and each layer of the crossmodal encoder. This enables effective bottom-up cross-modal alignment and fusion between visual and textual representations of different semantic levels of pre-trained uni-modal encoders in the cross-modal encoder. Pre-trained with only 4M images, BRIDGETOWER achieves state-of-the-art performance on various downstream vision-language tasks. In particular, on the VQAv2 test-std set, BRIDGETOWER achieves an accuracy of 78.73%, outperforming the previous state-of-the-art model METER by 1.09% with the same pre-training data and almost negligible additional parameters and computational costs. Notably, when further scaling the model, BRIDGETOWER achieves an accuracy of 81.15%, surpassing models that are pre-trained on orders-of-magnitude larger datasets._

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/bridgetower_architecture%20.jpg) BridgeTower architecture. Taken from the [original paper.](https://arxiv.org/abs/2206.08657)

## Usage

BridgeTower consists of a visual encoder, a textual encoder and cross-modal encoder with multiple lightweight bridge layers. The goal of this approach was to build a bridge between each uni-modal encoder and the cross-modal encoder to enable comprehensive and detailed interaction at each layer of the cross-modal encoder. In principle, one can apply any visual, textual or cross-modal encoder in the proposed architecture.

The [BridgeTowerProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerProcessor) wraps [RobertaTokenizer](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaTokenizer) and [BridgeTowerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerImageProcessor) into a single instance to both encode the text and prepare the images respectively.

The following example shows how to run contrastive learning using [BridgeTowerProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerProcessor) and [BridgeTowerForContrastiveLearning](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerForContrastiveLearning).

```
>>> from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
>>> import requests
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
>>> model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

>>> 
>>> scores = dict()
>>> for text in texts:
...     
...     encoding = processor(image, text, return_tensors="pt")
...     outputs = model(**encoding)
...     scores[text] = outputs
```

The following example shows how to run image-text retrieval using [BridgeTowerProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerProcessor) and [BridgeTowerForImageAndTextRetrieval](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerForImageAndTextRetrieval).

```
>>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
>>> import requests
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
>>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

>>> 
>>> scores = dict()
>>> for text in texts:
...     
...     encoding = processor(image, text, return_tensors="pt")
...     outputs = model(**encoding)
...     scores[text] = outputs.logits[0, 1].item()
```

The following example shows how to run masked language modeling using [BridgeTowerProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerProcessor) and [BridgeTowerForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerForMaskedLM).

```
>>> from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000360943.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
>>> text = "a <mask> looking out of the window"

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
>>> model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

>>> 
>>> encoding = processor(image, text, return_tensors="pt")

>>> 
>>> outputs = model(**encoding)

>>> results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())

>>> print(results)
.a cat looking out of the window.
```

This model was contributed by [Anahita Bhiwandiwalla](https://huggingface.co/anahita-b), [Tiep Le](https://huggingface.co/Tile) and [Shaoyen Tseng](https://huggingface.co/shaoyent). The original code can be found [here](https://github.com/microsoft/BridgeTower).

Tips:

-   This implementation of BridgeTower uses [RobertaTokenizer](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaTokenizer) to generate text embeddings and OpenAI’s CLIP/ViT model to compute visual embeddings.
-   Checkpoints for pre-trained [bridgeTower-base](https://huggingface.co/BridgeTower/bridgetower-base) and [bridgetower masked language modeling and image text matching](https://huggingface.co/BridgeTower/bridgetower-base-itm-mlm) are released.
-   Please refer to [Table 5](https://arxiv.org/pdf/2206.08657.pdf) for BridgeTower’s performance on Image Retrieval and other down stream tasks.
-   The PyTorch version of this model is only available in torch 1.10 and higher.

## BridgeTowerConfig

### class transformers.BridgeTowerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/configuration_bridgetower.py#L241)

( share\_cross\_modal\_transformer\_layers = True hidden\_act = 'gelu' hidden\_size = 768 initializer\_factor = 1 layer\_norm\_eps = 1e-05 share\_link\_tower\_layers = False link\_tower\_type = 'add' num\_attention\_heads = 12 num\_hidden\_layers = 6 tie\_word\_embeddings = False init\_layernorm\_from\_vision\_encoder = False text\_config = None vision\_config = None \*\*kwargs )

Parameters

-   **share\_cross\_modal\_transformer\_layers** (`bool`, _optional_, defaults to `True`) — Whether cross modal transformer layers are shared.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler.
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **initializer\_factor** (\`float“, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-05) — The epsilon used by the layer normalization layers.
-   **share\_link\_tower\_layers** (`bool`, _optional_, defaults to `False`) — Whether the bride/link tower layers are shared.
-   **link\_tower\_type** (`str`, _optional_, defaults to `"add"`) — Type of the bridge/link layer.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 6) — Number of hidden layers in the Transformer encoder.
-   **tie\_word\_embeddings** (`bool`, _optional_, defaults to `False`) — Whether to tie input and output embeddings.
-   **init\_layernorm\_from\_vision\_encoder** (`bool`, _optional_, defaults to `False`) — Whether to init LayerNorm from the vision encoder.
-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [BridgeTowerTextConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [BridgeTowerVisionConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerVisionConfig).

This is the configuration class to store the configuration of a [BridgeTowerModel](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerModel). It is used to instantiate a BridgeTower model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the bridgetower-base [BridgeTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BridgeTowerModel, BridgeTowerConfig

>>> 
>>> configuration = BridgeTowerConfig()

>>> 
>>> model = BridgeTowerModel(configuration)

>>> 
>>> configuration = model.config
```

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/configuration_bridgetower.py#L341)

( text\_config: BridgeTowerTextConfig vision\_config: BridgeTowerVisionConfig \*\*kwargs )

Instantiate a [BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig) (or a derived class) from BridgeTower text model configuration. Returns: [BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig): An instance of a configuration object

## BridgeTowerTextConfig

### class transformers.BridgeTowerTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/configuration_bridgetower.py#L120)

( vocab\_size = 50265 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 initializer\_factor = 1 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 514 type\_vocab\_size = 1 layer\_norm\_eps = 1e-05 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 position\_embedding\_type = 'absolute' use\_cache = True \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 50265) — Vocabulary size of the text part of the model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [BridgeTowerModel](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerModel).
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
-   **hidden\_act** (`str` or `Callable`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 514) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **type\_vocab\_size** (`int`, _optional_, defaults to 2) — The vocabulary size of the `token_type_ids`.
-   **initializer\_factor** (\`float“, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-05) — The epsilon used by the layer normalization layers.
-   **position\_embedding\_type** (`str`, _optional_, defaults to `"absolute"`) — Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155). For more information on `"relative_key_query"`, please refer to _Method 4_ in [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
-   **is\_decoder** (`bool`, _optional_, defaults to `False`) — Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.

This is the configuration class to store the text configuration of a [BridgeTowerModel](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerModel). The default values here are copied from RoBERTa. Instantiating a configuration with the defaults will yield a similar configuration to that of the bridgetower-base [BridegTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BridgeTowerTextConfig

>>> 
>>> configuration = BridgeTowerTextConfig()

>>> 
>>> configuration
```

## BridgeTowerVisionConfig

### class transformers.BridgeTowerVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/configuration_bridgetower.py#L34)

( hidden\_size = 768 num\_hidden\_layers = 12 num\_channels = 3 patch\_size = 16 image\_size = 288 initializer\_factor = 1 layer\_norm\_eps = 1e-05 stop\_gradient = False share\_layernorm = True remove\_last\_layer = False \*\*kwargs )

Parameters

-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in visual encoder model.
-   **patch\_size** (`int`, _optional_, defaults to 16) — The size (resolution) of each patch.
-   **image\_size** (`int`, _optional_, defaults to 288) — The size (resolution) of each image.
-   **initializer\_factor** (\`float“, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-05) — The epsilon used by the layer normalization layers.
-   **stop\_gradient** (`bool`, _optional_, defaults to `False`) — Whether to stop gradient for training.
-   **share\_layernorm** (`bool`, _optional_, defaults to `True`) — Whether LayerNorm layers are shared.
-   **remove\_last\_layer** (`bool`, _optional_, defaults to `False`) — Whether to remove the last layer from the vision encoder.

This is the configuration class to store the vision configuration of a [BridgeTowerModel](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerModel). Instantiating a configuration with the defaults will yield a similar configuration to that of the bridgetower-base [BridgeTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BridgeTowerVisionConfig

>>> 
>>> configuration = BridgeTowerVisionConfig()

>>> 
>>> configuration
```

## BridgeTowerImageProcessor

### class transformers.BridgeTowerImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/image_processing_bridgetower.py#L123)

( do\_resize: bool = True size: typing.Dict\[str, int\] = 288 size\_divisor: int = 32 resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None do\_center\_crop: bool = True do\_pad: bool = True \*\*kwargs )

Parameters

-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
-   **size** (`Dict[str, int]` _optional_, defaults to `288`) — Resize the shorter side of the input to `size["shortest_edge"]`. The longer side will be limited to under `int((1333 / 800) * size["shortest_edge"])` while preserving the aspect ratio. Only has an effect if `do_resize` is set to `True`. Can be overridden by the `size` parameter in the `preprocess` method.
-   **size\_divisor** (`int`, _optional_, defaults to 32) — The size by which to make sure both the height and width can be divided. Only has an effect if `do_resize` is set to `True`. Can be overridden by the `size_divisor` parameter in the `preprocess` method.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BICUBIC`) — Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be overridden by the `resample` parameter in the `preprocess` method.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be overridden by the `rescale_factor` parameter in the `preprocess` method.
-   **do\_normalize** (`bool`, _optional_, defaults to `True`) — Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_MEAN`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_STD`) — Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method. Can be overridden by the `image_std` parameter in the `preprocess` method.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `True`) — Whether to center crop the image. Can be overridden by the `do_center_crop` parameter in the `preprocess` method.
-   **do\_pad** (`bool`, _optional_, defaults to `True`) — Whether to pad the image to the `(max_height, max_width)` of the images in the batch. Can be overridden by the `do_pad` parameter in the `preprocess` method.

Constructs a BridgeTower image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/image_processing_bridgetower.py#L367)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] do\_resize: typing.Optional\[bool\] = None size: typing.Union\[typing.Dict\[str, int\], NoneType\] = None size\_divisor: typing.Optional\[int\] = None resample: Resampling = None do\_rescale: typing.Optional\[bool\] = None rescale\_factor: typing.Optional\[float\] = None do\_normalize: typing.Optional\[bool\] = None image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None do\_pad: typing.Optional\[bool\] = None do\_center\_crop: typing.Optional\[bool\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
-   **do\_resize** (`bool`, _optional_, defaults to `self.do_resize`) — Whether to resize the image.
-   **size** (`Dict[str, int]`, _optional_, defaults to `self.size`) — Controls the size of the image after `resize`. The shortest edge of the image is resized to `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
-   **size\_divisor** (`int`, _optional_, defaults to `self.size_divisor`) — The image is resized to a size that is a multiple of this value.
-   **resample** (`PILImageResampling`, _optional_, defaults to `self.resample`) — Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
-   **do\_rescale** (`bool`, _optional_, defaults to `self.do_rescale`) — Whether to rescale the image values between \[0 - 1\].
-   **rescale\_factor** (`float`, _optional_, defaults to `self.rescale_factor`) — Rescale factor to rescale the image by if `do_rescale` is set to `True`.
-   **do\_normalize** (`bool`, _optional_, defaults to `self.do_normalize`) — Whether to normalize the image.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `self.image_mean`) — Image mean to normalize the image by if `do_normalize` is set to `True`.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `self.image_std`) — Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
-   **do\_pad** (`bool`, _optional_, defaults to `self.do_pad`) — Whether to pad the image to the (max\_height, max\_width) in the batch. If `True`, a pixel mask is also created and returned.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `self.do_center_crop`) — Whether to center crop the image. If the input size is smaller than `crop_size` along any edge, the image is padded with 0’s and then center cropped.
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

## BridgeTowerProcessor

### class transformers.BridgeTowerProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/processing_bridgetower.py#L26)

( image\_processor tokenizer )

Parameters

-   **image\_processor** (`BridgeTowerImageProcessor`) — An instance of [BridgeTowerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerImageProcessor). The image processor is a required input.
-   **tokenizer** (`RobertaTokenizerFast`) — An instance of \[‘RobertaTokenizerFast\`\]. The tokenizer is a required input.

Constructs a BridgeTower processor which wraps a Roberta tokenizer and BridgeTower image processor into a single processor.

[BridgeTowerProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerProcessor) offers all the functionalities of [BridgeTowerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerImageProcessor) and [RobertaTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaTokenizerFast). See the docstring of [**call**()](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerProcessor.__call__) and `decode()` for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/processing_bridgetower.py#L48)

( images text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = None add\_special\_tokens: bool = True padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = False truncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = None max\_length: typing.Optional\[int\] = None stride: int = 0 pad\_to\_multiple\_of: typing.Optional\[int\] = None return\_token\_type\_ids: typing.Optional\[bool\] = None return\_attention\_mask: typing.Optional\[bool\] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None \*\*kwargs )

This method uses [BridgeTowerImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) method to prepare image(s) for the model, and [RobertaTokenizerFast.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) to prepare text for the model.

Please refer to the docstring of the above two methods for more information.

## BridgeTowerModel

### class transformers.BridgeTowerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/modeling_bridgetower.py#L1207)

( config )

Parameters

-   **config** ([BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare BridgeTower Model transformer outputting BridgeTowerModelOutput object without any specific head on top. This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`\_ subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/modeling_bridgetower.py#L1270)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None pixel\_mask: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None image\_embeds: typing.Optional\[torch.FloatTensor\] = None image\_token\_type\_idx: typing.Optional\[int\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None labels: typing.Optional\[torch.LongTensor\] = None ) → `transformers.models.bridgetower.modeling_bridgetower.BridgeTowerModelOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `({0})`) — Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details. [What are input IDs?](../glossary#input-ids)
-   **attention\_mask** (`torch.FloatTensor` of shape `({0})`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**. [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `({0})`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token. [What are token type IDs?](../glossary#token-type-ids)
    
-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [BridgeTowerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerImageProcessor). See [BridgeTowerImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, _optional_) — Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
    
    -   1 for pixels that are real (i.e. **not masked**),
    -   0 for pixels that are padding (i.e. **masked**). `What are attention masks? <../glossary.html#attention-mask>`\_\_
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `({0}, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, _optional_) — Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
-   **image\_token\_type\_idx** (`int`, _optional_) —
    
    -   The token type ids for images.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **output\_hidden\_states** (`bool`, _optional_) — If set to `True`, hidden states are returned as a list containing the hidden states of text, image, and cross-modal components respectively. i.e. `(hidden_states_text, hidden_states_image, hidden_states_cross_modal)` where each element is a list of the hidden states of the corresponding modality. `hidden_states_txt/img` are a list of tensors corresponding to unimodal hidden states and `hidden_states_cross_modal` is a list of tuples containing `cross_modal_text_hidden_states` and `cross_modal_image_hidden_states` of each brdige layer.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels are currently not supported.

Returns

`transformers.models.bridgetower.modeling_bridgetower.BridgeTowerModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.bridgetower.modeling_bridgetower.BridgeTowerModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig)) and inputs.

-   **text\_features** (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_size)`) — Sequence of hidden-states at the text output of the last layer of the model.
    
-   **image\_features** (`torch.FloatTensor` of shape `(batch_size, image_sequence_length, hidden_size)`) — Sequence of hidden-states at the image output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size x 2)`) — Concatenation of last layer hidden-state of the first token of the text and image sequence (classification token), respectively, after further processing through layers used for auxiliary pretraining tasks.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [BridgeTowerModel](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import BridgeTowerProcessor, BridgeTowerModel
>>> from PIL import Image
>>> import requests

>>> 
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "hello world"
>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base")
>>> model = BridgeTowerModel.from_pretrained("BridgeTower/bridgetower-base")

>>> inputs = processor(image, text, return_tensors="pt")
>>> outputs = model(**inputs)
>>> outputs.keys()
odict_keys(['text_features', 'image_features', 'pooler_output'])
```

## BridgeTowerForContrastiveLearning

### class transformers.BridgeTowerForContrastiveLearning

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/modeling_bridgetower.py#L1772)

( config )

Parameters

-   **config** ([BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BridgeTower Model with a image-text contrastive head on top computing image-text contrastive loss.

This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`\_ subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/modeling_bridgetower.py#L1786)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None pixel\_mask: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None image\_embeds: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = True return\_dict: typing.Optional\[bool\] = None return\_loss: typing.Optional\[bool\] = None ) → `transformers.models.bridgetower.modeling_bridgetower.BridgeTowerContrastiveOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `({0})`) — Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details. [What are input IDs?](../glossary#input-ids)
-   **attention\_mask** (`torch.FloatTensor` of shape `({0})`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**. [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `({0})`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token. [What are token type IDs?](../glossary#token-type-ids)
    
-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [BridgeTowerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerImageProcessor). See [BridgeTowerImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, _optional_) — Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
    
    -   1 for pixels that are real (i.e. **not masked**),
    -   0 for pixels that are padding (i.e. **masked**). `What are attention masks? <../glossary.html#attention-mask>`\_\_
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `({0}, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, _optional_) — Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
-   **image\_token\_type\_idx** (`int`, _optional_) —
    
    -   The token type ids for images.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **return\_loss** (`bool`, _optional_) — Whether or not to return the contrastive loss.

Returns

`transformers.models.bridgetower.modeling_bridgetower.BridgeTowerContrastiveOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.bridgetower.modeling_bridgetower.BridgeTowerContrastiveOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `return_loss` is `True` — Image-text contrastive loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **text\_embeds** (`torch.FloatTensor)`, _optional_, returned when model is initialized with `with_projection=True`) — The text embeddings obtained by applying the projection layer to the pooler\_output.
-   **image\_embeds** (`torch.FloatTensor)`, _optional_, returned when model is initialized with `with_projection=True`) — The image embeddings obtained by applying the projection layer to the pooler\_output.
-   **cross\_embeds** (`torch.FloatTensor)`, _optional_, returned when model is initialized with `with_projection=True`) — The text-image cross-modal embeddings obtained by applying the projection layer to the pooler\_output.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

The [BridgeTowerForContrastiveLearning](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerForContrastiveLearning) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
>>> import requests
>>> from PIL import Image
>>> import torch

>>> image_urls = [
...     "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg",
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
... ]
>>> texts = ["two dogs in a car", "two cats sleeping on a couch"]
>>> images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
>>> model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

>>> inputs = processor(images, texts, padding=True, return_tensors="pt")
>>> loss = model(**inputs, return_loss=True).loss

>>> inputs = processor(images, texts[::-1], padding=True, return_tensors="pt")
>>> loss_swapped = model(**inputs, return_loss=True).loss

>>> print("Loss", round(loss.item(), 4))
Loss 0.0019

>>> print("Loss with swapped images", round(loss_swapped.item(), 4))
Loss with swapped images 2.126
```

## BridgeTowerForMaskedLM

### class transformers.BridgeTowerForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/modeling_bridgetower.py#L1552)

( config )

Parameters

-   **config** ([BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BridgeTower Model with a language modeling head on top as done during pretraining.

This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`\_ subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/modeling_bridgetower.py#L1570)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None pixel\_mask: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None image\_embeds: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None labels: typing.Optional\[torch.LongTensor\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details. [What are input IDs?](../glossary#input-ids)
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**. [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token. [What are token type IDs?](../glossary#token-type-ids)
    
-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [BridgeTowerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerImageProcessor). See [BridgeTowerImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, _optional_) — Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
    
    -   1 for pixels that are real (i.e. **not masked**),
    -   0 for pixels that are padding (i.e. **masked**). `What are attention masks? <../glossary.html#attention-mask>`\_\_
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, _optional_) — Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
-   **image\_token\_type\_idx** (`int`, _optional_) —
    
    -   The token type ids for images.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Masked language modeling (MLM) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [BridgeTowerForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000360943.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
>>> text = "a <mask> looking out of the window"

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
>>> model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

>>> 
>>> encoding = processor(image, text, return_tensors="pt")

>>> 
>>> outputs = model(**encoding)

>>> results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())

>>> print(results)
.a cat looking out of the window.
```

## BridgeTowerForImageAndTextRetrieval

### class transformers.BridgeTowerForImageAndTextRetrieval

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/modeling_bridgetower.py#L1661)

( config )

Parameters

-   **config** ([BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BridgeTower Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the \[CLS\] token) for image-to-text matching.

This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`\_ subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bridgetower/modeling_bridgetower.py#L1672)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None pixel\_mask: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None image\_embeds: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None labels: typing.Optional\[torch.LongTensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `({0})`) — Indices of input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details. [What are input IDs?](../glossary#input-ids)
-   **attention\_mask** (`torch.FloatTensor` of shape `({0})`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**. [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `({0})`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token. [What are token type IDs?](../glossary#token-type-ids)
    
-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [BridgeTowerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerImageProcessor). See [BridgeTowerImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **pixel\_mask** (`torch.LongTensor` of shape `(batch_size, height, width)`, _optional_) — Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
    
    -   1 for pixels that are real (i.e. **not masked**),
    -   0 for pixels that are padding (i.e. **masked**). `What are attention masks? <../glossary.html#attention-mask>`\_\_
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `({0}, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`, _optional_) — Optionally, instead of passing `pixel_values`, you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `pixel_values` into patch embeddings.
-   **image\_token\_type\_idx** (`int`, _optional_) —
    
    -   The token type ids for images.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, 1)`, _optional_) — Labels for computing the image-text matching loss. 0 means the pairs don’t match and 1 means they match. The pairs with 0 will be skipped for calculation.

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([BridgeTowerConfig](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [BridgeTowerForImageAndTextRetrieval](/docs/transformers/v4.34.0/en/model_doc/bridgetower#transformers.BridgeTowerForImageAndTextRetrieval) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
>>> import requests
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

>>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
>>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

>>> 
>>> scores = dict()
>>> for text in texts:
...     
...     encoding = processor(image, text, return_tensors="pt")
...     outputs = model(**encoding)
...     scores[text] = outputs.logits[0, 1].item()
```