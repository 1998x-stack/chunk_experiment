# CLIP

## Overview

The CLIP model was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

The abstract from the paper is the following:

_State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at this https URL._

## Usage

CLIP is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image classification. CLIP uses a ViT like transformer to get visual features and a causal language model to get the text features. Both the text and visual features are then projected to a latent space with identical dimension. The dot product between the projected image and text features is then used as a similar score.

To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches, which are then linearly embedded. A \[CLS\] token is added to serve as representation of an entire image. The authors also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. The [CLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPImageProcessor) can be used to resize (or rescale) and normalize images for the model.

The [CLIPTokenizer](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizer) is used to encode the text. The [CLIPProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPProcessor) wraps [CLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPImageProcessor) and [CLIPTokenizer](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizer) into a single instance to both encode the text and prepare the images. The following example shows how to get the image-text similarity scores using [CLIPProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPProcessor) and [CLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPModel).

```
>>> from PIL import Image
>>> import requests

>>> from transformers import CLIPProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

This model was contributed by [valhalla](https://huggingface.co/valhalla). The original code can be found [here](https://github.com/openai/CLIP).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with CLIP.

-   A blog post on [How to fine-tune CLIP on 10,000 image-text pairs](https://huggingface.co/blog/fine-tune-clip-rsicd).
-   CLIP is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/contrastive-image-text).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we will review it. The resource should ideally demonstrate something new instead of duplicating an existing resource.

## CLIPConfig

### class transformers.CLIPConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/configuration_clip.py#L253)

( text\_config = None vision\_config = None projection\_dim = 512 logit\_scale\_init\_value = 2.6592 \*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [CLIPTextConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [CLIPVisionConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPVisionConfig).
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Dimentionality of text and vision projection layers.
-   **logit\_scale\_init\_value** (`float`, _optional_, defaults to 2.6592) — The inital value of the _logit\_scale_ paramter. Default is used as per the original CLIP implementation.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig) is the configuration class to store the configuration of a [CLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPModel). It is used to instantiate a CLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the CLIP [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CLIPConfig, CLIPModel

>>> 
>>> configuration = CLIPConfig()

>>> 
>>> model = CLIPModel(configuration)

>>> 
>>> configuration = model.config

>>> 
>>> from transformers import CLIPTextConfig, CLIPVisionConfig

>>> 
>>> config_text = CLIPTextConfig()
>>> config_vision = CLIPVisionConfig()

>>> config = CLIPConfig.from_text_vision_configs(config_text, config_vision)
```

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/configuration_clip.py#L389)

( text\_config: CLIPTextConfig vision\_config: CLIPVisionConfig \*\*kwargs ) → [CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig)

An instance of a configuration object

Instantiate a [CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig) (or a derived class) from clip text model configuration and clip vision model configuration.

## CLIPTextConfig

### class transformers.CLIPTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/configuration_clip.py#L39)

( vocab\_size = 49408 hidden\_size = 512 intermediate\_size = 2048 projection\_dim = 512 num\_hidden\_layers = 12 num\_attention\_heads = 8 max\_position\_embeddings = 77 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 pad\_token\_id = 1 bos\_token\_id = 49406 eos\_token\_id = 49407 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 49408) — Vocabulary size of the CLIP text model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [CLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPModel).
-   **hidden\_size** (`int`, _optional_, defaults to 512) — Dimensionality of the encoder layers and the pooler layer.
-   **intermediate\_size** (`int`, _optional_, defaults to 2048) — Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 8) — Number of attention heads for each attention layer in the Transformer encoder.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 77) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"quick_gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization layers.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **initializer\_factor** (`float`, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

This is the configuration class to store the configuration of a [CLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTextModel). It is used to instantiate a CLIP text encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the text encoder of the CLIP [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CLIPTextConfig, CLIPTextModel

>>> 
>>> configuration = CLIPTextConfig()

>>> 
>>> model = CLIPTextModel(configuration)

>>> 
>>> configuration = model.config
```

## CLIPVisionConfig

### class transformers.CLIPVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/configuration_clip.py#L148)

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
-   **initializer\_factor** (`float`, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

This is the configuration class to store the configuration of a [CLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPVisionModel). It is used to instantiate a CLIP vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the vision encoder of the CLIP [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CLIPVisionConfig, CLIPVisionModel

>>> 
>>> configuration = CLIPVisionConfig()

>>> 
>>> model = CLIPVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## CLIPTokenizer

### class transformers.CLIPTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/tokenization_clip.py#L272)

( vocab\_file merges\_file errors = 'replace' unk\_token = '<|endoftext|>' bos\_token = '<|startoftext|>' eos\_token = '<|endoftext|>' pad\_token = '<|endoftext|>' \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **errors** (`str`, _optional_, defaults to `"replace"`) — Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
-   **unk\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **bos\_token** (`str`, _optional_, defaults to `<|startoftext|>`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The end of sequence token.

Construct a CLIP tokenizer. Based on byte-level Byte-Pair-Encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/tokenization_clip.py#L357)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A CLIP sequence has the following format:

-   single sequence: `<|startoftext|> X <|endoftext|>`

Pairs of sequences are not the expected use case, but they will be handled without a separator.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/tokenization_clip.py#L384)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/tokenization_clip.py#L412)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed. CLIP does not make use of token type ids, therefore a list of zeros is returned.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/tokenization_clip.py#L507)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None )

## CLIPTokenizerFast

### class transformers.CLIPTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/tokenization_clip_fast.py#L50)

( vocab\_file = None merges\_file = None tokenizer\_file = None unk\_token = '<|endoftext|>' bos\_token = '<|startoftext|>' eos\_token = '<|endoftext|>' pad\_token = '<|endoftext|>' \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **unk\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **bos\_token** (`str`, _optional_, defaults to `<|startoftext|>`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The end of sequence token.

Construct a “fast” CLIP tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on byte-level Byte-Pair-Encoding.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/tokenization_clip_fast.py#L123)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A CLIP sequence has the following format:

-   single sequence: `<|startoftext|> X <|endoftext|>`

Pairs of sequences are not the expected use case, but they will be handled without a separator.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/tokenization_clip_fast.py#L150)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed. CLIP does not make use of token type ids, therefore a list of zeros is returned.

## CLIPImageProcessor

### class transformers.CLIPImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/image_processing_clip.py#L50)

( do\_resize: bool = True size: typing.Dict\[str, int\] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_center\_crop: bool = True crop\_size: typing.Dict\[str, int\] = None do\_rescale: bool = True rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None do\_convert\_rgb: bool = True \*\*kwargs )

Parameters

-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by `do_resize` in the `preprocess` method.
-   **size** (`Dict[str, int]` _optional_, defaults to `{"shortest_edge" -- 224}`): Size of the image after resizing. The shortest edge of the image is resized to size\[“shortest\_edge”\], with the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess` method.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BICUBIC`) — Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `True`) — Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the `preprocess` method.
-   **crop\_size** (`Dict[str, int]` _optional_, defaults to 224) — Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess` method.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess` method. do\_normalize — Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `[0.48145466, 0.4578275, 0.40821073]`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `[0.26862954, 0.26130258, 0.27577711]`) — Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method. Can be overridden by the `image_std` parameter in the `preprocess` method.
-   **do\_convert\_rgb** (`bool`, _optional_, defaults to `True`) — Whether to convert the image to RGB.

Constructs a CLIP image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/image_processing_clip.py#L164)

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

## CLIPFeatureExtractor

## CLIPProcessor

### class transformers.CLIPProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/processing_clip.py#L25)

( image\_processor = None tokenizer = None \*\*kwargs )

Parameters

-   **image\_processor** ([CLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPImageProcessor)) — The image processor is a required input.
-   **tokenizer** ([CLIPTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizerFast)) — The tokenizer is a required input.

Constructs a CLIP processor which wraps a CLIP image processor and a CLIP tokenizer into a single processor.

[CLIPProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPProcessor) offers all the functionalities of [CLIPImageProcessor](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPImageProcessor) and [CLIPTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizerFast). See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPProcessor.decode) for more information.

This method forwards all its arguments to CLIPTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to CLIPTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## CLIPModel

### class transformers.CLIPModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L967)

( config: CLIPConfig )

Parameters

-   **config** ([CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L1098)

( input\_ids: typing.Optional\[torch.LongTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None return\_loss: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clip.modeling_clip.CLIPOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **return\_loss** (`bool`, _optional_) — Whether or not to return the contrastive loss.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clip.modeling_clip.CLIPOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clip.modeling_clip.CLIPOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPConfig'>`) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
-   **logits\_per\_image:(`torch.FloatTensor`** of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text similarity scores.
-   **logits\_per\_text:(`torch.FloatTensor`** of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image similarity scores.
-   **text\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [CLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTextModel).
-   **image\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [CLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPVisionModel).
-   **text\_model\_output(`BaseModelOutputWithPooling`):** The output of the [CLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTextModel).
-   **vision\_model\_output(`BaseModelOutputWithPooling`):** The output of the [CLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPVisionModel).

The [CLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L1002)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by applying the projection layer to the pooled output of [CLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTextModel).

The [CLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L1049)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [CLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPVisionModel).

The [CLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, CLIPModel

>>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

## CLIPTextModel

### class transformers.CLIPTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L793)

( config: CLIPTextConfig )

Parameters

-   **config** ([CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The text model from CLIP without any head or projection on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L810)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPTextConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, CLIPTextModel

>>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## CLIPTextModelWithProjection

### class transformers.CLIPTextModelWithProjection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L1198)

( config: CLIPTextConfig )

Parameters

-   **config** ([CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CLIP Text Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L1219)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clip.modeling_clip.CLIPTextModelOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clip.modeling_clip.CLIPTextModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clip.modeling_clip.CLIPTextModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPTextConfig'>`) and inputs.

-   **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` _optional_ returned when model is initialized with `with_projection=True`) — The text embeddings obtained by applying the projection layer to the pooler\_output.
    
-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CLIPTextModelWithProjection](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTextModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, CLIPTextModelWithProjection

>>> model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> text_embeds = outputs.text_embeds
```

## CLIPVisionModelWithProjection

### class transformers.CLIPVisionModelWithProjection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L1279)

( config: CLIPVisionConfig )

Parameters

-   **config** ([CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CLIP Vision Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L1296)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clip.modeling_clip.CLIPVisionModelOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clip.modeling_clip.CLIPVisionModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clip.modeling_clip.CLIPVisionModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPVisionConfig'>`) and inputs.

-   **image\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` _optional_ returned when model is initialized with `with_projection=True`) — The image embeddings obtained by applying the projection layer to the pooler\_output.
    
-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CLIPVisionModelWithProjection](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPVisionModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, CLIPVisionModelWithProjection

>>> model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> image_embeds = outputs.image_embeds
```

## CLIPVisionModel

### class transformers.CLIPVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L912)

( config: CLIPVisionConfig )

Parameters

-   **config** ([CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The vision model from CLIP without any head or projection on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L925)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPVisionConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, CLIPVisionModel

>>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## TFCLIPModel

### class transformers.TFCLIPModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_tf_clip.py#L1167)

( \*args \*\*kwargs )

Parameters

-   **config** ([CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_tf_clip.py#L1257)

( input\_ids: TFModelInputType | None = None pixel\_values: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None return\_loss: Optional\[bool\] = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: bool = False ) → `transformers.models.clip.modeling_tf_clip.TFCLIPOutput` or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` \``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **pixel\_values** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **return\_loss** (`bool`, _optional_) — Whether or not to return the contrastive loss.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to \`False“) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

Returns

`transformers.models.clip.modeling_tf_clip.TFCLIPOutput` or `tuple(tf.Tensor)`

A `transformers.models.clip.modeling_tf_clip.TFCLIPOutput` or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPConfig'>`) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `return_loss` is `True`) — Contrastive loss for image-text similarity.
-   **logits\_per\_image:(`tf.Tensor`** of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text similarity scores.
-   **logits\_per\_text:(`tf.Tensor`** of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image similarity scores.
-   **text\_embeds(`tf.Tensor`** of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [TFCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPTextModel).
-   **image\_embeds(`tf.Tensor`** of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [TFCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPVisionModel).
-   **text\_model\_output(`~modeling_tf_utils.TFBaseModelOutputWithPooling`):** The output of the [TFCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPTextModel).
-   **vision\_model\_output(`~modeling_tf_utils.TFBaseModelOutputWithPooling`):** The output of the [TFCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPVisionModel).

The [TFCLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import tensorflow as tf
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFCLIPModel

>>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = tf.nn.softmax(logits_per_image, axis=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_tf_clip.py#L1175)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: bool = False ) → text\_features (`tf.Tensor` of shape `(batch_size, output_dim`)

Parameters

-   **input\_ids** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` \``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to \`False“) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

Returns

text\_features (`tf.Tensor` of shape `(batch_size, output_dim`)

The text embeddings obtained by applying the projection layer to the pooled output of [TFCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPTextModel).

The [TFCLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TFCLIPModel

>>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_tf_clip.py#L1215)

( pixel\_values: TFModelInputType | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: bool = False ) → image\_features (`tf.Tensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` \``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details. output\_attentions (`bool`, _optional_): Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to \`False“) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

Returns

image\_features (`tf.Tensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [TFCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPVisionModel).

The [TFCLIPModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFCLIPModel

>>> model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="tf")

>>> image_features = model.get_image_features(**inputs)
```

## TFCLIPTextModel

### class transformers.TFCLIPTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_tf_clip.py#L1060)

( \*args \*\*kwargs )

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_tf_clip.py#L1068)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` \``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to \`False“) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

A [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPTextConfig'>`) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`tf.Tensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
    This output is usually _not_ a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TFCLIPTextModel

>>> model = TFCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## TFCLIPVisionModel

### class transformers.TFCLIPVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_tf_clip.py#L1112)

( \*args \*\*kwargs )

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_tf_clip.py#L1121)

( pixel\_values: TFModelInputType | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or `tuple(tf.Tensor)`

Parameters

-   **pixel\_values** (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` \``Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details. output\_attentions (`bool`, _optional_): Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to \`False“) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

A [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPVisionConfig'>`) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`tf.Tensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
    This output is usually _not_ a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.TFCLIPVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFCLIPVisionModel

>>> model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="tf")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## FlaxCLIPModel

### class transformers.FlaxCLIPModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L1262)

( config: CLIPConfig input\_shape: typing.Optional\[typing.Tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True \*\*kwargs )

Parameters

-   **config** ([CLIPConfig](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L819)

( input\_ids pixel\_values attention\_mask = None position\_ids = None params: dict = None dropout\_rng: PRNGKey = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clip.modeling_flax_clip.FlaxCLIPOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **pixel\_values** (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clip.modeling_flax_clip.FlaxCLIPOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clip.modeling_flax_clip.FlaxCLIPOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPConfig'>`) and inputs.

-   **logits\_per\_image:(`jnp.ndarray`** of shape `(image_batch_size, text_batch_size)`) — The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text similarity scores.
-   **logits\_per\_text:(`jnp.ndarray`** of shape `(text_batch_size, image_batch_size)`) — The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image similarity scores.
-   **text\_embeds(`jnp.ndarray`** of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [FlaxCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.FlaxCLIPTextModel).
-   **image\_embeds(`jnp.ndarray`** of shape `(batch_size, output_dim`) — The image embeddings obtained by applying the projection layer to the pooled output of [FlaxCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.FlaxCLIPVisionModel).
-   **text\_model\_output(`FlaxBaseModelOutputWithPooling`):** The output of the [FlaxCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.FlaxCLIPTextModel).
-   **vision\_model\_output(`FlaxBaseModelOutputWithPooling`):** The output of the [FlaxCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.FlaxCLIPVisionModel).

The `FlaxCLIPPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import jax
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, FlaxCLIPModel

>>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="np", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = jax.nn.softmax(logits_per_image, axis=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L864)

( input\_ids attention\_mask = None position\_ids = None params: dict = None dropout\_rng: PRNGKey = None train = False ) → text\_features (`jnp.ndarray` of shape `(batch_size, output_dim`)

Examples:

```
>>> from transformers import AutoTokenizer, FlaxCLIPModel

>>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L931)

( pixel\_values params: dict = None dropout\_rng: PRNGKey = None train = False ) → image\_features (`jnp.ndarray` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.

Returns

image\_features (`jnp.ndarray` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [FlaxCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.FlaxCLIPVisionModel)

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, FlaxCLIPModel

>>> model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="np")

>>> image_features = model.get_image_features(**inputs)
```

## FlaxCLIPTextModel

### class transformers.FlaxCLIPTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L1011)

( config: CLIPTextConfig input\_shape = (1, 1) seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True \*\*kwargs )

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L664)

( input\_ids attention\_mask = None position\_ids = None params: dict = None dropout\_rng: PRNGKey = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPTextConfig'>`) and inputs.

-   **last\_hidden\_state** (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`jnp.ndarray` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxCLIPTextPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxCLIPTextModel

>>> model = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooler_output = outputs.pooler_output  
```

## FlaxCLIPTextModelWithProjection

### class transformers.FlaxCLIPTextModelWithProjection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L1082)

( config: CLIPTextConfig input\_shape = (1, 1) seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True \*\*kwargs )

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L664)

( input\_ids attention\_mask = None position\_ids = None params: dict = None dropout\_rng: PRNGKey = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clip.modeling_flax_clip.FlaxCLIPTextModelOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clip.modeling_flax_clip.FlaxCLIPTextModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clip.modeling_flax_clip.FlaxCLIPTextModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPTextConfig'>`) and inputs.

-   **text\_embeds** (`jnp.ndarray` of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [FlaxCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.FlaxCLIPTextModel).
    
-   **last\_hidden\_state** (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxCLIPTextPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxCLIPTextModelWithProjection

>>> model = FlaxCLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")

>>> outputs = model(**inputs)
>>> text_embeds = outputs.text_embeds
```

## FlaxCLIPVisionModel

### class transformers.FlaxCLIPVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L1136)

( config: CLIPVisionConfig input\_shape: typing.Optional\[typing.Tuple\] = None seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True \*\*kwargs )

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_flax_clip.py#L744)

( pixel\_values params: dict = None dropout\_rng: PRNGKey = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clip.configuration_clip.CLIPVisionConfig'>`) and inputs.

-   **last\_hidden\_state** (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`jnp.ndarray` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxCLIPVisionPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, FlaxCLIPVisionModel

>>> model = FlaxCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
>>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="np")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooler_output = outputs.pooler_output  
```