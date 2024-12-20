# Donut

## Overview

The Donut model was proposed in [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) by Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park. Donut consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform document understanding tasks such as document image classification, form understanding and visual question answering.

The abstract from the paper is the following:

_Understanding document images (e.g., invoices) is a core but challenging task since it requires complex functions such as reading text and a holistic understanding of the document. Current Visual Document Understanding (VDU) methods outsource the task of reading text to off-the-shelf Optical Character Recognition (OCR) engines and focus on the understanding task with the OCR outputs. Although such OCR-based approaches have shown promising performance, they suffer from 1) high computational costs for using OCR; 2) inflexibility of OCR models on languages or types of document; 3) OCR error propagation to the subsequent process. To address these issues, in this paper, we introduce a novel OCR-free VDU model named Donut, which stands for Document understanding transformer. As the first step in OCR-free VDU research, we propose a simple architecture (i.e., Transformer) with a pre-training objective (i.e., cross-entropy loss). Donut is conceptually simple yet effective. Through extensive experiments and analyses, we show a simple OCR-free VDU model, Donut, achieves state-of-the-art performances on various VDU tasks in terms of both speed and accuracy. In addition, we offer a synthetic data generator that helps the model pre-training to be flexible in various languages and domains._

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/donut_architecture.jpg) Donut high-level overview. Taken from the [original paper](https://arxiv.org/abs/2111.15664).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/clovaai/donut).

Tips:

-   The quickest way to get started with Donut is by checking the [tutorial notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut), which show how to use the model at inference time as well as fine-tuning on custom data.
-   Donut is always used within the [VisionEncoderDecoder](vision-encoder-decoder) framework.

## Inference

Donut’s `VisionEncoderDecoder` model accepts images as input and makes use of [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate) to autoregressively generate text given the input image.

The [DonutImageProcessor](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutImageProcessor) class is responsible for preprocessing the input image and \[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`\] decodes the generated target tokens to the target string. The [DonutProcessor](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutProcessor) wraps [DonutImageProcessor](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutImageProcessor) and \[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`\] into a single instance to both extract the input features and decode the predicted token ids.

-   Step-by-step Document Image Classification

```
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)
>>> 
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[1]["image"]

>>> 
>>> task_prompt = "<s_rvlcdip>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  
>>> print(processor.token2json(sequence))
{'class': 'advertisement'}
```

-   Step-by-step Document Parsing

```
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)
>>> 
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[2]["image"]

>>> 
>>> task_prompt = "<s_cord-v2>"
>>> decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  
>>> print(processor.token2json(sequence))
{'menu': {'nm': 'CINNAMON SUGAR', 'unitprice': '17,000', 'cnt': '1 x', 'price': '17,000'}, 'sub_total': {'subtotal_price': '17,000'}, 'total': {'total_price': '17,000', 'cashprice': '20,000', 'changeprice': '3,000'}}
```

-   Step-by-step Document Visual Question Answering (DocVQA)

```
>>> import re

>>> from transformers import DonutProcessor, VisionEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
>>> model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> model.to(device)
>>> 
>>> dataset = load_dataset("hf-internal-testing/example-documents", split="test")
>>> image = dataset[0]["image"]

>>> 
>>> task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
>>> question = "When is the coffee break?"
>>> prompt = task_prompt.replace("{user_input}", question)
>>> decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids

>>> pixel_values = processor(image, return_tensors="pt").pixel_values

>>> outputs = model.generate(
...     pixel_values.to(device),
...     decoder_input_ids=decoder_input_ids.to(device),
...     max_length=model.decoder.config.max_position_embeddings,
...     pad_token_id=processor.tokenizer.pad_token_id,
...     eos_token_id=processor.tokenizer.eos_token_id,
...     use_cache=True,
...     bad_words_ids=[[processor.tokenizer.unk_token_id]],
...     return_dict_in_generate=True,
... )

>>> sequence = processor.batch_decode(outputs.sequences)[0]
>>> sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
>>> sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  
>>> print(processor.token2json(sequence))
{'question': 'When is the coffee break?', 'answer': '11-14 to 11:39 a.m.'}
```

See the [model hub](https://huggingface.co/models?filter=donut) to look for Donut checkpoints.

## Training

We refer to the [tutorial notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Donut).

## DonutSwinConfig

### class transformers.DonutSwinConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/donut/configuration_donut_swin.py#L29)

( image\_size = 224 patch\_size = 4 num\_channels = 3 embed\_dim = 96 depths = \[2, 2, 6, 2\] num\_heads = \[3, 6, 12, 24\] window\_size = 7 mlp\_ratio = 4.0 qkv\_bias = True hidden\_dropout\_prob = 0.0 attention\_probs\_dropout\_prob = 0.0 drop\_path\_rate = 0.1 hidden\_act = 'gelu' use\_absolute\_embeddings = False initializer\_range = 0.02 layer\_norm\_eps = 1e-05 \*\*kwargs )

Parameters

-   **image\_size** (`int`, _optional_, defaults to 224) — The size (resolution) of each image.
-   **patch\_size** (`int`, _optional_, defaults to 4) — The size (resolution) of each patch.
-   **num\_channels** (`int`, _optional_, defaults to 3) — The number of input channels.
-   **embed\_dim** (`int`, _optional_, defaults to 96) — Dimensionality of patch embedding.
-   **depths** (`list(int)`, _optional_, defaults to \[2, 2, 6, 2\]) — Depth of each layer in the Transformer encoder.
-   **num\_heads** (`list(int)`, _optional_, defaults to \[3, 6, 12, 24\]) — Number of attention heads in each layer of the Transformer encoder.
-   **window\_size** (`int`, _optional_, defaults to 7) — Size of windows.
-   **mlp\_ratio** (`float`, _optional_, defaults to 4.0) — Ratio of MLP hidden dimensionality to embedding dimensionality.
-   **qkv\_bias** (`bool`, _optional_, defaults to True) — Whether or not a learnable bias should be added to the queries, keys and values.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.0) — The dropout probability for all fully connected layers in the embeddings and encoder.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **drop\_path\_rate** (`float`, _optional_, defaults to 0.1) — Stochastic depth rate.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
-   **use\_absolute\_embeddings** (`bool`, _optional_, defaults to False) — Whether or not to add absolute position embeddings to the patch embeddings.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.

This is the configuration class to store the configuration of a [DonutSwinModel](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutSwinModel). It is used to instantiate a Donut model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Donut [naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import DonutSwinConfig, DonutSwinModel

>>> 
>>> configuration = DonutSwinConfig()

>>> 
>>> model = DonutSwinModel(configuration)

>>> 
>>> configuration = model.config
```

## DonutImageProcessor

### class transformers.DonutImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/donut/image_processing_donut.py#L52)

( do\_resize: bool = True size: typing.Dict\[str, int\] = None resample: Resampling = <Resampling.BILINEAR: 2> do\_thumbnail: bool = True do\_align\_long\_axis: bool = False do\_pad: bool = True do\_rescale: bool = True rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None \*\*kwargs )

Parameters

-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s (height, width) dimensions to the specified `size`. Can be overridden by `do_resize` in the `preprocess` method.
-   **size** (`Dict[str, int]` _optional_, defaults to `{"shortest_edge" -- 224}`): Size of the image after resizing. The shortest edge of the image is resized to size\[“shortest\_edge”\], with the longest edge resized to keep the input aspect ratio. Can be overridden by `size` in the `preprocess` method.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BILINEAR`) — Resampling filter to use if resizing the image. Can be overridden by `resample` in the `preprocess` method.
-   **do\_thumbnail** (`bool`, _optional_, defaults to `True`) — Whether to resize the image using thumbnail method.
-   **do\_align\_long\_axis** (`bool`, _optional_, defaults to `False`) — Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
-   **do\_pad** (`bool`, _optional_, defaults to `True`) — Whether to pad the image. If `random_padding` is set to `True` in `preprocess`, each image is padded with a random amont of padding on each size, up to the largest image size in the batch. Otherwise, all images are padded to the largest image size in the batch.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by `do_rescale` in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Scale factor to use if rescaling the image. Can be overridden by `rescale_factor` in the `preprocess` method. do\_normalize — Whether to normalize the image. Can be overridden by `do_normalize` in the `preprocess` method.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_MEAN`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_STD`) — Image standard deviation.

Constructs a Donut image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/donut/image_processing_donut.py#L297)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] do\_resize: bool = None size: typing.Dict\[str, int\] = None resample: Resampling = None do\_thumbnail: bool = None do\_align\_long\_axis: bool = None do\_pad: bool = None random\_padding: bool = False do\_rescale: bool = None rescale\_factor: float = None do\_normalize: bool = None image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None data\_format: typing.Optional\[transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
-   **do\_resize** (`bool`, _optional_, defaults to `self.do_resize`) — Whether to resize the image.
-   **size** (`Dict[str, int]`, _optional_, defaults to `self.size`) — Size of the image after resizing. Shortest edge of the image is resized to min(size\[“height”\], size\[“width”\]) with the longest edge resized to keep the input aspect ratio.
-   **resample** (`int`, _optional_, defaults to `self.resample`) — Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.
-   **do\_thumbnail** (`bool`, _optional_, defaults to `self.do_thumbnail`) — Whether to resize the image using thumbnail method.
-   **do\_align\_long\_axis** (`bool`, _optional_, defaults to `self.do_align_long_axis`) — Whether to align the long axis of the image with the long axis of `size` by rotating by 90 degrees.
-   **do\_pad** (`bool`, _optional_, defaults to `self.do_pad`) — Whether to pad the image. If `random_padding` is set to `True`, each image is padded with a random amont of padding on each size, up to the largest image size in the batch. Otherwise, all images are padded to the largest image size in the batch.
-   **random\_padding** (`bool`, _optional_, defaults to `self.random_padding`) — Whether to use random padding when padding the image. If `True`, each image in the batch with be padded with a random amount of padding on each side up to the size of the largest image in the batch.
-   **do\_rescale** (`bool`, _optional_, defaults to `self.do_rescale`) — Whether to rescale the image pixel values.
-   **rescale\_factor** (`float`, _optional_, defaults to `self.rescale_factor`) — Rescale factor to rescale the image by if `do_rescale` is set to `True`.
-   **do\_normalize** (`bool`, _optional_, defaults to `self.do_normalize`) — Whether to normalize the image.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `self.image_mean`) — Image mean to use for normalization.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `self.image_std`) — Image standard deviation to use for normalization.
-   **return\_tensors** (`str` or `TensorType`, _optional_) — The type of tensors to return. Can be one of:
    
    -   Unset: Return a list of `np.ndarray`.
    -   `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
    -   `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
    -   `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
    -   `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
    
-   **data\_format** (`ChannelDimension` or `str`, _optional_, defaults to `ChannelDimension.FIRST`) — The channel dimension format for the output image. Can be one of:
    
    -   `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   Unset: defaults to the channel dimension format of the input image.
    
-   **input\_data\_format** (`ChannelDimension` or `str`, _optional_) — The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    

Preprocess an image or batch of images.

## DonutFeatureExtractor

Preprocess an image or a batch of images.

## DonutProcessor

### class transformers.DonutProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/donut/processing_donut.py#L25)

( image\_processor = None tokenizer = None \*\*kwargs )

Parameters

-   **image\_processor** ([DonutImageProcessor](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutImageProcessor)) — An instance of [DonutImageProcessor](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutImageProcessor). The image processor is a required input.
-   **tokenizer** (\[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`\]) — An instance of \[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`\]. The tokenizer is a required input.

Constructs a Donut processor which wraps a Donut image processor and an XLMRoBERTa tokenizer into a single processor.

[DonutProcessor](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutProcessor) offers all the functionalities of [DonutImageProcessor](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutImageProcessor) and \[`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`\]. See the [**call**()](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutProcessor.__call__) and [decode()](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutProcessor.decode) for more information.

When used in normal mode, this method forwards all its arguments to AutoImageProcessor’s `__call__()` and returns its output. If used in the context `as_target_processor()` this method forwards all its arguments to DonutTokenizer’s `~DonutTokenizer.__call__`. Please refer to the doctsring of the above two methods for more information.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/processing_utils.py#L167)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\] cache\_dir: typing.Union\[str, os.PathLike, NoneType\] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union\[bool, str, NoneType\] = None revision: str = 'main' \*\*kwargs )

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — This can be either:
    
    -   a string, the _model id_ of a pretrained feature\_extractor hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
    -   a path to a _directory_ containing a feature extractor file saved using the [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`.
    -   a path or url to a saved feature extractor JSON _file_, e.g., `./my_model_directory/preprocessor_config.json`. \*\*kwargs — Additional keyword arguments passed along to both [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
    

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the methods above for more information.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/processing_utils.py#L93)

( save\_directory push\_to\_hub: bool = False \*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it can be reloaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/nougat#transformers.NougatProcessor.from_pretrained) method.

This class method is simply calling [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and [save\_pretrained()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the methods above for more information.

This method forwards all its arguments to DonutTokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to DonutTokenizer’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## DonutSwinModel

### class transformers.DonutSwinModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/donut/modeling_donut_swin.py#L871)

( config add\_pooling\_layer = True use\_mask\_token = False )

Parameters

-   **config** ([DonutSwinConfig](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutSwinConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Donut Swin Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/donut/modeling_donut_swin.py#L897)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None bool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [DonutImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **bool\_masked\_pos** (`torch.BoolTensor` of shape `(batch_size, num_patches)`) — Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).

Returns

`transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.donut.modeling_donut_swin.DonutSwinModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([DonutSwinConfig](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutSwinConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, _optional_, returned when `add_pooling_layer=True` is passed) — Average pooling of the last layer hidden-state.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **reshaped\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of shape `(batch_size, hidden_size, height, width)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to include the spatial dimensions.
    

The [DonutSwinModel](/docs/transformers/v4.34.0/en/model_doc/donut#transformers.DonutSwinModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, DonutSwinModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("https://huggingface.co/naver-clova-ix/donut-base")
>>> model = DonutSwinModel.from_pretrained("https://huggingface.co/naver-clova-ix/donut-base")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 49, 768]
```