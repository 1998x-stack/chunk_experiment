# TVLT

## Overview

The TVLT model was proposed in [TVLT: Textless Vision-Language Transformer](https://arxiv.org/abs/2209.14156) by Zineng Tang, Jaemin Cho, Yixin Nie, Mohit Bansal (the first three authors contributed equally). The Textless Vision-Language Transformer (TVLT) is a model that uses raw visual and audio inputs for vision-and-language representation learning, without using text-specific modules such as tokenization or automatic speech recognition (ASR). It can perform various audiovisual and vision-language tasks like retrieval, question answering, etc.

The abstract from the paper is the following:

_In this work, we present the Textless Vision-Language Transformer (TVLT), where homogeneous transformer blocks take raw visual and audio inputs for vision-and-language representation learning with minimal modality-specific design, and do not use text-specific modules such as tokenization or automatic speech recognition (ASR). TVLT is trained by reconstructing masked patches of continuous video frames and audio spectrograms (masked autoencoding) and contrastive modeling to align video and audio. TVLT attains performance comparable to its text-based counterpart on various multimodal tasks, such as visual question answering, image retrieval, video retrieval, and multimodal sentiment analysis, with 28x faster inference speed and only 1/3 of the parameters. Our findings suggest the possibility of learning compact and efficient visual-linguistic representations from low-level visual and audio signals without assuming the prior existence of text._

Tips:

-   TVLT is a model that takes both `pixel_values` and `audio_values` as input. One can use [TvltProcessor](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltProcessor) to prepare data for the model. This processor wraps an image processor (for the image/video modality) and an audio feature extractor (for the audio modality) into one.
-   TVLT is trained with images/videos and audios of various sizes: the authors resize and crop the input images/videos to 224 and limit the length of audio spectrogram to 2048. To make batching of videos and audios possible, the authors use a `pixel_mask` that indicates which pixels are real/padding and `audio_mask` that indicates which audio values are real/padding.
-   The design of TVLT is very similar to that of a standard Vision Transformer (ViT) and masked autoencoder (MAE) as in [ViTMAE](vitmae). The difference is that the model includes embedding layers for the audio modality.
-   The PyTorch version of this model is only available in torch 1.10 and higher.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/tvlt_architecture.png)

TVLT architecture. Taken from the [original paper]([https://arxiv.org/abs/2102.03334](https://arxiv.org/abs/2209.14156)).

The original code can be found [here](https://github.com/zinengtang/TVLT). This model was contributed by [Zineng Tang](https://huggingface.co/ZinengTang).

## TvltConfig

### class transformers.TvltConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/configuration_tvlt.py#L28)

( image\_size = 224spectrogram\_length = 2048frequency\_length = 128image\_patch\_size = \[16, 16\]audio\_patch\_size = \[16, 16\]num\_image\_channels = 3num\_audio\_channels = 1num\_frames = 8hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-06qkv\_bias = Trueuse\_mean\_pooling = Falsedecoder\_num\_attention\_heads = 16decoder\_hidden\_size = 512decoder\_num\_hidden\_layers = 8decoder\_intermediate\_size = 2048pixel\_mask\_ratio = 0.75audio\_mask\_ratio = 0.15audio\_mask\_type = 'frame-level'task\_matching = Truetask\_mae = Trueloss\_type = 'classification'\*\*kwargs )

This is the configuration class to store the configuration of a [TvltModel](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltModel). It is used to instantiate a TVLT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the TVLT [ZinengTang/tvlt-base](https://huggingface.co/ZinengTang/tvlt-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import TvltConfig, TvltModel

>>> 
>>> configuration = TvltConfig()

>>> 
>>> model = TvltModel(configuration)

>>> 
>>> configuration = model.config
```

## TvltProcessor

### class transformers.TvltProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/processing_tvlt.py#L22)

( image\_processorfeature\_extractor )

Parameters

-   **image\_processor** (`TvltImageProcessor`) — An instance of [TvltImageProcessor](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltImageProcessor). The image processor is a required input.
-   **feature\_extractor** (`TvltFeatureExtractor`) — An instance of [TvltFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltFeatureExtractor). The feature extractor is a required input.

Constructs a TVLT processor which wraps a TVLT image processor and TVLT feature extractor into a single processor.

[TvltProcessor](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltProcessor) offers all the functionalities of [TvltImageProcessor](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltImageProcessor) and [TvltFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltFeatureExtractor). See the docstring of [**call**()](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltProcessor.__call__) for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/processing_tvlt.py#L45)

( images = Noneaudio = Noneimages\_mixed = Nonesampling\_rate = Nonemask\_audio = Falsemask\_pixel = False\*args\*\*kwargs )

Forwards the `images` argument to TvltImageProcessor’s [preprocess()](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltImageProcessor.preprocess) and the `audio` argument to TvltFeatureExtractor’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltFeatureExtractor.__call__). Please refer to the docstring of the above two methods for more information.

## TvltImageProcessor

### class transformers.TvltImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/image_processing_tvlt.py#L67)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Nonepatch\_size: typing.List\[int\] = \[16, 16\]num\_frames: int = 8resample: Resampling = <Resampling.BILINEAR: 2>do\_center\_crop: bool = Truecrop\_size: typing.Dict\[str, int\] = Nonedo\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = \[0.5, 0.5, 0.5\]image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = \[0.5, 0.5, 0.5\]init\_mask\_generator = False\*\*kwargs )

Constructs a TVLT image processor.

This processor can be used to prepare either videos or images for the model by converting images to 1-frame videos.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/image_processing_tvlt.py#L253)

( videos: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: bool = Nonesize: typing.Dict\[str, int\] = Nonepatch\_size: typing.List\[int\] = Nonenum\_frames: int = Noneresample: Resampling = Nonedo\_center\_crop: bool = Nonecrop\_size: typing.Dict\[str, int\] = Nonedo\_rescale: bool = Nonerescale\_factor: float = Nonedo\_normalize: bool = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Noneis\_mixed: bool = Falsereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs ) → [BatchFeature](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.BatchFeature)

Preprocess an videos or image or batch of videos or images.

## TvltFeatureExtractor

( spectrogram\_length = 2048num\_channels = 1patch\_size = \[16, 16\]feature\_size = 128sampling\_rate = 44100hop\_length\_to\_sampling\_rate = 86n\_fft = 2048padding\_value = 0.0\*\*kwargs )

Parameters

-   **spectrogram\_length** (`Dict[str, int]` _optional_, defaults to 2048) — The time length of each audio spectrogram.
-   **num\_channels** (`int` _optional_, defaults to 1) — Number of audio channels.
-   **patch\_size** (`List[int]` _optional_, defaults to `[16, 16]`) — The patch size of audio patch embedding.
-   **feature\_size** (`int`, defaults to 128) — The frequency length of audio spectrogram.
-   **sampling\_rate** (`int`, defaults to 44100) — The sampling rate at which the audio files should be digitalized expressed in Hertz (Hz).
-   **hop\_length\_to\_sampling\_rate** (`int`, defaults to 86) — Hop length is length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients. For example, with sampling rate 44100, the hop length is 512, with 44100 / 512 = 86
-   **n\_fft** (`int`, defaults to 2048) — Size of the Fourier transform.
-   **padding\_value** (`float`, _optional_, defaults to 0.0) — Padding value used to pad the audio. Should correspond to silences.

Constructs a TVLT audio feature extractor. This feature extractor can be used to prepare audios for the model.

This feature extractor inherits from [FeatureExtractionMixin](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

( raw\_speech: typing.Union\[numpy.ndarray, typing.List\[float\], typing.List\[numpy.ndarray\], typing.List\[typing.List\[float\]\]\]return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Truesampling\_rate: typing.Optional\[int\] = Noneresample: bool = Falsemask\_audio: bool = False\*\*kwargs ) → [BatchFeature](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.BatchFeature)

Main method to prepare one or several audio(s) for the model.

## TvltModel

### class transformers.TvltModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/modeling_tvlt.py#L684)

( config )

Parameters

-   **config** ([TvltConfig](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare TVLT Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/modeling_tvlt.py#L714)

( pixel\_values: FloatTensoraudio\_values: FloatTensorpixel\_mask: typing.Optional\[torch.FloatTensor\] = Noneaudio\_mask: typing.Optional\[torch.FloatTensor\] = Nonemask\_pixel: bool = Falsemask\_audio: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.tvlt.modeling_tvlt.TvltModelOutput` or `tuple(torch.FloatTensor)`

The [TvltModel](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import TvltProcessor, TvltModel
>>> import numpy as np
>>> import torch

>>> num_frames = 8
>>> images = list(np.random.randn(num_frames, 3, 224, 224))
>>> audio = list(np.random.randn(10000))

>>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
>>> model = TvltModel.from_pretrained("ZinengTang/tvlt-base")

>>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

>>> outputs = model(**input_dict)
>>> loss = outputs.loss
```

## TvltForPreTraining

### class transformers.TvltForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/modeling_tvlt.py#L915)

( config )

Parameters

-   **config** ([TvltConfig](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The TVLT Model transformer with the decoder on top for self-supervised pre-training. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/modeling_tvlt.py#L1048)

( pixel\_values: FloatTensoraudio\_values: FloatTensorpixel\_mask: typing.Optional\[torch.FloatTensor\] = Noneaudio\_mask: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonepixel\_values\_mixed: typing.Optional\[torch.FloatTensor\] = Nonepixel\_mask\_mixed: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.tvlt.modeling_tvlt.TvltForPreTrainingOutput` or `tuple(torch.FloatTensor)`

The [TvltForPreTraining](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import TvltProcessor, TvltForPreTraining
>>> import numpy as np
>>> import torch

>>> num_frames = 8
>>> images = list(np.random.randn(num_frames, 3, 224, 224))
>>> images_mixed = list(np.random.randn(num_frames, 3, 224, 224))
>>> audio = list(np.random.randn(10000))
>>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
>>> model = TvltForPreTraining.from_pretrained("ZinengTang/tvlt-base")
>>> input_dict = processor(
...     images, audio, images_mixed, sampling_rate=44100, mask_pixel=True, mask_audio=True, return_tensors="pt"
... )

>>> outputs = model(**input_dict)
>>> loss = outputs.loss
```

## TvltForAudioVisualClassification

### class transformers.TvltForAudioVisualClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/modeling_tvlt.py#L1231)

( config )

Parameters

-   **config** ([TvltConfig](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Tvlt Model transformer with a classifier head on top (an MLP on top of the final hidden state of the \[CLS\] token) for audiovisual classification tasks, e.g. CMU-MOSEI Sentiment Analysis and Audio to Video Retrieval.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/tvlt/modeling_tvlt.py#L1249)

( pixel\_values: FloatTensoraudio\_values: FloatTensorpixel\_mask: typing.Optional\[torch.FloatTensor\] = Noneaudio\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [TvltForAudioVisualClassification](/docs/transformers/v4.34.0/en/model_doc/tvlt#transformers.TvltForAudioVisualClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import TvltProcessor, TvltForAudioVisualClassification
>>> import numpy as np
>>> import torch

>>> num_frames = 8
>>> images = list(np.random.randn(num_frames, 3, 224, 224))
>>> audio = list(np.random.randn(10000))
>>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
>>> model = TvltForAudioVisualClassification.from_pretrained("ZinengTang/tvlt-base")
>>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

>>> outputs = model(**input_dict)
>>> loss = outputs.loss
```