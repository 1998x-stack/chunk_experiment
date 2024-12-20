# X-CLIP

## Overview

The X-CLIP model was proposed in [Expanding Language-Image Pretrained Models for General Video Recognition](https://arxiv.org/abs/2208.02816) by Bolin Ni, Houwen Peng, Minghao Chen, Songyang Zhang, Gaofeng Meng, Jianlong Fu, Shiming Xiang, Haibin Ling. X-CLIP is a minimal extension of [CLIP](clip) for video. The model consists of a text encoder, a cross-frame vision encoder, a multi-frame integration Transformer, and a video-specific prompt generator.

The abstract from the paper is the following:

_Contrastive language-image pretraining has shown great success in learning visual-textual joint representation from web-scale data, demonstrating remarkable “zero-shot” generalization ability for various image tasks. However, how to effectively expand such new language-image pretraining methods to video domains is still an open problem. In this work, we present a simple yet effective approach that adapts the pretrained language-image models to video recognition directly, instead of pretraining a new model from scratch. More concretely, to capture the long-range dependencies of frames along the temporal dimension, we propose a cross-frame attention mechanism that explicitly exchanges information across frames. Such module is lightweight and can be plugged into pretrained language-image models seamlessly. Moreover, we propose a video-specific prompting scheme, which leverages video content information for generating discriminative textual prompts. Extensive experiments demonstrate that our approach is effective and can be generalized to different video recognition scenarios. In particular, under fully-supervised settings, our approach achieves a top-1 accuracy of 87.1% on Kinectics-400, while using 12 times fewer FLOPs compared with Swin-L and ViViT-H. In zero-shot experiments, our approach surpasses the current state-of-the-art methods by +7.6% and +14.9% in terms of top-1 accuracy under two popular protocols. In few-shot scenarios, our approach outperforms previous best methods by +32.1% and +23.1% when the labeled data is extremely limited._

Tips:

-   Usage of X-CLIP is identical to [CLIP](clip).

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/xclip_architecture.png) X-CLIP architecture. Taken from the [original paper.](https://arxiv.org/abs/2208.02816)

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/microsoft/VideoX/tree/master/X-CLIP).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with X-CLIP.

-   Demo notebooks for X-CLIP can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/X-CLIP).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## XCLIPProcessor

### class transformers.XCLIPProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/processing_x_clip.py#L25)

( image\_processor = None tokenizer = None \*\*kwargs )

Parameters

-   **image\_processor** ([VideoMAEImageProcessor](/docs/transformers/v4.34.0/en/model_doc/videomae#transformers.VideoMAEImageProcessor)) — The image processor is a required input.
-   **tokenizer** ([CLIPTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizerFast)) — The tokenizer is a required input.

Constructs an X-CLIP processor which wraps a VideoMAE image processor and a CLIP tokenizer into a single processor.

[XCLIPProcessor](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPProcessor) offers all the functionalities of [VideoMAEImageProcessor](/docs/transformers/v4.34.0/en/model_doc/videomae#transformers.VideoMAEImageProcessor) and [CLIPTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/clip#transformers.CLIPTokenizerFast). See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPProcessor.decode) for more information.

This method forwards all its arguments to CLIPTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to CLIPTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## XCLIPConfig

### class transformers.XCLIPConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/configuration_x_clip.py#L264)

( text\_config = None vision\_config = None projection\_dim = 512 prompt\_layers = 2 prompt\_alpha = 0.1 prompt\_hidden\_act = 'quick\_gelu' prompt\_num\_attention\_heads = 8 prompt\_attention\_dropout = 0.0 prompt\_projection\_dropout = 0.0 logit\_scale\_init\_value = 2.6592 \*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [XCLIPTextConfig](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [XCLIPVisionConfig](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPVisionConfig).
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Dimentionality of text and vision projection layers.
-   **prompt\_layers** (`int`, _optional_, defaults to 2) — Number of layers in the video specific prompt generator.
-   **prompt\_alpha** (`float`, _optional_, defaults to 0.1) — Alpha value to use in the video specific prompt generator.
-   **prompt\_hidden\_act** (`str` or `function`, _optional_, defaults to `"quick_gelu"`) — The non-linear activation function (function or string) in the video specific prompt generator. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` \``"quick_gelu"` are supported.
-   **prompt\_num\_attention\_heads** (`int`, _optional_, defaults to 8) — Number of attention heads in the cross-attention of the video specific prompt generator.
-   **prompt\_attention\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout probability for the attention layers in the video specific prompt generator.
-   **prompt\_projection\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout probability for the projection layers in the video specific prompt generator.
-   **logit\_scale\_init\_value** (`float`, _optional_, defaults to 2.6592) — The inital value of the _logit\_scale_ parameter. Default is used as per the original XCLIP implementation.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[XCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPConfig) is the configuration class to store the configuration of a [XCLIPModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPModel). It is used to instantiate X-CLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the X-CLIP [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/configuration_x_clip.py#L407)

( text\_config: XCLIPTextConfig vision\_config: XCLIPVisionConfig \*\*kwargs ) → [XCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPConfig)

An instance of a configuration object

Instantiate a [XCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPConfig) (or a derived class) from xclip text model configuration and xclip vision model configuration.

## XCLIPTextConfig

### class transformers.XCLIPTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/configuration_x_clip.py#L31)

( vocab\_size = 49408 hidden\_size = 512 intermediate\_size = 2048 num\_hidden\_layers = 12 num\_attention\_heads = 8 max\_position\_embeddings = 77 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 49408) — Vocabulary size of the X-CLIP text model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [XCLIPModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPModel).
-   **hidden\_size** (`int`, _optional_, defaults to 512) — Dimensionality of the encoder layers and the pooler layer.
-   **intermediate\_size** (`int`, _optional_, defaults to 2048) — Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 8) — Number of attention heads for each attention layer in the Transformer encoder.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 77) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"quick_gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` \``"quick_gelu"` are supported.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization layers.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **initializer\_factor** (\`float“, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

This is the configuration class to store the configuration of a [XCLIPModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPModel). It is used to instantiate an X-CLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the X-CLIP [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import XCLIPTextModel, XCLIPTextConfig

>>> 
>>> configuration = XCLIPTextConfig()

>>> 
>>> model = XCLIPTextModel(configuration)

>>> 
>>> configuration = model.config
```

## XCLIPVisionConfig

### class transformers.XCLIPVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/configuration_x_clip.py#L137)

( hidden\_size = 768 intermediate\_size = 3072 num\_hidden\_layers = 12 num\_attention\_heads = 12 mit\_hidden\_size = 512 mit\_intermediate\_size = 2048 mit\_num\_hidden\_layers = 1 mit\_num\_attention\_heads = 8 num\_channels = 3 image\_size = 224 patch\_size = 32 num\_frames = 8 hidden\_act = 'quick\_gelu' layer\_norm\_eps = 1e-05 attention\_dropout = 0.0 initializer\_range = 0.02 initializer\_factor = 1.0 drop\_path\_rate = 0.0 \*\*kwargs )

Parameters

-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **mit\_hidden\_size** (`int`, _optional_, defaults to 512) — Dimensionality of the encoder layers of the Multiframe Integration Transformer (MIT).
-   **mit\_intermediate\_size** (`int`, _optional_, defaults to 2048) — Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Multiframe Integration Transformer (MIT).
-   **mit\_num\_hidden\_layers** (`int`, _optional_, defaults to 1) — Number of hidden layers in the Multiframe Integration Transformer (MIT).
-   **mit\_num\_attention\_heads** (`int`, _optional_, defaults to 8) — Number of attention heads for each attention layer in the Multiframe Integration Transformer (MIT).
-   **image\_size** (`int`, _optional_, defaults to 224) — The size (resolution) of each image.
-   **patch\_size** (`int`, _optional_, defaults to 32) — The size (resolution) of each patch.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"quick_gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"`, `"gelu_new"` and \``"quick_gelu"` are supported.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization layers.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **initializer\_factor** (\`float“, _optional_, defaults to 1) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).
-   **drop\_path\_rate** (`float`, _optional_, defaults to 0.0) — Stochastic depth rate.

This is the configuration class to store the configuration of a [XCLIPModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPModel). It is used to instantiate an X-CLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the X-CLIP [microsoft/xclip-base-patch32](https://huggingface.co/microsoft/xclip-base-patch32) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import XCLIPVisionModel, XCLIPVisionConfig

>>> 
>>> configuration = XCLIPVisionConfig()

>>> 
>>> model = XCLIPVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## XCLIPModel

### class transformers.XCLIPModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/modeling_x_clip.py#L1292)

( config: XCLIPConfig )

Parameters

-   **config** ([XCLIPConfig](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/modeling_x_clip.py#L1501)

( input\_ids: typing.Optional\[torch.LongTensor\] = None pixel\_values: typing.Optional\[torch.FloatTensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None return\_loss: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.x_clip.modeling_x_clip.XCLIPOutput` or `tuple(torch.FloatTensor)`

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

`transformers.models.x_clip.modeling_x_clip.XCLIPOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.x_clip.modeling_x_clip.XCLIPOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.x_clip.configuration_x_clip.XCLIPConfig'>`) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `return_loss` is `True`) — Contrastive loss for video-text similarity.
-   **logits\_per\_video** (`torch.FloatTensor` of shape `(video_batch_size, text_batch_size)`) — The scaled dot product scores between `video_embeds` and `text_embeds`. This represents the video-text similarity scores.
-   **logits\_per\_text** (`torch.FloatTensor` of shape `(text_batch_size, video_batch_size)`) — The scaled dot product scores between `text_embeds` and `video_embeds`. This represents the text-video similarity scores.
-   **text\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [XCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPTextModel).
-   **video\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The video embeddings obtained by applying the projection layer to the pooled output of [XCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPVisionModel).
-   **text\_model\_output** (`BaseModelOutputWithPooling`) — The output of the [XCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPTextModel).
-   **vision\_model\_output** (`BaseModelOutputWithPooling`) — The output of the [XCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPVisionModel).
-   **mit\_output** (`BaseModelOutputWithPooling`) — The output of `XCLIPMultiframeIntegrationTransformer` (MIT for short).

The [XCLIPModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import av
>>> import torch
>>> import numpy as np

>>> from transformers import AutoProcessor, AutoModel
>>> from huggingface_hub import hf_hub_download

>>> np.random.seed(0)


>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`List[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
...     '''
...     Sample a given number of frame indices from the video.
...     Args:
...         clip_len (`int`): Total number of frames to sample.
...         frame_sample_rate (`int`): Sample every n-th frame.
...         seg_len (`int`): Maximum allowed index of sample's last frame.
...     Returns:
...         indices (`List[int]`): List of sampled frame indices
...     '''
...     converted_len = int(clip_len * frame_sample_rate)
...     end_idx = np.random.randint(converted_len, seg_len)
...     start_idx = end_idx - converted_len
...     indices = np.linspace(start_idx, end_idx, num=clip_len)
...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
...     return indices


>>> 
>>> file_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )
>>> container = av.open(file_path)

>>> 
>>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
>>> video = read_video_pyav(container, indices)

>>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
>>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

>>> inputs = processor(
...     text=["playing sports", "eating spaghetti", "go shopping"],
...     videos=list(video),
...     return_tensors="pt",
...     padding=True,
... )

>>> 
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits_per_video = outputs.logits_per_video  
>>> probs = logits_per_video.softmax(dim=1)  
>>> print(probs)
tensor([[1.9496e-04, 9.9960e-01, 2.0825e-04]])
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/modeling_x_clip.py#L1339)

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

The text embeddings obtained by applying the projection layer to the pooled output of [XCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPTextModel).

The [XCLIPModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
>>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_video\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/modeling_x_clip.py#L1386)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → video\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

video\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The video embeddings obtained by applying the projection layer to the pooled output of [XCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPVisionModel) and `XCLIPMultiframeIntegrationTransformer`.

The [XCLIPModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import av
>>> import torch
>>> import numpy as np

>>> from transformers import AutoProcessor, AutoModel
>>> from huggingface_hub import hf_hub_download

>>> np.random.seed(0)


>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`List[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
...     '''
...     Sample a given number of frame indices from the video.
...     Args:
...         clip_len (`int`): Total number of frames to sample.
...         frame_sample_rate (`int`): Sample every n-th frame.
...         seg_len (`int`): Maximum allowed index of sample's last frame.
...     Returns:
...         indices (`List[int]`): List of sampled frame indices
...     '''
...     converted_len = int(clip_len * frame_sample_rate)
...     end_idx = np.random.randint(converted_len, seg_len)
...     start_idx = end_idx - converted_len
...     indices = np.linspace(start_idx, end_idx, num=clip_len)
...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
...     return indices


>>> 
>>> file_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )
>>> container = av.open(file_path)

>>> 
>>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
>>> video = read_video_pyav(container, indices)

>>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
>>> model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

>>> inputs = processor(videos=list(video), return_tensors="pt")

>>> video_features = model.get_video_features(**inputs)
```

## XCLIPTextModel

### class transformers.XCLIPTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/modeling_x_clip.py#L833)

( config: XCLIPTextConfig )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/modeling_x_clip.py#L848)

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

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.x_clip.configuration_x_clip.XCLIPTextConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [XCLIPTextModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, XCLIPTextModel

>>> model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## XCLIPVisionModel

### class transformers.XCLIPVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/modeling_x_clip.py#L1048)

( config: XCLIPVisionConfig )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/x_clip/modeling_x_clip.py#L1061)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.x_clip.configuration_x_clip.XCLIPVisionConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [XCLIPVisionModel](/docs/transformers/v4.34.0/en/model_doc/xclip#transformers.XCLIPVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import av
>>> import torch
>>> import numpy as np

>>> from transformers import AutoProcessor, XCLIPVisionModel
>>> from huggingface_hub import hf_hub_download

>>> np.random.seed(0)


>>> def read_video_pyav(container, indices):
...     '''
...     Decode the video with PyAV decoder.
...     Args:
...         container (`av.container.input.InputContainer`): PyAV container.
...         indices (`List[int]`): List of frame indices to decode.
...     Returns:
...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
...     '''
...     frames = []
...     container.seek(0)
...     start_index = indices[0]
...     end_index = indices[-1]
...     for i, frame in enumerate(container.decode(video=0)):
...         if i > end_index:
...             break
...         if i >= start_index and i in indices:
...             frames.append(frame)
...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


>>> def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
...     '''
...     Sample a given number of frame indices from the video.
...     Args:
...         clip_len (`int`): Total number of frames to sample.
...         frame_sample_rate (`int`): Sample every n-th frame.
...         seg_len (`int`): Maximum allowed index of sample's last frame.
...     Returns:
...         indices (`List[int]`): List of sampled frame indices
...     '''
...     converted_len = int(clip_len * frame_sample_rate)
...     end_idx = np.random.randint(converted_len, seg_len)
...     start_idx = end_idx - converted_len
...     indices = np.linspace(start_idx, end_idx, num=clip_len)
...     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
...     return indices


>>> 
>>> file_path = hf_hub_download(
...     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
... )
>>> container = av.open(file_path)

>>> 
>>> indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
>>> video = read_video_pyav(container, indices)

>>> processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
>>> model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")

>>> pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values

>>> batch_size, num_frames, num_channels, height, width = pixel_values.shape
>>> pixel_values = pixel_values.reshape(-1, num_channels, height, width)

>>> outputs = model(pixel_values)
>>> last_hidden_state = outputs.last_hidden_state
```