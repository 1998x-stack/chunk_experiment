# CLAP

## Overview

The CLAP model was proposed in [Large Scale Contrastive Language-Audio pretraining with feature fusion and keyword-to-caption augmentation](https://arxiv.org/pdf/2211.06687.pdf) by Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, Shlomo Dubnov.

CLAP (Contrastive Language-Audio Pretraining) is a neural network trained on a variety of (audio, text) pairs. It can be instructed in to predict the most relevant text snippet, given an audio, without directly optimizing for the task. The CLAP model uses a SWINTransformer to get audio features from a log-Mel spectrogram input, and a RoBERTa model to get text features. Both the text and audio features are then projected to a latent space with identical dimension. The dot product between the projected audio and text features is then used as a similar score.

The abstract from the paper is the following:

_Contrastive learning has shown remarkable success in the field of multimodal representation learning. In this paper, we propose a pipeline of contrastive language-audio pretraining to develop an audio representation by combining audio data with natural language descriptions. To accomplish this target, we first release LAION-Audio-630K, a large collection of 633,526 audio-text pairs from different data sources. Second, we construct a contrastive language-audio pretraining model by considering different audio encoders and text encoders. We incorporate the feature fusion mechanism and keyword-to-caption augmentation into the model design to further enable the model to process audio inputs of variable lengths and enhance the performance. Third, we perform comprehensive experiments to evaluate our model across three tasks: text-to-audio retrieval, zero-shot audio classification, and supervised audio classification. The results demonstrate that our model achieves superior performance in text-to-audio retrieval task. In audio classification tasks, the model achieves state-of-the-art performance in the zeroshot setting and is able to obtain performance comparable to models’ results in the non-zero-shot setting. LAION-Audio-6_

This model was contributed by [Younes Belkada](https://huggingface.co/ybelkada) and [Arthur Zucker](https://huggingface.co/ArtZucker) . The original code can be found [here](https://github.com/LAION-AI/Clap).

## ClapConfig

### class transformers.ClapConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/configuration_clap.py#L333)

( text\_config = None audio\_config = None logit\_scale\_init\_value = 14.285714285714285 projection\_dim = 512 projection\_hidden\_act = 'relu' initializer\_factor = 1.0 \*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [ClapTextConfig](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapTextConfig).
-   **audio\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [ClapAudioConfig](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapAudioConfig).
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Dimentionality of text and audio projection layers.
-   **logit\_scale\_init\_value** (`float`, _optional_, defaults to 2.6592) — The inital value of the _logit\_scale_ paramter. Default is used as per the original CLAP implementation.
-   **projection\_hidden\_act** (`str`, _optional_, defaults to `"relu"`) — Activation function for the projection layers.
-   **initializer\_factor** (`float`, _optional_, defaults to 1.0) — Factor to scale the initialization of the model weights.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[ClapConfig](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapConfig) is the configuration class to store the configuration of a [ClapModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapModel). It is used to instantiate a CLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the CLAP [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ClapConfig, ClapModel

>>> 
>>> configuration = ClapConfig()

>>> 
>>> model = ClapModel(configuration)

>>> 
>>> configuration = model.config

>>> 
>>> from transformers import ClapTextConfig, ClapAudioConfig

>>> 
>>> config_text = ClapTextConfig()
>>> config_audio = ClapAudioConfig()

>>> config = ClapConfig.from_text_audio_configs(config_text, config_audio)
```

#### from\_text\_audio\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/configuration_clap.py#L421)

( text\_config: ClapTextConfig audio\_config: ClapAudioConfig \*\*kwargs ) → [ClapConfig](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapConfig)

An instance of a configuration object

Instantiate a [ClapConfig](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapConfig) (or a derived class) from clap text model configuration and clap audio model configuration.

## ClapTextConfig

### class transformers.ClapTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/configuration_clap.py#L32)

( vocab\_size = 50265 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 514 type\_vocab\_size = 1 initializer\_factor = 1.0 layer\_norm\_eps = 1e-12 projection\_dim = 512 pad\_token\_id = 1 bos\_token\_id = 0 eos\_token\_id = 2 position\_embedding\_type = 'absolute' use\_cache = True projection\_hidden\_act = 'relu' \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 30522) — Vocabulary size of the CLAP model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [ClapTextModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapTextModel).
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
-   **hidden\_act** (`str` or `Callable`, _optional_, defaults to `"relu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"relu"`, `"relu"`, `"silu"` and `"relu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 512) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **type\_vocab\_size** (`int`, _optional_, defaults to 2) — The vocabulary size of the `token_type_ids` passed when calling [ClapTextModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapTextModel).
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **position\_embedding\_type** (`str`, _optional_, defaults to `"absolute"`) — Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155). For more information on `"relative_key_query"`, please refer to _Method 4_ in [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
-   **is\_decoder** (`bool`, _optional_, defaults to `False`) — Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.
-   **projection\_hidden\_act** (`str`, _optional_, defaults to `"relu"`) — The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Dimension of the projection head of the `ClapTextModelWithProjection`.

This is the configuration class to store the configuration of a [ClapTextModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapTextModel). It is used to instantiate a CLAP model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the CLAP [calp-hsat-fused](https://huggingface.co/laion/clap-hsat-fused) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import ClapTextConfig, ClapTextModel

>>> 
>>> configuration = ClapTextConfig()

>>> 
>>> model = ClapTextModel(configuration)

>>> 
>>> configuration = model.config
```

## ClapAudioConfig

### class transformers.ClapAudioConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/configuration_clap.py#L163)

( window\_size = 8 num\_mel\_bins = 64 spec\_size = 256 hidden\_act = 'gelu' patch\_size = 4 patch\_stride = \[4, 4\] num\_classes = 527 hidden\_size = 768 projection\_dim = 512 depths = \[2, 2, 6, 2\] num\_attention\_heads = \[4, 8, 16, 32\] enable\_fusion = False hidden\_dropout\_prob = 0.1 fusion\_type = None patch\_embed\_input\_channels = 1 flatten\_patch\_embeds = True patch\_embeds\_hidden\_size = 96 enable\_patch\_layer\_norm = True drop\_path\_rate = 0.0 attention\_probs\_dropout\_prob = 0.0 qkv\_bias = True mlp\_ratio = 4.0 aff\_block\_r = 4 num\_hidden\_layers = 4 projection\_hidden\_act = 'relu' layer\_norm\_eps = 1e-05 initializer\_factor = 1.0 \*\*kwargs )

Parameters

-   **window\_size** (`int`, _optional_, defaults to 8) — Image size of the spectrogram
-   **num\_mel\_bins** (`int`, _optional_, defaults to 64) — Number of mel features used per frames. Should correspond to the value used in the `ClapProcessor` class.
-   **spec\_size** (`int`, _optional_, defaults to 256) — Desired input size of the spectrogram that the model supports. It can be different from the output of the `ClapFeatureExtractor`, in which case the input features will be resized. Corresponds to the `image_size` of the audio models.
-   **hidden\_act** (`str`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **patch\_size** (`int`, _optional_, defaults to 4) — Patch size for the audio spectrogram
-   **patch\_stride** (`list`, _optional_, defaults to `[4, 4]`) — Patch stride for the audio spectrogram
-   **num\_classes** (`int`, _optional_, defaults to 527) — Number of classes used for the head training
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Hidden size of the output of the audio encoder. Correspond to the dimension of the penultimate layer’s output,which is sent to the projection MLP layer.
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Hidden size of the projection layer.
-   **depths** (`list`, _optional_, defaults to `[2, 2, 6, 2]`) — Depths used for the Swin Layers of the audio model
-   **num\_attention\_heads** (`list`, _optional_, defaults to `[4, 8, 16, 32]`) — Number of attention heads used for the Swin Layers of the audio model
-   **enable\_fusion** (`bool`, _optional_, defaults to `False`) — Whether or not to enable patch fusion. This is the main contribution of the authors, and should give the best results.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probabilitiy for all fully connected layers in the encoder.
-   **fusion\_type** (`[type]`, _optional_) — Fusion type used for the patch fusion.
-   **patch\_embed\_input\_channels** (`int`, _optional_, defaults to 1) — Number of channels used for the input spectrogram
-   **flatten\_patch\_embeds** (`bool`, _optional_, defaults to `True`) — Whether or not to flatten the patch embeddings
-   **patch\_embeds\_hidden\_size** (`int`, _optional_, defaults to 96) — Hidden size of the patch embeddings. It is used as the number of output channels.
-   **enable\_patch\_layer\_norm** (`bool`, _optional_, defaults to `True`) — Whether or not to enable layer normalization for the patch embeddings
-   **drop\_path\_rate** (`float`, _optional_, defaults to 0.0) — Drop path rate for the patch fusion
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **qkv\_bias** (`bool`, _optional_, defaults to `True`) — Whether or not to add a bias to the query, key, value projections.
-   **mlp\_ratio** (`float`, _optional_, defaults to 4.0) — Ratio of the mlp hidden dim to embedding dim.
-   **aff\_block\_r** (`int`, _optional_, defaults to 4) — downsize\_ratio used in the AudioFF block
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 4) — Number of hidden layers in the Transformer encoder.
-   **projection\_hidden\_act** (`str`, _optional_, defaults to `"relu"`) — The non-linear activation function (function or string) in the projection layer. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **layer\_norm\_eps** (`[type]`, _optional_, defaults to `1e-5`) — The epsilon used by the layer normalization layers.
-   **initializer\_factor** (`float`, _optional_, defaults to 1.0) — A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

This is the configuration class to store the configuration of a [ClapAudioModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapAudioModel). It is used to instantiate a CLAP audio encoder according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the audio encoder of the CLAP [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ClapAudioConfig, ClapAudioModel

>>> 
>>> configuration = ClapAudioConfig()

>>> 
>>> model = ClapAudioModel(configuration)

>>> 
>>> configuration = model.config
```

## ClapFeatureExtractor

( feature\_size = 64 sampling\_rate = 48000 hop\_length = 480 max\_length\_s = 10 fft\_window\_size = 1024 padding\_value = 0.0 return\_attention\_mask = False frequency\_min: float = 0 frequency\_max: float = 14000 top\_db: int = None truncation: str = 'fusion' padding: str = 'repeatpad' \*\*kwargs )

Parameters

-   **feature\_size** (`int`, defaults to 64) — The feature dimension of the extracted Mel spectrograms. This corresponds to the number of mel filters (`n_mels`).
-   **sampling\_rate** (`int`, defaults to 48\_000) — The sampling rate at which the audio files should be digitalized expressed in hertz (Hz). This only serves to warn users if the audio fed to the feature extractor does not have the same sampling rate.
-   **hop\_length** (`int`, defaults to 480) — Length of the overlaping windows for the STFT used to obtain the Mel Spectrogram. The audio will be split in smaller `frames` with a step of `hop_length` between each frame.
-   **max\_length\_s** (`int`, defaults to 10) — The maximum input length of the model in seconds. This is used to pad the audio.
-   **fft\_window\_size** (`int`, defaults to 1024) — Size of the window (in samples) on which the Fourier transform is applied. This controls the frequency resolution of the spectrogram. 400 means that the fourrier transform is computed on windows of 400 samples.
-   **padding\_value** (`float`, _optional_, defaults to 0.0) — Padding value used to pad the audio. Should correspond to silences.
-   **return\_attention\_mask** (`bool`, _optional_, defaults to `False`) — Whether or not the model should return the attention masks coresponding to the input.
-   **frequency\_min** (`float`, _optional_, default to 0) — The lowest frequency of interest. The STFT will not be computed for values below this.
-   **frequency\_max** (`float`, _optional_, default to 14\_000) — The highest frequency of interest. The STFT will not be computed for values above this.
-   **top\_db** (`float`, _optional_) — The highest decibel value used to convert the mel spectrogram to the log scale. For more details see the `audio_utils.power_to_db` function
-   **truncation** (`str`, _optional_, default to `"fusions"`) — Truncation pattern for long audio inputs. Two patterns are available:
    
    -   `fusion` will use `_random_mel_fusion`, which stacks 3 random crops from the mel spectrogram and a downsampled version of the entire mel spectrogram. If `config.fusion` is set to True, shorter audios also need to to return 4 mels, which will just be a copy of the original mel obtained from the padded audio.
    -   `rand_trunc` will select a random crop of the mel spectrogram.
    
-   **padding** (`str`, _optional_, defaults to `"repeatpad"`) — Padding pattern for shorter audio inputs. Three patterns were originally implemented:
    
    -   `repeatpad`: the audio is repeated, and then padded to fit the `max_length`.
    -   `repeat`: the audio is repeated and then cut to fit the `max_length`
    -   `pad`: the audio is padded.
    

Constructs a CLAP feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the _Short Time Fourier Transform_ (STFT) which should match pytorch’s `torch.stft` equivalent.

( ) → `Dict[str, Any]`

Dictionary of all the attributes that make up this configuration instance, excpet for the mel filter banks, which do not need to be saved or printed as they are too long.

Serializes this instance to a Python dictionary.

## ClapProcessor

Constructs a CLAP processor which wraps a CLAP feature extractor and a RoBerta tokenizer into a single processor.

[ClapProcessor](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapProcessor) offers all the functionalities of [ClapFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapFeatureExtractor) and [RobertaTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaTokenizerFast). See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapProcessor.decode) for more information.

This method forwards all its arguments to RobertaTokenizerFast’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to RobertaTokenizerFast’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## ClapModel

### class transformers.ClapModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L1937)

( config: ClapConfig )

Parameters

-   **config** ([ClapConfig](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L2066)

( input\_ids: typing.Optional\[torch.LongTensor\] = None input\_features: typing.Optional\[torch.FloatTensor\] = None is\_longer: typing.Optional\[torch.BoolTensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None return\_loss: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clap.modeling_clap.ClapOutput` or `tuple(torch.FloatTensor)`

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
    
-   **input\_features** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Input audio features. This should be returnes by the [ClapFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapFeatureExtractor) class that you can also retrieve from [AutoFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoFeatureExtractor). See `ClapFeatureExtractor.__call__()` for details.
-   **return\_loss** (`bool`, _optional_) — Whether or not to return the contrastive loss.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clap.modeling_clap.ClapOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clap.modeling_clap.ClapOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clap.configuration_clap.ClapConfig'>`) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `return_loss` is `True`) — Contrastive loss for audio-text similarity.
-   **logits\_per\_audio:(`torch.FloatTensor`** of shape `(audio_batch_size, text_batch_size)`) — The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text similarity scores.
-   **logits\_per\_text:(`torch.FloatTensor`** of shape `(text_batch_size, audio_batch_size)`) — The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio similarity scores.
-   **text\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The text embeddings obtained by applying the projection layer to the pooled output of [ClapTextModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapTextModel).
-   **audio\_embeds(`torch.FloatTensor`** of shape `(batch_size, output_dim`) — The audio embeddings obtained by applying the projection layer to the pooled output of [ClapAudioModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapAudioModel).
-   **text\_model\_output(`BaseModelOutputWithPooling`):** The output of the [ClapTextModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapTextModel).
-   **audio\_model\_output(`BaseModelOutputWithPooling`):** The output of the [ClapAudioModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapAudioModel).

The [ClapModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from datasets import load_dataset
>>> from transformers import AutoProcessor, ClapModel

>>> dataset = load_dataset("ashraq/esc50")
>>> audio_sample = dataset["train"]["audio"][0]["array"]

>>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
>>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

>>> input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

>>> inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)
>>> logits_per_audio = outputs.logits_per_audio  
>>> probs = logits_per_audio.softmax(dim=-1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L1972)

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

The text embeddings obtained by applying the projection layer to the pooled output of [ClapTextModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapTextModel).

The [ClapModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, ClapModel

>>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
>>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

>>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_audio\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L2020)

( input\_features: typing.Optional\[torch.Tensor\] = None is\_longer: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → audio\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **input\_features** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Input audio features. This should be returnes by the [ClapFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapFeatureExtractor) class that you can also retrieve from [AutoFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoFeatureExtractor). See `ClapFeatureExtractor.__call__()` for details.
-   **is\_longer** (`torch.FloatTensor`, of shape `(batch_size, 1)`, _optional_) — Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance the features.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

audio\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The audio embeddings obtained by applying the projection layer to the pooled output of [ClapAudioModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapAudioModel).

The [ClapModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoFeatureExtractor, ClapModel
>>> import torch

>>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
>>> random_audio = torch.rand((16_000))
>>> inputs = feature_extractor(random_audio, return_tensors="pt")
>>> audio_features = model.get_audio_features(**inputs)
```

## ClapTextModel

### class transformers.ClapTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L1767)

( config add\_pooling\_layer = True )

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in _Attention is all you need_\_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

.. \__Attention is all you need_: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L1805)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None token\_type\_ids: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None encoder\_hidden\_states: typing.Optional\[torch.Tensor\] = None encoder\_attention\_mask: typing.Optional\[torch.Tensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None )

encoder\_hidden\_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_): Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder. encoder\_attention\_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_): Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

-   1 for tokens that are **not masked**,
-   0 for tokens that are **masked**. past\_key\_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`. use\_cache (`bool`, _optional_): If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

## ClapTextModelWithProjection

### class transformers.ClapTextModelWithProjection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L2170)

( config: ClapTextConfig )

Parameters

-   **config** ([ClapConfig](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CLAP Text Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L2186)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clap.modeling_clap.ClapTextModelOutput` or `tuple(torch.FloatTensor)`

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

`transformers.models.clap.modeling_clap.ClapTextModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clap.modeling_clap.ClapTextModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clap.configuration_clap.ClapTextConfig'>`) and inputs.

-   **text\_embeds** (`torch.FloatTensor` of shape `(batch_size, output_dim)` _optional_ returned when model is initialized with `with_projection=True`) — The text embeddings obtained by applying the projection layer to the pooler\_output.
    
-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [ClapTextModelWithProjection](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapTextModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, ClapTextModelWithProjection

>>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
>>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

>>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> text_embeds = outputs.text_embeds
```

## ClapAudioModel

### class transformers.ClapAudioModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L1709)

( config: ClapAudioConfig )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L1722)

( input\_features: typing.Optional\[torch.FloatTensor\] = None is\_longer: typing.Optional\[torch.BoolTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_features** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Input audio features. This should be returnes by the [ClapFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapFeatureExtractor) class that you can also retrieve from [AutoFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoFeatureExtractor). See `ClapFeatureExtractor.__call__()` for details.
-   **is\_longer** (`torch.FloatTensor`, of shape `(batch_size, 1)`, _optional_) — Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance the features.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clap.configuration_clap.ClapAudioConfig'>`) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [ClapAudioModel](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapAudioModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from datasets import load_dataset
>>> from transformers import AutoProcessor, ClapAudioModel

>>> dataset = load_dataset("ashraq/esc50")
>>> audio_sample = dataset["train"]["audio"][0]["array"]

>>> model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
>>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

>>> inputs = processor(audios=audio_sample, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
```

## ClapAudioModelWithProjection

### class transformers.ClapAudioModelWithProjection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L2246)

( config: ClapAudioConfig )

Parameters

-   **config** ([ClapConfig](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

CLAP Audio Model with a projection layer on top (a linear layer on top of the pooled output).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clap/modeling_clap.py#L2260)

( input\_features: typing.Optional\[torch.FloatTensor\] = None is\_longer: typing.Optional\[torch.BoolTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clap.modeling_clap.ClapAudioModelOutput` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_features** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Input audio features. This should be returnes by the [ClapFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapFeatureExtractor) class that you can also retrieve from [AutoFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoFeatureExtractor). See `ClapFeatureExtractor.__call__()` for details.
-   **is\_longer** (`torch.FloatTensor`, of shape `(batch_size, 1)`, _optional_) — Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance the features.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

`transformers.models.clap.modeling_clap.ClapAudioModelOutput` or `tuple(torch.FloatTensor)`

A `transformers.models.clap.modeling_clap.ClapAudioModelOutput` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration (`<class 'transformers.models.clap.configuration_clap.ClapAudioConfig'>`) and inputs.

-   **audio\_embeds** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — The Audio embeddings obtained by applying the projection layer to the pooler\_output.
    
-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    

The [ClapAudioModelWithProjection](/docs/transformers/v4.34.0/en/model_doc/clap#transformers.ClapAudioModelWithProjection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from datasets import load_dataset
>>> from transformers import ClapAudioModelWithProjection, ClapProcessor

>>> model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
>>> processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

>>> dataset = load_dataset("ashraq/esc50")
>>> audio_sample = dataset["train"]["audio"][0]["array"]

>>> inputs = processor(audios=audio_sample, return_tensors="pt")
>>> outputs = model(**inputs)
>>> audio_embeds = outputs.audio_embeds
```