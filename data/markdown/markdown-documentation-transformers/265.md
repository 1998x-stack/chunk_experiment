# SpeechT5

## Overview

The SpeechT5 model was proposed in [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205) by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.

The abstract from the paper is the following:

_Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder. Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder. Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification._

This model was contributed by [Matthijs](https://huggingface.co/Matthijs). The original code can be found [here](https://github.com/microsoft/SpeechT5).

## SpeechT5Config

### class transformers.SpeechT5Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/configuration_speecht5.py#L37)

( vocab\_size = 81hidden\_size = 768encoder\_layers = 12encoder\_attention\_heads = 12encoder\_ffn\_dim = 3072encoder\_layerdrop = 0.1decoder\_layers = 6decoder\_ffn\_dim = 3072decoder\_attention\_heads = 12decoder\_layerdrop = 0.1hidden\_act = 'gelu'positional\_dropout = 0.1hidden\_dropout = 0.1attention\_dropout = 0.1activation\_dropout = 0.1initializer\_range = 0.02layer\_norm\_eps = 1e-05scale\_embedding = Falsefeat\_extract\_norm = 'group'feat\_proj\_dropout = 0.0feat\_extract\_activation = 'gelu'conv\_dim = (512, 512, 512, 512, 512, 512, 512)conv\_stride = (5, 2, 2, 2, 2, 2, 2)conv\_kernel = (10, 3, 3, 3, 3, 2, 2)conv\_bias = Falsenum\_conv\_pos\_embeddings = 128num\_conv\_pos\_embedding\_groups = 16apply\_spec\_augment = Truemask\_time\_prob = 0.05mask\_time\_length = 10mask\_time\_min\_masks = 2mask\_feature\_prob = 0.0mask\_feature\_length = 10mask\_feature\_min\_masks = 0pad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2decoder\_start\_token\_id = 2num\_mel\_bins = 80speech\_decoder\_prenet\_layers = 2speech\_decoder\_prenet\_units = 256speech\_decoder\_prenet\_dropout = 0.5speaker\_embedding\_dim = 512speech\_decoder\_postnet\_layers = 5speech\_decoder\_postnet\_units = 256speech\_decoder\_postnet\_kernel = 5speech\_decoder\_postnet\_dropout = 0.5reduction\_factor = 2max\_speech\_positions = 4000max\_text\_positions = 450encoder\_max\_relative\_position = 160use\_guided\_attention\_loss = Trueguided\_attention\_loss\_num\_heads = 2guided\_attention\_loss\_sigma = 0.4guided\_attention\_loss\_scale = 10.0use\_cache = Trueis\_encoder\_decoder = True\*\*kwargs )

This is the configuration class to store the configuration of a [SpeechT5Model](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Model). It is used to instantiate a SpeechT5 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the SpeechT5 [microsoft/speecht5\_asr](https://huggingface.co/microsoft/speecht5_asr) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import SpeechT5Model, SpeechT5Config

>>> 
>>> configuration = SpeechT5Config()

>>> 
>>> model = SpeechT5Model(configuration)

>>> 
>>> configuration = model.config
```

## SpeechT5HifiGanConfig

### class transformers.SpeechT5HifiGanConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/configuration_speecht5.py#L349)

( model\_in\_dim = 80sampling\_rate = 16000upsample\_initial\_channel = 512upsample\_rates = \[4, 4, 4, 4\]upsample\_kernel\_sizes = \[8, 8, 8, 8\]resblock\_kernel\_sizes = \[3, 7, 11\]resblock\_dilation\_sizes = \[\[1, 3, 5\], \[1, 3, 5\], \[1, 3, 5\]\]initializer\_range = 0.01leaky\_relu\_slope = 0.1normalize\_before = True\*\*kwargs )

This is the configuration class to store the configuration of a `SpeechT5HifiGanModel`. It is used to instantiate a SpeechT5 HiFi-GAN vocoder model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the SpeechT5 [microsoft/speecht5\_hifigan](https://huggingface.co/microsoft/speecht5_hifigan) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig

>>> 
>>> configuration = SpeechT5HifiGanConfig()

>>> 
>>> model = SpeechT5HifiGan(configuration)

>>> 
>>> configuration = model.config
```

## SpeechT5Tokenizer

### class transformers.SpeechT5Tokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/tokenization_speecht5.py#L48)

( vocab\_filebos\_token = '<s>'eos\_token = '</s>'unk\_token = '<unk>'pad\_token = '<pad>'normalize = Falsesp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None\*\*kwargs )

Construct a SpeechT5 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2732)

( text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Nonetext\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonestride: int = 0is\_split\_into\_words: bool = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonereturn\_token\_type\_ids: typing.Optional\[bool\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = True\*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/tokenization_speecht5.py#L214)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3724)

( token\_ids: typing.Union\[int, typing.List\[int\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = None\*\*kwargs ) → `str`

Parameters

-   **token\_ids** (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`) — List of tokenized input ids. Can be obtained using the `__call__` method.
-   **skip\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to remove special tokens in the decoding.
-   **clean\_up\_tokenization\_spaces** (`bool`, _optional_) — Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.
-   **kwargs** (additional keyword arguments, _optional_) — Will be passed to the underlying model specific decode method.

The decoded sentence.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

#### batch\_decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3690)

( sequences: typing.Union\[typing.List\[int\], typing.List\[typing.List\[int\]\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = None\*\*kwargs ) → `List[str]`

Parameters

-   **sequences** (`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`) — List of tokenized input ids. Can be obtained using the `__call__` method.
-   **skip\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to remove special tokens in the decoding.
-   **clean\_up\_tokenization\_spaces** (`bool`, _optional_) — Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces`.
-   **kwargs** (additional keyword arguments, _optional_) — Will be passed to the underlying model specific decode method.

The list of decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

## SpeechT5FeatureExtractor

( feature\_size: int = 1sampling\_rate: int = 16000padding\_value: float = 0.0do\_normalize: bool = Falsenum\_mel\_bins: int = 80hop\_length: int = 16win\_length: int = 64win\_function: str = 'hann\_window'frame\_signal\_scale: float = 1.0fmin: float = 80fmax: float = 7600mel\_floor: float = 1e-10reduction\_factor: int = 2return\_attention\_mask: bool = True\*\*kwargs )

Constructs a SpeechT5 feature extractor.

This class can pre-process a raw speech signal by (optionally) normalizing to zero-mean unit-variance, for use by the SpeechT5 speech encoder prenet.

This class can also extract log-mel filter bank features from raw speech, for use by the SpeechT5 speech decoder prenet.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

( audio: typing.Union\[numpy.ndarray, typing.List\[float\], typing.List\[numpy.ndarray\], typing.List\[typing.List\[float\]\], NoneType\] = Noneaudio\_target: typing.Union\[numpy.ndarray, typing.List\[float\], typing.List\[numpy.ndarray\], typing.List\[typing.List\[float\]\], NoneType\] = Nonepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsemax\_length: typing.Optional\[int\] = Nonetruncation: bool = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonesampling\_rate: typing.Optional\[int\] = None\*\*kwargs )

Main method to featurize and prepare for the model one or several sequence(s).

Pass in a value for `audio` to extract waveform features. Pass in a value for `audio_target` to extract log-mel spectrogram features.

## SpeechT5Processor

### class transformers.SpeechT5Processor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/processing_speecht5.py#L20)

( feature\_extractortokenizer )

Parameters

-   **feature\_extractor** (`SpeechT5FeatureExtractor`) — An instance of [SpeechT5FeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor). The feature extractor is a required input.
-   **tokenizer** (`SpeechT5Tokenizer`) — An instance of [SpeechT5Tokenizer](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer). The tokenizer is a required input.

Constructs a SpeechT5 processor which wraps a feature extractor and a tokenizer into a single processor.

[SpeechT5Processor](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Processor) offers all the functionalities of [SpeechT5FeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor) and [SpeechT5Tokenizer](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer). See the docstring of [**call**()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Processor.__call__) and [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Processor.decode) for more information.

Processes audio and text input, as well as audio and text targets.

You can process audio by using the argument `audio`, or process audio targets by using the argument `audio_target`. This forwards the arguments to SpeechT5FeatureExtractor’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5FeatureExtractor.__call__).

You can process text by using the argument `text`, or process text labels by using the argument `text_target`. This forwards the arguments to SpeechT5Tokenizer’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__).

Valid input combinations are:

-   `text` only
-   `audio` only
-   `text_target` only
-   `audio_target` only
-   `text` and `audio_target`
-   `audio` and `audio_target`
-   `text` and `text_target`
-   `audio` and `text_target`

Please refer to the docstring of the above two methods for more information.

Collates the audio and text inputs, as well as their targets, into a padded batch.

Audio inputs are padded by SpeechT5FeatureExtractor’s [pad()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad). Text inputs are padded by SpeechT5Tokenizer’s [pad()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad).

Valid input combinations are:

-   `input_ids` only
-   `input_values` only
-   `labels` only, either log-mel spectrograms or text tokens
-   `input_ids` and log-mel spectrogram `labels`
-   `input_values` and text `labels`

Please refer to the docstring of the above two methods for more information.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/processing_utils.py#L167)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\]cache\_dir: typing.Union\[str, os.PathLike, NoneType\] = Noneforce\_download: bool = Falselocal\_files\_only: bool = Falsetoken: typing.Union\[bool, str, NoneType\] = Nonerevision: str = 'main'\*\*kwargs )

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — This can be either:
    
    -   a string, the _model id_ of a pretrained feature\_extractor hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
    -   a path to a _directory_ containing a feature extractor file saved using the [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`.
    -   a path or url to a saved feature extractor JSON _file_, e.g., `./my_model_directory/preprocessor_config.json`. \*\*kwargs — Additional keyword arguments passed along to both [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
    

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the methods above for more information.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/processing_utils.py#L93)

( save\_directorypush\_to\_hub: bool = False\*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it can be reloaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/nougat#transformers.NougatProcessor.from_pretrained) method.

This class method is simply calling [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and [save\_pretrained()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the methods above for more information.

This method forwards all its arguments to SpeechT5Tokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to SpeechT5Tokenizer’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## SpeechT5Model

### class transformers.SpeechT5Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L2178)

( config: SpeechT5Configencoder: typing.Optional\[torch.nn.modules.module.Module\] = Nonedecoder: typing.Optional\[torch.nn.modules.module.Module\] = None )

Parameters

-   **config** ([SpeechT5Config](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
-   **encoder** (`SpeechT5EncoderWithSpeechPrenet` or `SpeechT5EncoderWithTextPrenet` or `None`) — The Transformer encoder module that applies the appropiate speech or text encoder prenet. If `None`, `SpeechT5EncoderWithoutPrenet` will be used and the `input_values` are assumed to be hidden states.
-   **decoder** (`SpeechT5DecoderWithSpeechPrenet` or `SpeechT5DecoderWithTextPrenet` or `None`) — The Transformer decoder module that applies the appropiate speech or text decoder prenet. If `None`, `SpeechT5DecoderWithoutPrenet` will be used and the `decoder_input_values` are assumed to be hidden states.

The bare SpeechT5 Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L2220)

( input\_values: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_values: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Nonespeaker\_embeddings: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [SpeechT5Model](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## SpeechT5ForSpeechToText

### class transformers.SpeechT5ForSpeechToText

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L2326)

( config: SpeechT5Config )

Parameters

-   **config** ([SpeechT5Config](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SpeechT5 Model with a speech encoder and a text decoder. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L2368)

( input\_values: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [SpeechT5ForSpeechToText](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToText) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
>>> from datasets import load_dataset

>>> dataset = load_dataset(
...     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
... )  
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
>>> model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

>>> 
>>> inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> predicted_ids = model.generate(**inputs, max_length=100)

>>> 
>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
>>> transcription[0]
'mister quilter is the apostle of the middle classes and we are glad to welcome his gospel'
```

```
>>> inputs["labels"] = processor(text_target=dataset[0]["text"], return_tensors="pt").input_ids

>>> 
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
19.68
```

## SpeechT5ForTextToSpeech

### class transformers.SpeechT5ForTextToSpeech

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L2629)

( config: SpeechT5Config )

Parameters

-   **config** ([SpeechT5Config](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SpeechT5 Model with a text encoder and a speech decoder. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L2658)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_values: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonespeaker\_embeddings: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.FloatTensor\] = Nonestop\_labels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.Seq2SeqSpectrogramOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput) or `tuple(torch.FloatTensor)`

The [SpeechT5ForTextToSpeech](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5ForTextToSpeech) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
>>> import torch

>>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
>>> model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
>>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

>>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
>>> speaker_embeddings = torch.zeros((1, 512))  

>>> set_seed(555)  

>>> 
>>> speech = model.generate(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
>>> speech.shape
torch.Size([15872])
```

#### generate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L2784)

( input\_ids: LongTensorspeaker\_embeddings: typing.Optional\[torch.FloatTensor\] = Nonethreshold: float = 0.5minlenratio: float = 0.0maxlenratio: float = 20.0vocoder: typing.Optional\[torch.nn.modules.module.Module\] = Noneoutput\_cross\_attentions: bool = False\*\*kwargs ) → `tuple(torch.FloatTensor)` comprising various elements depending on the inputs

Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a speech waveform using a vocoder.

## SpeechT5ForSpeechToSpeech

### class transformers.SpeechT5ForSpeechToSpeech

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L2906)

( config: SpeechT5Config )

Parameters

-   **config** ([SpeechT5Config](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SpeechT5 Model with a speech encoder and a speech decoder. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L2932)

( input\_values: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_values: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.FloatTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonespeaker\_embeddings: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.FloatTensor\] = Nonestop\_labels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.Seq2SeqSpectrogramOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSpectrogramOutput) or `tuple(torch.FloatTensor)`

The [SpeechT5ForSpeechToSpeech](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5ForSpeechToSpeech) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, set_seed
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset(
...     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
... )  
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
>>> model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
>>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

>>> 
>>> inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

>>> speaker_embeddings = torch.zeros((1, 512))  

>>> set_seed(555)  

>>> 
>>> speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)
>>> speech.shape
torch.Size([77824])
```

#### generate\_speech

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L3052)

( input\_values: FloatTensorspeaker\_embeddings: typing.Optional\[torch.FloatTensor\] = Nonethreshold: float = 0.5minlenratio: float = 0.0maxlenratio: float = 20.0vocoder: typing.Optional\[torch.nn.modules.module.Module\] = Noneoutput\_cross\_attentions: bool = False ) → `tuple(torch.FloatTensor)` comprising various elements depending on the inputs

Converts a raw speech waveform into a sequence of mel spectrograms, which are subsequently turned back into a speech waveform using a vocoder.

## SpeechT5HifiGan

### class transformers.SpeechT5HifiGan

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L3193)

( config: SpeechT5HifiGanConfig )

Parameters

-   **config** ([SpeechT5HifiGanConfig](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5HifiGanConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

HiFi-GAN vocoder. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speecht5/modeling_speecht5.py#L3258)

( spectrogram: FloatTensor ) → `torch.FloatTensor`

Parameters

-   **spectrogram** (`torch.FloatTensor`) — Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length, config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

Returns

`torch.FloatTensor`

Tensor containing the speech waveform. If the input spectrogram is batched, will be of shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.

Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech waveform.