# Whisper

## Overview

The Whisper model was proposed in [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf) by Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever.

The abstract from the paper is the following:

_We study the capabilities of speech processing systems trained simply to predict large amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual and multitask supervision, the resulting models generalize well to standard benchmarks and are often competitive with prior fully supervised results but in a zeroshot transfer setting without the need for any finetuning. When compared to humans, the models approach their accuracy and robustness. We are releasing models and inference code to serve as a foundation for further work on robust speech processing._

Tips:

-   The model usually performs well without requiring any finetuning.
-   The architecture follows a classic encoder-decoder architecture, which means that it relies on the [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate) function for inference.
-   Inference is currently only implemented for short-form i.e. audio is pre-segmented into <=30s segments. Long-form (including timestamps) will be implemented in a future release.
-   One can use [WhisperProcessor](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperProcessor) to prepare audio for the model, and decode the predicted ID’s back into text.

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ). The Tensorflow version of this model was contributed by [amyeroberts](https://huggingface.co/amyeroberts). The original code can be found [here](https://github.com/openai/whisper).

## WhisperConfig

### class transformers.WhisperConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/configuration_whisper.py#L62)

( vocab\_size = 51865num\_mel\_bins = 80encoder\_layers = 6encoder\_attention\_heads = 4decoder\_layers = 6decoder\_attention\_heads = 4decoder\_ffn\_dim = 1536encoder\_ffn\_dim = 1536encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0decoder\_start\_token\_id = 50257use\_cache = Trueis\_encoder\_decoder = Trueactivation\_function = 'gelu'd\_model = 256dropout = 0.0attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02scale\_embedding = Falsemax\_source\_positions = 1500max\_target\_positions = 448pad\_token\_id = 50256bos\_token\_id = 50256eos\_token\_id = 50256suppress\_tokens = Nonebegin\_suppress\_tokens = \[220, 50256\]use\_weighted\_layer\_sum = Falseclassifier\_proj\_size = 256apply\_spec\_augment = Falsemask\_time\_prob = 0.05mask\_time\_length = 10mask\_time\_min\_masks = 2mask\_feature\_prob = 0.0mask\_feature\_length = 10mask\_feature\_min\_masks = 0median\_filter\_width = 7\*\*kwargs )

This is the configuration class to store the configuration of a [WhisperModel](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperModel). It is used to instantiate a Whisper model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Whisper [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import WhisperConfig, WhisperModel

>>> 
>>> configuration = WhisperConfig()

>>> 
>>> model = WhisperModel(configuration)

>>> 
>>> configuration = model.config
```

## WhisperTokenizer

### class transformers.WhisperTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper.py#L215)

( vocab\_filemerges\_filenormalizer\_file = Noneerrors = 'replace'unk\_token = '<|endoftext|>'bos\_token = '<|endoftext|>'eos\_token = '<|endoftext|>'pad\_token = Noneadd\_prefix\_space = Falselanguage = Nonetask = Nonepredict\_timestamps = False\*\*kwargs )

Construct a Whisper tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to the superclass for more information regarding such methods.

#### set\_prefix\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper.py#L385)

( language: str = Nonetask: str = Nonepredict\_timestamps: bool = None )

Parameters

-   **language** (`str`, _optional_, defaults to `None`) — The language of the transcription text.
-   **task** (`str`, _optional_, defaults to `None`) — Task identifier to append at the start of sequence (if any).
-   **predict\_timestamps** (`bool`, _optional_, defaults to `None`) — Whether to omit the `<|notimestamps|>` token at the start of the sequence.

Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to

update the prefix tokens as required when fine-tuning. Example:

```
>>> 
>>> tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="spanish")
>>> 
>>> tokenizer.set_prefix_tokens(language="french")
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper.py#L444)

( token\_ids\_0token\_ids\_1 = None )

Build model inputs from a sequence by appending eos\_token\_id.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper.py#L452)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3305)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — The first tokenized sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — The second tokenized sequence.

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper.py#L718)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## WhisperTokenizerFast

### class transformers.WhisperTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper_fast.py#L90)

( vocab\_file = Nonemerges\_file = Nonenormalizer\_file = Nonetokenizer\_file = Noneunk\_token = '<|endoftext|>'bos\_token = '<|endoftext|>'eos\_token = '<|endoftext|>'add\_prefix\_space = Falselanguage = Nonetask = Nonepredict\_timestamps = False\*\*kwargs )

Construct a “fast” Whisper tokenizer (backed by HuggingFace’s _tokenizers_ library).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### set\_prefix\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper_fast.py#L421)

( language: str = Nonetask: str = Nonepredict\_timestamps: bool = None )

Parameters

-   **language** (`str`, _optional_, defaults to `None`) — The language of the transcription text.
-   **task** (`str`, _optional_, defaults to `None`) — Task identifier to append at the start of sequence (if any).
-   **predict\_timestamps** (`bool`, _optional_, defaults to `None`) — Whether to omit the `<|notimestamps|>` token at the start of the sequence.

Override the prefix tokens appended to the start of the label sequence. This method can be used standalone to

update the prefix tokens as required when fine-tuning. Example:

```
>>> 
>>> tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny", language="spanish")
>>> 
>>> tokenizer.set_prefix_tokens(language="french")
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper_fast.py#L495)

( token\_ids\_0token\_ids\_1 = None )

Build model inputs from a sequence by appending eos\_token\_id.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper_fast.py#L503)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3305)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — The first tokenized sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — The second tokenized sequence.

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/tokenization_whisper_fast.py#L406)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## WhisperFeatureExtractor

( feature\_size = 80sampling\_rate = 16000hop\_length = 160chunk\_length = 30n\_fft = 400padding\_value = 0.0return\_attention\_mask = False\*\*kwargs )

Parameters

-   **feature\_size** (`int`, defaults to 80) — The feature dimension of the extracted features.
-   **sampling\_rate** (`int`, defaults to 16000) — The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
-   **hop\_length** (`int`, defaults to 160) — Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
-   **chunk\_length** (`int`, defaults to 30) — The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio sequences.
-   **n\_fft** (`int`, defaults to 400) — Size of the Fourier transform.
-   **padding\_value** (`float`, _optional_, defaults to 0.0) — Padding value used to pad the audio. Should correspond to silences.

Constructs a Whisper feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

This class extracts mel-filter bank features from raw speech using a custom numpy implementation of the `Short Time Fourier Transform` which should match pytorch’s `torch.stft` equivalent.

( raw\_speech: typing.Union\[numpy.ndarray, typing.List\[float\], typing.List\[numpy.ndarray\], typing.List\[typing.List\[float\]\]\]truncation: bool = Truepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonepadding: typing.Optional\[str\] = 'max\_length'max\_length: typing.Optional\[int\] = Nonesampling\_rate: typing.Optional\[int\] = Nonedo\_normalize: typing.Optional\[bool\] = None\*\*kwargs )

Main method to featurize and prepare for the model one or several sequence(s).

## WhisperProcessor

### class transformers.WhisperProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/processing_whisper.py#L23)

( feature\_extractortokenizer )

Parameters

-   **feature\_extractor** (`WhisperFeatureExtractor`) — An instance of [WhisperFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperFeatureExtractor). The feature extractor is a required input.
-   **tokenizer** (`WhisperTokenizer`) — An instance of [WhisperTokenizer](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperTokenizer). The tokenizer is a required input.

Constructs a Whisper processor which wraps a Whisper feature extractor and a Whisper tokenizer into a single processor.

[WhisperProcessor](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperProcessor) offers all the functionalities of [WhisperFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperFeatureExtractor) and [WhisperTokenizer](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperTokenizer). See the [**call**()](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperProcessor.__call__) and [decode()](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperProcessor.decode) for more information.

Forwards the `audio` argument to WhisperFeatureExtractor’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperFeatureExtractor.__call__) and the `text` argument to [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__). Please refer to the doctsring of the above two methods for more information.

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

This method forwards all its arguments to WhisperTokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to WhisperTokenizer’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## WhisperModel

### class transformers.WhisperModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_whisper.py#L1227)

( config: WhisperConfig )

Parameters

-   **config** ([WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Whisper Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_whisper.py#L1298)

( input\_features: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_inputs\_embeds: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [WhisperModel](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoFeatureExtractor, WhisperModel
>>> from datasets import load_dataset

>>> model = WhisperModel.from_pretrained("openai/whisper-base")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
>>> input_features = inputs.input_features
>>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
>>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 2, 512]
```

#### \_mask\_input\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_whisper.py#L1255)

( input\_features: FloatTensorattention\_mask: typing.Optional\[torch.LongTensor\] = None )

Masks extracted features along time axis and/or along feature axis according to [SpecAugment](https://arxiv.org/abs/1904.08779).

## WhisperForConditionalGeneration

### class transformers.WhisperForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_whisper.py#L1395)

( config: WhisperConfig )

Parameters

-   **config** ([WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Whisper Model with a language modeling head. Can be used for automatic speech recognition. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_whisper.py#L1429)

( input\_features: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_inputs\_embeds: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [WhisperForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoProcessor, WhisperForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
>>> input_features = inputs.input_features

>>> generated_ids = model.generate(inputs=input_features)

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> transcription
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

## WhisperForAudioClassification

### class transformers.WhisperForAudioClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_whisper.py#L1866)

( config )

Whisper Encoder Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB Keyword Spotting.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_whisper.py#L1893)

( input\_features: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [WhisperForAudioClassification](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperForAudioClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoFeatureExtractor, WhisperForAudioClassification
>>> from datasets import load_dataset

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
>>> model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

>>> ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
>>> sample = next(iter(ds))

>>> inputs = feature_extractor(
...     sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
... )
>>> input_features = inputs.input_features

>>> with torch.no_grad():
...     logits = model(input_features).logits

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'Afrikaans'
```

## TFWhisperModel

### class transformers.TFWhisperModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_tf_whisper.py#L1093)

( \*args\*\*kwargs )

Parameters

-   **config** ([WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare Whisper Model outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_tf_whisper.py#L1117)

( input\_features: TFModelInputType | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_position\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Nonecross\_attn\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Nonedecoder\_inputs\_embeds: Optional\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\] = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput) or `tuple(tf.Tensor)`

The [TFWhisperModel](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.TFWhisperModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import tensorflow as tf
>>> from transformers import TFWhisperModel, AutoFeatureExtractor
>>> from datasets import load_dataset

>>> model = TFWhisperModel.from_pretrained("openai/whisper-base")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="tf")
>>> input_features = inputs.input_features
>>> decoder_input_ids = tf.convert_to_tensor([[1, 1]]) * model.config.decoder_start_token_id
>>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 2, 512]
```

## TFWhisperForConditionalGeneration

### class transformers.TFWhisperForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_tf_whisper.py#L1201)

( \*args\*\*kwargs )

Parameters

-   **config** ([WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The Whisper Model with a language modeling head. Can be used for automatic speech recognition. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_tf_whisper.py#L1232)

( input\_features: TFModelInputType | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_position\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Nonecross\_attn\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Nonedecoder\_inputs\_embeds: Optional\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\] = Nonelabels: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or `tuple(tf.Tensor)`

The [TFWhisperForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.TFWhisperForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import tensorflow as tf
>>> from transformers import AutoProcessor, TFWhisperForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], return_tensors="tf")
>>> input_features = inputs.input_features

>>> generated_ids = model.generate(input_features=input_features)

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> transcription
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

## FlaxWhisperModel

### class transformers.FlaxWhisperModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_flax_whisper.py#L1165)

( config: WhisperConfiginput\_shape: typing.Tuple\[int\] = (1, 80, 3000)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = Truegradient\_checkpointing: bool = False\*\*kwargs )

Parameters

-   **config** ([WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs). This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`. **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.** If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).

The bare Whisper Model transformer outputting raw hidden-states without any specific head on top. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.) This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior. Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_flax_whisper.py#L1110)

( input\_features: Arraydecoder\_input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The `FlaxWhisperPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxWhisperModel

>>> tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
>>> model = FlaxWhisperModel.from_pretrained("openai/whisper-tiny")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## FlaxWhisperForConditionalGeneration

### class transformers.FlaxWhisperForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_flax_whisper.py#L1244)

( config: WhisperConfiginput\_shape: typing.Tuple\[int\] = (1, 80, 3000)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = Truegradient\_checkpointing: bool = False\*\*kwargs )

Parameters

-   **config** ([WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs). This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`. **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.** If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).

The Whisper Model with a language modeling head. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.) This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior. Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_flax_whisper.py#L1110)

( input\_features: Arraydecoder\_input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The `FlaxWhisperPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Transcription example:

```
>>> from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration
>>> from datasets import load_dataset

>>> processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
>>> model = FlaxWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", from_pt=True)
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = processor(ds[0]["audio"]["array"], return_tensors="np")
>>> input_features = inputs.input_features
>>> generated_ids = model.generate(input_ids=input_features)
>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> transcription
' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
```

## FlaxWhisperForAudioClassification

### class transformers.FlaxWhisperForAudioClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_flax_whisper.py#L1574)

( config: WhisperConfiginput\_shape: typing.Tuple\[int\] = (1, 80, 3000)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = Truegradient\_checkpointing: bool = False\*\*kwargs )

Parameters

-   **config** ([WhisperConfig](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.WhisperConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs). This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`. **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.** If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).

The Whisper Model with an audio classification head on top. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.) This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior. Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/whisper/modeling_flax_whisper.py#L1601)

( input\_features: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None\*\*kwargs ) → [transformers.modeling\_flax\_outputs.FlaxSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [FlaxWhisperForAudioClassification](/docs/transformers/v4.34.0/en/model_doc/whisper#transformers.FlaxWhisperForAudioClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Transcription example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoFeatureExtractor, FlaxWhisperForAudioClassification
>>> from datasets import load_dataset

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
>>> model = FlaxWhisperForAudioClassification.from_pretrained(
...     "sanchit-gandhi/whisper-medium-fleurs-lang-id", from_pt=True
... )
>>> ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)

>>> sample = next(iter(ds))

>>> inputs = feature_extractor(
...     sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="np"
... )
>>> input_features = inputs.input_features

>>> logits = model(input_features).logits

>>> predicted_class_ids = jnp.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'af_za'
```