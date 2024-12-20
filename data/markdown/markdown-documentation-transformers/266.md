# Speech2Text

## Overview

The Speech2Text model was proposed in [fairseq S2T: Fast Speech-to-Text Modeling with fairseq](https://arxiv.org/abs/2010.05171) by Changhan Wang, Yun Tang, Xutai Ma, Anne Wu, Dmytro Okhonko, Juan Pino. It’s a transformer-based seq2seq (encoder-decoder) model designed for end-to-end Automatic Speech Recognition (ASR) and Speech Translation (ST). It uses a convolutional downsampler to reduce the length of speech inputs by 3/4th before they are fed into the encoder. The model is trained with standard autoregressive cross-entropy loss and generates the transcripts/translations autoregressively. Speech2Text has been fine-tuned on several datasets for ASR and ST: [LibriSpeech](http://www.openslr.org/12), [CoVoST 2](https://github.com/facebookresearch/covost), [MuST-C](https://ict.fbk.eu/must-c/).

This model was contributed by [valhalla](https://huggingface.co/valhalla). The original code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text).

## Inference

Speech2Text is a speech model that accepts a float tensor of log-mel filter-bank features extracted from the speech signal. It’s a transformer-based seq2seq model, so the transcripts/translations are generated autoregressively. The `generate()` method can be used for inference.

The [Speech2TextFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) class is responsible for extracting the log-mel filter-bank features. The [Speech2TextProcessor](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextProcessor) wraps [Speech2TextFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) and [Speech2TextTokenizer](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer) into a single instance to both extract the input features and decode the predicted token ids.

The feature extractor depends on `torchaudio` and the tokenizer depends on `sentencepiece` so be sure to install those packages before running the examples. You could either install those as extra speech dependencies with `pip install transformers"[speech, sentencepiece]"` or install the packages separately with `pip install torchaudio sentencepiece`. Also `torchaudio` requires the development version of the [libsndfile](http://www.mega-nerd.com/libsndfile/) package which can be installed via a system package manager. On Ubuntu it can be installed as follows: `apt install libsndfile1-dev`

-   ASR and Speech Translation

```
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> transcription
['mister quilter is the apostle of the middle classes and we are glad to welcome his gospel']
```

-   Multilingual speech translation
    
    For multilingual speech translation models, `eos_token_id` is used as the `decoder_start_token_id` and the target language id is forced as the first generated token. To force the target language id as the first generated token, pass the `forced_bos_token_id` parameter to the `generate()` method. The following example shows how to transate English speech to French text using the _facebook/s2t-medium-mustc-multilingual-st_ checkpoint.
    

```
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-mustc-multilingual-st")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

>>> inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
>>> generated_ids = model.generate(
...     inputs["input_features"],
...     attention_mask=inputs["attention_mask"],
...     forced_bos_token_id=processor.tokenizer.lang_code_to_id["fr"],
... )

>>> translation = processor.batch_decode(generated_ids, skip_special_tokens=True)
>>> translation
["(Vidéo) Si M. Kilder est l'apossible des classes moyennes, et nous sommes heureux d'être accueillis dans son évangile."]
```

See the [model hub](https://huggingface.co/models?filter=speech_to_text) to look for Speech2Text checkpoints.

## Speech2TextConfig

### class transformers.Speech2TextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/configuration_speech_to_text.py#L31)

( vocab\_size = 10000encoder\_layers = 12encoder\_ffn\_dim = 2048encoder\_attention\_heads = 4decoder\_layers = 6decoder\_ffn\_dim = 2048decoder\_attention\_heads = 4encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0use\_cache = Trueis\_encoder\_decoder = Trueactivation\_function = 'relu'd\_model = 256dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02decoder\_start\_token\_id = 2scale\_embedding = Truepad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2max\_source\_positions = 6000max\_target\_positions = 1024num\_conv\_layers = 2conv\_kernel\_sizes = (5, 5)conv\_channels = 1024input\_feat\_per\_channel = 80input\_channels = 1\*\*kwargs )

This is the configuration class to store the configuration of a [Speech2TextModel](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextModel). It is used to instantiate an Speech2Text model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Speech2Text [facebook/s2t-small-librispeech-asr](https://huggingface.co/facebook/s2t-small-librispeech-asr) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Speech2TextConfig, Speech2TextModel

>>> 
>>> configuration = Speech2TextConfig()

>>> 
>>> model = Speech2TextModel(configuration)

>>> 
>>> configuration = model.config
```

## Speech2TextTokenizer

### class transformers.Speech2TextTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L59)

( vocab\_filespm\_filebos\_token = '<s>'eos\_token = '</s>'pad\_token = '<pad>'unk\_token = '<unk>'do\_upper\_case = Falsedo\_lower\_case = Falsetgt\_lang = Nonelang\_codes = Noneadditional\_special\_tokens = Nonesp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None\*\*kwargs )

Construct an Speech2Text tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to the superclass for more information regarding such methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L217)

( token\_ids\_0token\_ids\_1 = None )

Build model inputs from a sequence by appending eos\_token\_id.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L224)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/tokenization_speech_to_text.py#L268)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## Speech2TextFeatureExtractor

( feature\_size = 80sampling\_rate = 16000num\_mel\_bins = 80padding\_value = 0.0do\_ceptral\_normalize = Truenormalize\_means = Truenormalize\_vars = True\*\*kwargs )

Parameters

-   **feature\_size** (`int`, defaults to 80) — The feature dimension of the extracted features.
-   **sampling\_rate** (`int`, defaults to 16000) — The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
-   **num\_mel\_bins** (`int`, defaults to 80) — Number of Mel-frequency bins.
-   **padding\_value** (`float`, defaults to 0.0) — The value that is used to fill the padding vectors.
-   **do\_ceptral\_normalize** (`bool`, _optional_, defaults to `True`) — Whether or not to apply utterance-level cepstral mean and variance normalization to extracted features.
-   **normalize\_means** (`bool`, _optional_, defaults to `True`) — Whether or not to zero-mean normalize the extracted features.
-   **normalize\_vars** (`bool`, _optional_, defaults to `True`) — Whether or not to unit-variance normalize the extracted features.

Constructs a Speech2Text feature extractor.

This feature extractor inherits from [Speech2TextFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

This class extracts mel-filter bank features from raw speech using TorchAudio and applies utterance-level cepstral mean and variance normalization to the extracted features.

( raw\_speech: typing.Union\[numpy.ndarray, typing.List\[float\], typing.List\[numpy.ndarray\], typing.List\[typing.List\[float\]\]\]padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsemax\_length: typing.Optional\[int\] = Nonetruncation: bool = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonesampling\_rate: typing.Optional\[int\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = None\*\*kwargs )

Main method to featurize and prepare for the model one or several sequence(s).

## Speech2TextProcessor

### class transformers.Speech2TextProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/processing_speech_to_text.py#L24)

( feature\_extractortokenizer )

Parameters

-   **feature\_extractor** (`Speech2TextFeatureExtractor`) — An instance of [Speech2TextFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor). The feature extractor is a required input.
-   **tokenizer** (`Speech2TextTokenizer`) — An instance of [Speech2TextTokenizer](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer). The tokenizer is a required input.

Constructs a Speech2Text processor which wraps a Speech2Text feature extractor and a Speech2Text tokenizer into a single processor.

[Speech2TextProcessor](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextProcessor) offers all the functionalities of [Speech2TextFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor) and [Speech2TextTokenizer](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextTokenizer). See the [**call**()](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextProcessor.__call__) and [decode()](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextProcessor.decode) for more information.

When used in normal mode, this method forwards all its arguments to Speech2TextFeatureExtractor’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextFeatureExtractor.__call__) and returns its output. If used in the context `as_target_processor()` this method forwards all its arguments to Speech2TextTokenizer’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__). Please refer to the doctsring of the above two methods for more information.

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

This method forwards all its arguments to Speech2TextTokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to Speech2TextTokenizer’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## Speech2TextModel

### class transformers.Speech2TextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1136)

( config: Speech2TextConfig )

Parameters

-   **config** ([Speech2TextConfig](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Speech2Text Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1158)

( input\_features: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [Speech2TextModel](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import Speech2TextModel, AutoFeatureExtractor
>>> from datasets import load_dataset

>>> model = Speech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(
...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
... )
>>> input_features = inputs.input_features
>>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
>>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 2, 256]
```

## Speech2TextForConditionalGeneration

### class transformers.Speech2TextForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1267)

( config: Speech2TextConfig )

Parameters

-   **config** ([Speech2TextConfig](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Speech2Text Model with a language modeling head. Can be used for summarization. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/modeling_speech_to_text.py#L1291)

( input\_features: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [Speech2TextForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
>>> from datasets import load_dataset

>>> model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> inputs = processor(
...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
... )
>>> input_features = inputs.input_features

>>> generated_ids = model.generate(inputs=input_features)

>>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> transcription
'mister quilter is the apostle of the middle classes and we are glad to welcome his gospel'
```

## TFSpeech2TextModel

### class transformers.TFSpeech2TextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/modeling_tf_speech_to_text.py#L1205)

( \*args\*\*kwargs )

Parameters

-   **config** ([Speech2TextConfig](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare Speech2Text Model outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/modeling_tf_speech_to_text.py#L1217)

( input\_features: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Nonecross\_attn\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: np.ndarray | tf.Tensor | None = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False\*\*kwargs ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput) or `tuple(tf.Tensor)`

The [TFSpeech2TextModel](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.TFSpeech2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFSpeech2TextModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/s2t-small-librispeech-asr")
>>> model = TFSpeech2TextModel.from_pretrained("facebook/s2t-small-librispeech-asr")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFSpeech2TextForConditionalGeneration

### class transformers.TFSpeech2TextForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/modeling_tf_speech_to_text.py#L1287)

( \*args\*\*kwargs )

Parameters

-   **config** ([Speech2TextConfig](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.Speech2TextConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The Speech2Text Model with a language modeling head. Can be used for summarization. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text/modeling_tf_speech_to_text.py#L1311)

( input\_features: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Nonecross\_attn\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: np.ndarray | tf.Tensor | None = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Nonelabels: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False\*\*kwargs ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or `tuple(tf.Tensor)`

The [TFSpeech2TextForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/speech_to_text#transformers.TFSpeech2TextForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import tensorflow as tf
>>> from transformers import Speech2TextProcessor, TFSpeech2TextForConditionalGeneration
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> model = TFSpeech2TextForConditionalGeneration.from_pretrained(
...     "facebook/s2t-small-librispeech-asr", from_pt=True
... )
>>> processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)
>>> ds.set_format(type="tf")

>>> input_features = processor(
...     ds["speech"][0], sampling_rate=16000, return_tensors="tf"
... ).input_features  
>>> generated_ids = model.generate(input_features)

>>> transcription = processor.batch_decode(generated_ids)
```