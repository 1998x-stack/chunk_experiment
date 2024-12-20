# Speech2Text2

## Overview

The Speech2Text2 model is used together with [Wav2Vec2](wav2vec2) for Speech Translation models proposed in [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://arxiv.org/abs/2104.06678) by Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau.

Speech2Text2 is a _decoder-only_ transformer model that can be used with any speech _encoder-only_, such as [Wav2Vec2](wav2vec2) or [HuBERT](hubert) for Speech-to-Text tasks. Please refer to the [SpeechEncoderDecoder](speech-encoder-decoder) class on how to combine Speech2Text2 with any speech _encoder-only_ model.

This model was contributed by [Patrick von Platen](https://huggingface.co/patrickvonplaten).

The original code can be found [here](https://github.com/pytorch/fairseq/blob/1f7ef9ed1e1061f8c7f88f8b94c7186834398690/fairseq/models/wav2vec/wav2vec2_asr.py#L266).

Tips:

-   Speech2Text2 achieves state-of-the-art results on the CoVoST Speech Translation dataset. For more information, see the [official models](https://huggingface.co/models?other=speech2text2) .
-   Speech2Text2 is always used within the [SpeechEncoderDecoder](speech-encoder-decoder) framework.
-   Speech2Text2’s tokenizer is based on [fastBPE](https://github.com/glample/fastBPE).

## Inference

Speech2Text2’s [SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) model accepts raw waveform input values from speech and makes use of [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate) to translate the input speech autoregressively to the target language.

The [Wav2Vec2FeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) class is responsible for preprocessing the input speech and [Speech2Text2Tokenizer](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer) decodes the generated target tokens to the target string. The [Speech2Text2Processor](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor) wraps [Wav2Vec2FeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) and [Speech2Text2Tokenizer](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer) into a single instance to both extract the input features and decode the predicted token ids.

-   Step-by-step Speech Translation

```
>>> import torch
>>> from transformers import Speech2Text2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
>>> processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
>>> generated_ids = model.generate(inputs=inputs["input_values"], attention_mask=inputs["attention_mask"])

>>> transcription = processor.batch_decode(generated_ids)
```

-   Speech Translation via Pipelines
    
    The automatic speech recognition pipeline can also be used to translate speech in just a couple lines of code
    

```
>>> from datasets import load_dataset
>>> from transformers import pipeline

>>> librispeech_en = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> asr = pipeline(
...     "automatic-speech-recognition",
...     model="facebook/s2t-wav2vec2-large-en-de",
...     feature_extractor="facebook/s2t-wav2vec2-large-en-de",
... )

>>> translation_de = asr(librispeech_en[0]["file"])
```

See [model hub](https://huggingface.co/models?filter=speech2text2) to look for Speech2Text2 checkpoints.

## Documentation resources

-   [Causal language modeling task guide](../tasks/language_modeling)

## Speech2Text2Config

### class transformers.Speech2Text2Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text_2/configuration_speech_to_text_2.py#L31)

( vocab\_size = 10000decoder\_layers = 6decoder\_ffn\_dim = 2048decoder\_attention\_heads = 4decoder\_layerdrop = 0.0use\_cache = Trueactivation\_function = 'relu'd\_model = 256dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02decoder\_start\_token\_id = 2scale\_embedding = Truepad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2max\_target\_positions = 1024\*\*kwargs )

This is the configuration class to store the configuration of a [Speech2Text2ForCausalLM](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2ForCausalLM). It is used to instantiate an Speech2Text2 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Speech2Text2 [facebook/s2t-wav2vec2-large-en-de](https://huggingface.co/facebook/s2t-wav2vec2-large-en-de) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Speech2Text2Config, Speech2Text2ForCausalLM

>>> 
>>> configuration = Speech2Text2Config()

>>> 
>>> model = Speech2Text2ForCausalLM(configuration)

>>> 
>>> configuration = model.config
```

## Speech2TextTokenizer

### class transformers.Speech2Text2Tokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text_2/tokenization_speech_to_text_2.py#L73)

( vocab\_filebos\_token = '<s>'pad\_token = '<pad>'eos\_token = '</s>'unk\_token = '<unk>'do\_lower\_case = Falsemerges\_file = None\*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — File containing the vocabulary.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sentence token.
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sentence token.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
    
    \*\*kwargs — Additional keyword arguments passed along to [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
    

Constructs a Speech2Text2Tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to the superclass for more information regarding such methods.

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

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text_2/tokenization_speech_to_text_2.py#L240)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## Speech2Text2Processor

### class transformers.Speech2Text2Processor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text_2/processing_speech_to_text_2.py#L24)

( feature\_extractortokenizer )

Parameters

-   **feature\_extractor** (`AutoFeatureExtractor`) — An instance of [AutoFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoFeatureExtractor). The feature extractor is a required input.
-   **tokenizer** (`Speech2Text2Tokenizer`) — An instance of [Speech2Text2Tokenizer](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer). The tokenizer is a required input.

Constructs a Speech2Text2 processor which wraps a Speech2Text2 feature extractor and a Speech2Text2 tokenizer into a single processor.

[Speech2Text2Processor](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor) offers all the functionalities of [AutoFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoFeatureExtractor) and [Speech2Text2Tokenizer](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2Tokenizer). See the [**call**()](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor.__call__) and [decode()](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2Processor.decode) for more information.

When used in normal mode, this method forwards all its arguments to AutoFeatureExtractor’s `__call__()` and returns its output. If used in the context `as_target_processor()` this method forwards all its arguments to Speech2Text2Tokenizer’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__). Please refer to the doctsring of the above two methods for more information.

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

This method forwards all its arguments to Speech2Text2Tokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to Speech2Text2Tokenizer’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## Speech2Text2ForCausalLM

### class transformers.Speech2Text2ForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py#L758)

( config )

Parameters

-   **config** ([Speech2Text2Config](/docs/transformers/v4.34.0/en/model_doc/speech_to_text_2#transformers.Speech2Text2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Speech2Text2 Decoder with a language modeling head. Can be used as the decoder part of [EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) and `SpeechEncoderDecoder`. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_to_text_2/modeling_speech_to_text_2.py#L791)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import (
...     SpeechEncoderDecoderModel,
...     Speech2Text2ForCausalLM,
...     Wav2Vec2Model,
...     Speech2Text2Config,
...     Wav2Vec2Config,
...     Wav2Vec2FeatureExtractor,
...     Speech2Text2Tokenizer,
... )
>>> from datasets import load_dataset

>>> feature_extractor = Wav2Vec2FeatureExtractor()
>>> tokenizer = Speech2Text2Tokenizer.from_pretrained("facebook/s2t-wav2vec2-large-en-de")

>>> encoder = Wav2Vec2Model(Wav2Vec2Config())
>>> decoder = Speech2Text2ForCausalLM(Speech2Text2Config())
>>> 

>>> model = SpeechEncoderDecoderModel(encoder=encoder, decoder=decoder)
>>> model.config.pad_token_id = tokenizer.pad_token_id
>>> model.config.decoder_start_token_id = tokenizer.bos_token_id
>>> 

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(
...     ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt"
... )
>>> input_values = inputs.input_values
>>> decoder_input_ids = tokenizer(ds[0]["text"], return_tensors="pt").input_ids
>>> 

>>> loss = model(inputs=input_values, labels=decoder_input_ids).loss
>>> 

>>> loss.backward()
```