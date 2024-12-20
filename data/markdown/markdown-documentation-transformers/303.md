# Wav2Vec2

## Overview

The Wav2Vec2 model was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

The abstract from the paper is the following:

_We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech recognition with limited amounts of labeled data._

Tips:

-   Wav2Vec2 is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
-   Wav2Vec2 model was trained using connectionist temporal classification (CTC) so the model output has to be decoded using [Wav2Vec2CTCTokenizer](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer).

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Wav2Vec2. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

-   A notebook on how to [leverage a pretrained Wav2Vec2 model for emotion classification](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb). 🌎
-   [Wav2Vec2ForCTC](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/audio-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb).
-   [Audio classification task guide](../tasks/audio_classification)

Automatic Speech Recognition

-   A blog post on [boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram).
-   A blog post on how to [finetune Wav2Vec2 for English ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english).
-   A blog post on [finetuning XLS-R for Multi-Lingual ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2).
-   A notebook on how to [create YouTube captions from any video by transcribing audio with Wav2Vec2](https://colab.research.google.com/github/Muennighoff/ytclipcc/blob/main/wav2vec_youtube_captions.ipynb). 🌎
-   [Wav2Vec2ForCTC](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) is supported by a notebook on [how to finetune a speech recognition model in English](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/speech_recognition.ipynb), and [how to finetune a speech recognition model in any language](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multi_lingual_speech_recognition.ipynb).
-   [Automatic speech recognition task guide](../tasks/asr)

🚀 Deploy

-   A blog post on how to deploy Wav2Vec2 for [Automatic Speech Recogntion with Hugging Face’s Transformers & Amazon SageMaker](https://www.philschmid.de/automatic-speech-recognition-sagemaker).

## Wav2Vec2Config

### class transformers.Wav2Vec2Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/configuration_wav2vec2.py#L32)

( vocab\_size = 32hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout = 0.1activation\_dropout = 0.1attention\_dropout = 0.1feat\_proj\_dropout = 0.0feat\_quantizer\_dropout = 0.0final\_dropout = 0.1layerdrop = 0.1initializer\_range = 0.02layer\_norm\_eps = 1e-05feat\_extract\_norm = 'group'feat\_extract\_activation = 'gelu'conv\_dim = (512, 512, 512, 512, 512, 512, 512)conv\_stride = (5, 2, 2, 2, 2, 2, 2)conv\_kernel = (10, 3, 3, 3, 3, 2, 2)conv\_bias = Falsenum\_conv\_pos\_embeddings = 128num\_conv\_pos\_embedding\_groups = 16do\_stable\_layer\_norm = Falseapply\_spec\_augment = Truemask\_time\_prob = 0.05mask\_time\_length = 10mask\_time\_min\_masks = 2mask\_feature\_prob = 0.0mask\_feature\_length = 10mask\_feature\_min\_masks = 0num\_codevectors\_per\_group = 320num\_codevector\_groups = 2contrastive\_logits\_temperature = 0.1num\_negatives = 100codevector\_dim = 256proj\_codevector\_dim = 256diversity\_loss\_weight = 0.1ctc\_loss\_reduction = 'sum'ctc\_zero\_infinity = Falseuse\_weighted\_layer\_sum = Falseclassifier\_proj\_size = 256tdnn\_dim = (512, 512, 512, 512, 1500)tdnn\_kernel = (5, 3, 3, 1, 1)tdnn\_dilation = (1, 2, 3, 1, 1)xvector\_output\_dim = 512pad\_token\_id = 0bos\_token\_id = 1eos\_token\_id = 2add\_adapter = Falseadapter\_kernel\_size = 3adapter\_stride = 2num\_adapter\_layers = 3output\_hidden\_size = Noneadapter\_attn\_dim = None\*\*kwargs )

This is the configuration class to store the configuration of a [Wav2Vec2Model](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Model). It is used to instantiate an Wav2Vec2 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Wav2Vec2 [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import Wav2Vec2Config, Wav2Vec2Model

>>> 
>>> configuration = Wav2Vec2Config()

>>> 
>>> model = Wav2Vec2Model(configuration)

>>> 
>>> configuration = model.config
```

## Wav2Vec2CTCTokenizer

### class transformers.Wav2Vec2CTCTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L127)

( vocab\_filebos\_token = '<s>'eos\_token = '</s>'unk\_token = '<unk>'pad\_token = '<pad>'word\_delimiter\_token = '|'replace\_word\_delimiter\_char = ' 'do\_lower\_case = Falsetarget\_lang = None\*\*kwargs )

Constructs a Wav2Vec2CTC tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains some of the main methods. Users should refer to the superclass for more information regarding such methods.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2732)

( text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Nonetext\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Nonetext\_pair\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonestride: int = 0is\_split\_into\_words: bool = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonereturn\_token\_type\_ids: typing.Optional\[bool\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = True\*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L649)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L544)

( token\_ids: typing.Union\[int, typing.List\[int\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = Noneoutput\_char\_offsets: bool = Falseoutput\_word\_offsets: bool = False\*\*kwargs ) → `str` or `Wav2Vec2CTCTokenizerOutput`

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

Example:

```
>>> 
>>> from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
>>> from datasets import load_dataset
>>> import datasets
>>> import torch

>>> 
>>> model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

>>> 
>>> dataset = load_dataset("common_voice", "en", split="train", streaming=True)
>>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
>>> dataset_iter = iter(dataset)
>>> sample = next(dataset_iter)

>>> 
>>> input_values = feature_extractor(sample["audio"]["array"], return_tensors="pt").input_values
>>> logits = model(input_values).logits[0]
>>> pred_ids = torch.argmax(logits, axis=-1)

>>> 
>>> outputs = tokenizer.decode(pred_ids, output_word_offsets=True)
>>> 
>>> time_offset = model.config.inputs_to_logits_ratio / feature_extractor.sampling_rate

>>> word_offsets = [
...     {
...         "word": d["word"],
...         "start_time": round(d["start_offset"] * time_offset, 2),
...         "end_time": round(d["end_offset"] * time_offset, 2),
...     }
...     for d in outputs.word_offsets
... ]
>>> 
>>> 
>>> word_offsets[:3]
[{'word': 'WHY', 'start_time': 1.42, 'end_time': 1.54}, {'word': 'DOES', 'start_time': 1.64, 'end_time': 1.9}, {'word': 'MILISANDRA', 'start_time': 2.26, 'end_time': 2.9}]
```

#### batch\_decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/tokenization_wav2vec2.py#L474)

( sequences: typing.Union\[typing.List\[int\], typing.List\[typing.List\[int\]\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\]skip\_special\_tokens: bool = Falseclean\_up\_tokenization\_spaces: bool = Noneoutput\_char\_offsets: bool = Falseoutput\_word\_offsets: bool = False\*\*kwargs ) → `List[str]` or `Wav2Vec2CTCTokenizerOutput`

Convert a list of lists of token ids into a list of strings by calling decode.

Set the target language of a nested multi-lingual dictionary

## Wav2Vec2FeatureExtractor

( feature\_size = 1sampling\_rate = 16000padding\_value = 0.0return\_attention\_mask = Falsedo\_normalize = True\*\*kwargs )

Constructs a Wav2Vec2 feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

( raw\_speech: typing.Union\[numpy.ndarray, typing.List\[float\], typing.List\[numpy.ndarray\], typing.List\[typing.List\[float\]\]\]padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsemax\_length: typing.Optional\[int\] = Nonetruncation: bool = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonesampling\_rate: typing.Optional\[int\] = None\*\*kwargs )

Main method to featurize and prepare for the model one or several sequence(s).

## Wav2Vec2Processor

Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor and a Wav2Vec2 CTC tokenizer into a single processor.

[Wav2Vec2Processor](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor) offers all the functionalities of [Wav2Vec2FeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor) and [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). See the docstring of [**call**()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.__call__) and [decode()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Processor.decode) for more information.

When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) and returns its output. If used in the context `as_target_processor()` this method forwards all its arguments to PreTrainedTokenizer’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__). Please refer to the docstring of the above two methods for more information.

When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor’s [pad()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) and returns its output. If used in the context `as_target_processor()` this method forwards all its arguments to PreTrainedTokenizer’s [pad()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad). Please refer to the docstring of the above two methods for more information.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/processing_wav2vec2.py#L48)

( pretrained\_model\_name\_or\_path\*\*kwargs )

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/processing_utils.py#L93)

( save\_directorypush\_to\_hub: bool = False\*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it can be reloaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/nougat#transformers.NougatProcessor.from_pretrained) method.

This class method is simply calling [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and [save\_pretrained()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the methods above for more information.

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to PreTrainedTokenizer’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## Wav2Vec2ProcessorWithLM

### class transformers.Wav2Vec2ProcessorWithLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L67)

( feature\_extractor: FeatureExtractionMixintokenizer: PreTrainedTokenizerBasedecoder: BeamSearchDecoderCTC )

Parameters

-   **feature\_extractor** ([Wav2Vec2FeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor)) — An instance of [Wav2Vec2FeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor). The feature extractor is a required input.
-   **tokenizer** ([Wav2Vec2CTCTokenizer](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer)) — An instance of [Wav2Vec2CTCTokenizer](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer). The tokenizer is a required input.
-   **decoder** (`pyctcdecode.BeamSearchDecoderCTC`) — An instance of `pyctcdecode.BeamSearchDecoderCTC`. The decoder is a required input.

Constructs a Wav2Vec2 processor which wraps a Wav2Vec2 feature extractor, a Wav2Vec2 CTC tokenizer and a decoder with language model support into a single processor for language model boosted speech recognition decoding.

When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor.__call__) and returns its output. If used in the context `as_target_processor()` this method forwards all its arguments to Wav2Vec2CTCTokenizer’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__). Please refer to the docstring of the above two methods for more information.

When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor’s [pad()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor.pad) and returns its output. If used in the context `as_target_processor()` this method forwards all its arguments to Wav2Vec2CTCTokenizer’s [pad()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.pad). Please refer to the docstring of the above two methods for more information.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L112)

( pretrained\_model\_name\_or\_path\*\*kwargs )

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — This can be either:
    
    -   a string, the _model id_ of a pretrained feature\_extractor hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
    -   a path to a _directory_ containing a feature extractor file saved using the [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`.
    -   a path or url to a saved feature extractor JSON _file_, e.g., `./my_model_directory/preprocessor_config.json`. \*\*kwargs — Additional keyword arguments passed along to both [SequenceFeatureExtractor](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) and [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)
    

Instantiate a [Wav2Vec2ProcessorWithLM](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM) from a pretrained Wav2Vec2 processor.

This class method is simply calling Wav2Vec2FeatureExtractor’s [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), Wav2Vec2CTCTokenizer’s [from\_pretrained()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.from_pretrained), and `pyctcdecode.BeamSearchDecoderCTC.load_from_hf_hub`.

Please refer to the docstrings of the methods above for more information.

#### batch\_decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L284)

( logits: ndarraypool: typing.Union\[<bound method BaseContext.Pool of <multiprocessing.context.DefaultContext object at 0x7f0b4ec9b370>>, NoneType\] = Nonenum\_processes: typing.Optional\[int\] = Nonebeam\_width: typing.Optional\[int\] = Nonebeam\_prune\_logp: typing.Optional\[float\] = Nonetoken\_min\_logp: typing.Optional\[float\] = Nonehotwords: typing.Optional\[typing.Iterable\[str\]\] = Nonehotword\_weight: typing.Optional\[float\] = Nonealpha: typing.Optional\[float\] = Nonebeta: typing.Optional\[float\] = Noneunk\_score\_offset: typing.Optional\[float\] = Nonelm\_score\_boundary: typing.Optional\[bool\] = Noneoutput\_word\_offsets: bool = Falsen\_best: int = 1 )

Batch decode output logits to audio transcription with language model support.

This function makes use of Python’s multiprocessing. Currently, multiprocessing is available only on Unix systems (see this [issue](https://github.com/kensho-technologies/pyctcdecode/issues/65)).

If you are decoding multiple batches, consider creating a `Pool` and passing it to `batch_decode`. Otherwise, `batch_decode` will be very slow since it will create a fresh `Pool` for each call. See usage example below.

Example: See [Decoding multiple audios](#decoding-multiple-audios).

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L469)

( logits: ndarraybeam\_width: typing.Optional\[int\] = Nonebeam\_prune\_logp: typing.Optional\[float\] = Nonetoken\_min\_logp: typing.Optional\[float\] = Nonehotwords: typing.Optional\[typing.Iterable\[str\]\] = Nonehotword\_weight: typing.Optional\[float\] = Nonealpha: typing.Optional\[float\] = Nonebeta: typing.Optional\[float\] = Noneunk\_score\_offset: typing.Optional\[float\] = Nonelm\_score\_boundary: typing.Optional\[bool\] = Noneoutput\_word\_offsets: bool = Falsen\_best: int = 1 )

Decode output logits to audio transcription with language model support.

Example:

```
>>> 
>>> from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
>>> from datasets import load_dataset
>>> import datasets
>>> import torch

>>> 
>>> model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
>>> processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

>>> 
>>> dataset = load_dataset("common_voice", "en", split="train", streaming=True)
>>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
>>> dataset_iter = iter(dataset)
>>> sample = next(dataset_iter)

>>> 
>>> input_values = processor(sample["audio"]["array"], return_tensors="pt").input_values
>>> with torch.no_grad():
...     logits = model(input_values).logits[0].cpu().numpy()

>>> 
>>> outputs = processor.decode(logits, output_word_offsets=True)
>>> 
>>> time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate

>>> word_offsets = [
...     {
...         "word": d["word"],
...         "start_time": round(d["start_offset"] * time_offset, 2),
...         "end_time": round(d["end_offset"] * time_offset, 2),
...     }
...     for d in outputs.word_offsets
... ]
>>> 
>>> 
>>> word_offsets[:4]
[{'word': 'WHY', 'start_time': 1.42, 'end_time': 1.54}, {'word': 'DOES', 'start_time': 1.66, 'end_time': 1.9}, {'word': 'MILISANDRA', 'start_time': 2.26, 'end_time': 2.9}, {'word': 'LOOK', 'start_time': 3.0, 'end_time': 3.16}]
```

### Decoding multiple audios

If you are planning to decode multiple batches of audios, you should consider using [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.batch_decode) and passing an instantiated `multiprocessing.Pool`. Otherwise, [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.batch_decode) performance will be slower than calling [decode()](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ProcessorWithLM.decode) for each audio individually, as it internally instantiates a new `Pool` for every call. See the example below:

```
>>> 
>>> from multiprocessing import get_context
>>> from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
>>> from datasets import load_dataset
>>> import datasets
>>> import torch

>>> 
>>> model = AutoModelForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm").to("cuda")
>>> processor = AutoProcessor.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")

>>> 
>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))


>>> def map_to_array(batch):
...     batch["speech"] = batch["audio"]["array"]
...     return batch


>>> 
>>> dataset = dataset.map(map_to_array, remove_columns=["audio"])


>>> def map_to_pred(batch, pool):
...     inputs = processor(batch["speech"], sampling_rate=16_000, padding=True, return_tensors="pt")
...     inputs = {k: v.to("cuda") for k, v in inputs.items()}

...     with torch.no_grad():
...         logits = model(**inputs).logits

...     transcription = processor.batch_decode(logits.cpu().numpy(), pool).text
...     batch["transcription"] = transcription
...     return batch


>>> 
>>> 
>>> 
>>> with get_context("fork").Pool(processes=2) as pool:
...     result = dataset.map(
...         map_to_pred, batched=True, batch_size=2, fn_kwargs={"pool": pool}, remove_columns=["speech"]
...     )

>>> result["transcription"][:2]
['MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL', "NOR IS MISTER COULTER'S MANNER LESS INTERESTING THAN HIS MATTER"]
```

## Wav2Vec2 specific outputs

### class transformers.models.wav2vec2\_with\_lm.processing\_wav2vec2\_with\_lm.Wav2Vec2DecoderWithLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2_with_lm/processing_wav2vec2_with_lm.py#L45)

( text: typing.Union\[typing.List\[typing.List\[str\]\], typing.List\[str\], str\]logit\_score: typing.Union\[typing.List\[typing.List\[float\]\], typing.List\[float\], float\] = Nonelm\_score: typing.Union\[typing.List\[typing.List\[float\]\], typing.List\[float\], float\] = Noneword\_offsets: typing.Union\[typing.List\[typing.List\[typing.List\[typing.Dict\[str, typing.Union\[int, str\]\]\]\]\], typing.List\[typing.List\[typing.Dict\[str, typing.Union\[int, str\]\]\]\], typing.List\[typing.Dict\[str, typing.Union\[int, str\]\]\]\] = None )

Parameters

-   **text** (list of `str` or `str`) — Decoded logits in text from. Usually the speech transcription.
-   **logit\_score** (list of `float` or `float`) — Total logit score of the beams associated with produced text.
-   **lm\_score** (list of `float`) — Fused lm\_score of the beams associated with produced text.
-   **word\_offsets** (list of `List[Dict[str, Union[int, str]]]` or `List[Dict[str, Union[int, str]]]`) — Offsets of the decoded words. In combination with sampling rate and model downsampling rate word offsets can be used to compute time stamps for each word.

Output type of `Wav2Vec2DecoderWithLM`, with transcription.

### class transformers.modeling\_outputs.Wav2Vec2BaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_outputs.py#L1286)

( last\_hidden\_state: FloatTensor = Noneextract\_features: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for models that have been trained with the Wav2Vec2 loss objective.

### class transformers.models.wav2vec2.modeling\_wav2vec2.Wav2Vec2ForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L100)

( loss: typing.Optional\[torch.FloatTensor\] = Noneprojected\_states: FloatTensor = Noneprojected\_quantized\_states: FloatTensor = Nonecodevector\_perplexity: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecontrastive\_loss: typing.Optional\[torch.FloatTensor\] = Nonediversity\_loss: typing.Optional\[torch.FloatTensor\] = None )

Output type of [Wav2Vec2ForPreTraining](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining), with potential hidden states and attentions.

### class transformers.models.wav2vec2.modeling\_flax\_wav2vec2.FlaxWav2Vec2BaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L45)

( last\_hidden\_state: Array = Noneextract\_features: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Output type of `FlaxWav2Vec2BaseModelOutput`, with potential hidden states and attentions.

“Returns a new object replacing the specified fields with new values.

### class transformers.models.wav2vec2.modeling\_flax\_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L75)

( projected\_states: Array = Noneprojected\_quantized\_states: Array = Nonecodevector\_perplexity: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Output type of `FlaxWav2Vec2ForPreTrainingOutput`, with potential hidden states and attentions.

“Returns a new object replacing the specified fields with new values.

## Wav2Vec2Model

### class transformers.Wav2Vec2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1456)

( config: Wav2Vec2Config )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top. Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1542)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Nonemask\_time\_indices: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Wav2Vec2BaseModelOutput](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.modeling_outputs.Wav2Vec2BaseModelOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2Model](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, Wav2Vec2Model
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
>>> model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

>>> 
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 292, 768]
```

## Wav2Vec2ForCTC

### class transformers.Wav2Vec2ForCTC

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1875)

( configtarget\_lang: typing.Optional\[str\] = None )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1947)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ForCTC](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, Wav2Vec2ForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
>>> model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

>>> 
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> 
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> 
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
53.48
```

#### load\_adapter

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1209)

( target\_lang: strforce\_load = True\*\*kwargs )

Load a language adapter model from a pre-trained adapter model.

Activate the special [“offline-mode”](https://huggingface.co/transformers/installation.html#offline-mode) to use this method in a firewalled environment.

Examples:

```
>>> from transformers import Wav2Vec2ForCTC, AutoProcessor

>>> ckpt = "facebook/mms-1b-all"
>>> processor = AutoProcessor.from_pretrained(ckpt)
>>> model = Wav2Vec2ForCTC.from_pretrained(ckpt, target_lang="eng")
>>> 
>>> processor.tokenizer.set_target_lang("spa")
>>> model.load_adapter("spa")
```

## Wav2Vec2ForSequenceClassification

### class transformers.Wav2Vec2ForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2034)

( config )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB Keyword Spotting.

Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2079)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
>>> model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks")

>>> 
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'_unknown_'

>>> 
>>> target_label = model.config.id2label[0]
>>> inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
6.54
```

## Wav2Vec2ForAudioFrameClassification

### class transformers.Wav2Vec2ForAudioFrameClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2156)

( config )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization.

Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2200)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ForAudioFrameClassification](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForAudioFrameClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ForAudioFrameClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sd")
>>> model = Wav2Vec2ForAudioFrameClassification.from_pretrained("anton-l/wav2vec2-base-superb-sd")

>>> 
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> probabilities = torch.sigmoid(logits[0])
>>> 
>>> labels = (probabilities > 0.5).long()
>>> labels[0].tolist()
[0, 0]
```

## Wav2Vec2ForXVector

### class transformers.Wav2Vec2ForXVector

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2317)

( config )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification.

Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L2379)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.XVectorOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.XVectorOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ForXVector](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForXVector) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ForXVector
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
>>> model = Wav2Vec2ForXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv")

>>> 
>>> inputs = feature_extractor(
...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
... )
>>> with torch.no_grad():
...     embeddings = model(**inputs).embeddings

>>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

>>> 
>>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
>>> similarity = cosine_sim(embeddings[0], embeddings[1])
>>> threshold = 0.7  
>>> if similarity < threshold:
...     print("Speakers are not the same!")
>>> round(similarity.item(), 2)
0.98
```

## Wav2Vec2ForPreTraining

### class transformers.Wav2Vec2ForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1604)

( config: Wav2Vec2Config )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Wav2Vec2 Model with a quantizer and `VQ` head on top. Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_wav2vec2.py#L1664)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Nonemask\_time\_indices: typing.Optional\[torch.BoolTensor\] = Nonesampled\_negative\_indices: typing.Optional\[torch.BoolTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.wav2vec2.modeling\_wav2vec2.Wav2Vec2ForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForPreTrainingOutput) or `tuple(torch.FloatTensor)`

The [Wav2Vec2ForPreTraining](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2ForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
>>> from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
>>> from datasets import load_dataset

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
>>> model = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values  

>>> 
>>> batch_size, raw_sequence_length = input_values.shape
>>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length).item()
>>> mask_time_indices = _compute_mask_indices(
...     shape=(batch_size, sequence_length), mask_prob=0.2, mask_length=2
... )
>>> sampled_negative_indices = _sample_negative_indices(
...     features_shape=(batch_size, sequence_length),
...     num_negatives=model.config.num_negatives,
...     mask_time_indices=mask_time_indices,
... )
>>> mask_time_indices = torch.tensor(data=mask_time_indices, device=input_values.device, dtype=torch.long)
>>> sampled_negative_indices = torch.tensor(
...     data=sampled_negative_indices, device=input_values.device, dtype=torch.long
... )

>>> with torch.no_grad():
...     outputs = model(input_values, mask_time_indices=mask_time_indices)

>>> 
>>> cosine_sim = torch.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states, dim=-1)

>>> 
>>> cosine_sim[mask_time_indices.to(torch.bool)].mean() > 0.5
tensor(True)

>>> 
>>> model = model.train()
>>> loss = model(
...     input_values, mask_time_indices=mask_time_indices, sampled_negative_indices=sampled_negative_indices
... ).loss
```

## TFWav2Vec2Model

### class transformers.TFWav2Vec2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1353)

( \*args\*\*kwargs )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare TFWav2Vec2 Model transformer outputing raw hidden-states without any specific head on top.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_values` only and nothing else: `model(input_values)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_values, attention_mask])` or `model([input_values, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_values": input_values, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1359)

( input\_values: tf.Tensorattention\_mask: tf.Tensor | None = Nonetoken\_type\_ids: tf.Tensor | None = Noneposition\_ids: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Noneinputs\_embeds: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutput) or `tuple(tf.Tensor)`

The [TFWav2Vec2Model](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.TFWav2Vec2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, TFWav2Vec2Model
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
>>> model = TFWav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  
>>> hidden_states = model(input_values).last_hidden_state
```

## TFWav2Vec2ForSequenceClassification

### class transformers.TFWav2Vec2ForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1576)

( \*args\*\*kwargs )

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1617)

( input\_values: tf.Tensorattention\_mask: tf.Tensor | None = Noneoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Nonereturn\_dict: bool | None = Nonelabels: tf.Tensor | None = Nonetraining: bool = False )

## TFWav2Vec2ForCTC

### class transformers.TFWav2Vec2ForCTC

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1427)

( \*args\*\*kwargs )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

TFWav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `input_values` only and nothing else: `model(input_values)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_values, attention_mask])` or `model([input_values, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_values": input_values, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_tf_wav2vec2.py#L1454)

( input\_values: tf.Tensorattention\_mask: tf.Tensor | None = Nonetoken\_type\_ids: tf.Tensor | None = Noneposition\_ids: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Noneinputs\_embeds: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Nonelabels: tf.Tensor | None = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFCausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutput) or `tuple(tf.Tensor)`

The [TFWav2Vec2ForCTC](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.TFWav2Vec2ForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import tensorflow as tf
>>> from transformers import AutoProcessor, TFWav2Vec2ForCTC
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
>>> model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  
>>> logits = model(input_values).logits
>>> predicted_ids = tf.argmax(logits, axis=-1)

>>> transcription = processor.decode(predicted_ids[0])

>>> 
>>> target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

>>> 
>>> labels = processor(text=transcription, return_tensors="tf").input_ids

>>> loss = model(input_values, labels=labels).loss
```

## FlaxWav2Vec2Model

### class transformers.FlaxWav2Vec2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L1055)

( config: Wav2Vec2Configinput\_shape: typing.Tuple = (1, 1024)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top. Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

The `FlaxWav2Vec2PreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, FlaxWav2Vec2Model
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-lv60")
>>> model = FlaxWav2Vec2Model.from_pretrained("facebook/wav2vec2-large-lv60")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> input_values = processor(
...     ds["speech"][0], sampling_rate=16_000, return_tensors="np"
... ).input_values  
>>> hidden_states = model(input_values).last_hidden_state
```

## FlaxWav2Vec2ForCTC

### class transformers.FlaxWav2Vec2ForCTC

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L1173)

( config: Wav2Vec2Configinput\_shape: typing.Tuple = (1, 1024)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L888)

( input\_valuesattention\_mask = Nonemask\_time\_indices = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonefreeze\_feature\_encoder: bool = Falsereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput) or `tuple(torch.FloatTensor)`

The `FlaxWav2Vec2PreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoProcessor, FlaxWav2Vec2ForCTC
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h-lv60")
>>> model = FlaxWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> input_values = processor(
...     ds["speech"][0], sampling_rate=16_000, return_tensors="np"
... ).input_values  
>>> logits = model(input_values).logits
>>> predicted_ids = jnp.argmax(logits, axis=-1)

>>> transcription = processor.decode(predicted_ids[0])
>>> 
```

## FlaxWav2Vec2ForPreTraining

### class transformers.FlaxWav2Vec2ForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L1319)

( config: Wav2Vec2Configinput\_shape: typing.Tuple = (1, 1024)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([Wav2Vec2Config](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Wav2Vec2 Model with a quantizer and `VQ` head on top. Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/wav2vec2/modeling_flax_wav2vec2.py#L1322)

( input\_valuesattention\_mask = Nonemask\_time\_indices = Nonegumbel\_temperature: int = 1params: dict = Nonedropout\_rng: PRNGKey = Nonegumbel\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonefreeze\_feature\_encoder: bool = Falsereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.wav2vec2.modeling\_flax\_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.models.wav2vec2.modeling_flax_wav2vec2.FlaxWav2Vec2ForPreTrainingOutput) or `tuple(torch.FloatTensor)`

The [FlaxWav2Vec2ForPreTraining](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.FlaxWav2Vec2ForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import optax
>>> import numpy as np
>>> import jax.numpy as jnp
>>> from transformers import AutoFeatureExtractor, FlaxWav2Vec2ForPreTraining
>>> from transformers.models.wav2vec2.modeling_flax_wav2vec2 import _compute_mask_indices
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-lv60")
>>> model = FlaxWav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-large-lv60")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> input_values = feature_extractor(ds["speech"][0], return_tensors="np").input_values  

>>> 
>>> batch_size, raw_sequence_length = input_values.shape
>>> sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
>>> mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)

>>> outputs = model(input_values, mask_time_indices=mask_time_indices)

>>> 
>>> cosine_sim = optax.cosine_similarity(outputs.projected_states, outputs.projected_quantized_states)

>>> 
>>> assert np.asarray(cosine_sim)[mask_time_indices].mean() > 0.5
```