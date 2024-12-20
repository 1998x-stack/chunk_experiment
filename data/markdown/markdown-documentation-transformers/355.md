# Speech Encoder Decoder Models

The [SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) can be used to initialize a speech-to-text model with any pretrained speech autoencoding model as the encoder (_e.g._ [Wav2Vec2](wav2vec2), [Hubert](hubert)) and any pretrained autoregressive model as the decoder.

The effectiveness of initializing speech-sequence-to-text-sequence models with pretrained checkpoints for speech recognition and speech translation has _e.g._ been shown in [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://arxiv.org/abs/2104.06678) by Changhan Wang, Anne Wu, Juan Pino, Alexei Baevski, Michael Auli, Alexis Conneau.

An example of how to use a [SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) for inference can be seen in [Speech2Text2](speech_to_text_2).

## Randomly initializing `SpeechEncoderDecoderModel` from model configurations.

[SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) can be randomly initialized from an encoder and a decoder config. In the following example, we show how to do this using the default [Wav2Vec2Model](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2Model) configuration for the encoder and the default `BertForCausalLM` configuration for the decoder.

```
>>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

>>> config_encoder = Wav2Vec2Config()
>>> config_decoder = BertConfig()

>>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = SpeechEncoderDecoderModel(config=config)
```

## Initialising `SpeechEncoderDecoderModel` from a pretrained encoder and a pretrained decoder.

[SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) can be initialized from a pretrained encoder checkpoint and a pretrained decoder checkpoint. Note that any pretrained Transformer-based speech model, _e.g._ [Wav2Vec2](wav2vec2), [Hubert](hubert) can serve as the encoder and both pretrained auto-encoding models, _e.g._ BERT, pretrained causal language models, _e.g._ GPT2, as well as the pretrained decoder part of sequence-to-sequence models, _e.g._ decoder of BART, can be used as the decoder. Depending on which architecture you choose as the decoder, the cross-attention layers might be randomly initialized. Initializing [SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) from a pretrained encoder and decoder checkpoint requires the model to be fine-tuned on a downstream task, as has been shown in [the _Warm-starting-encoder-decoder blog post_](https://huggingface.co/blog/warm-starting-encoder-decoder). To do so, the `SpeechEncoderDecoderModel` class provides a [SpeechEncoderDecoderModel.from\_encoder\_decoder\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel.from_encoder_decoder_pretrained) method.

```
>>> from transformers import SpeechEncoderDecoderModel

>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "facebook/hubert-large-ll60k", "bert-base-uncased"
... )
```

## Loading an existing `SpeechEncoderDecoderModel` checkpoint and perform inference.

To load fine-tuned checkpoints of the `SpeechEncoderDecoderModel` class, [SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) provides the `from_pretrained(...)` method just like any other model architecture in Transformers.

To perform inference, one uses the `generate` method, which allows to autoregressively generate text. This method supports various forms of decoding, such as greedy, beam search and multinomial sampling.

```
>>> from transformers import Wav2Vec2Processor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset
>>> import torch

>>> 
>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
>>> processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

>>> 
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> 
>>> generated_ids = model.generate(input_values)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
Mr. Quilter ist der Apostel der Mittelschicht und wir freuen uns, sein Evangelium willkommen heißen zu können.
```

## Training

Once the model is created, it can be fine-tuned similar to BART, T5 or any other encoder-decoder model on a dataset of (speech, text) pairs. As you can see, only 2 inputs are required for the model in order to compute a loss: `input_values` (which are the speech inputs) and `labels` (which are the `input_ids` of the encoded target sequence).

```
>>> from transformers import AutoTokenizer, AutoFeatureExtractor, SpeechEncoderDecoderModel
>>> from datasets import load_dataset

>>> encoder_id = "facebook/wav2vec2-base-960h"  
>>> decoder_id = "bert-base-uncased"  

>>> feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
>>> tokenizer = AutoTokenizer.from_pretrained(decoder_id)
>>> 
>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id)

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> 
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> input_values = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt").input_values

>>> 
>>> labels = tokenizer(ds[0]["text"], return_tensors="pt").input_ids

>>> 
>>> loss = model(input_values=input_values, labels=labels).loss
>>> loss.backward()
```

## SpeechEncoderDecoderConfig

### class transformers.SpeechEncoderDecoderConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py#L26)

( \*\*kwargs )

Parameters

-   **kwargs** (_optional_) — Dictionary of keyword arguments. Notably:
    
    -   **encoder** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig), _optional_) — An instance of a configuration object that defines the encoder config.
    -   **decoder** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig), _optional_) — An instance of a configuration object that defines the decoder config.
    

[SpeechEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig) is the configuration class to store the configuration of a [SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel). It is used to instantiate an Encoder Decoder model according to the specified arguments, defining the encoder and decoder configs.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel

>>> 
>>> config_encoder = Wav2Vec2Config()
>>> config_decoder = BertConfig()

>>> config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

>>> 
>>> model = SpeechEncoderDecoderModel(config=config)

>>> 
>>> config_encoder = model.config.encoder
>>> config_decoder = model.config.decoder
>>> 
>>> config_decoder.is_decoder = True
>>> config_decoder.add_cross_attention = True

>>> 
>>> model.save_pretrained("my-model")

>>> 
>>> encoder_decoder_config = SpeechEncoderDecoderConfig.from_pretrained("my-model")
>>> model = SpeechEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
```

#### from\_encoder\_decoder\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_encoder_decoder/configuration_speech_encoder_decoder.py#L92)

( encoder\_config: PretrainedConfigdecoder\_config: PretrainedConfig\*\*kwargs ) → [SpeechEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig)

An instance of a configuration object

Instantiate a [SpeechEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig) (or a derived class) from a pre-trained encoder model configuration and decoder model configuration.

## SpeechEncoderDecoderModel

### class transformers.SpeechEncoderDecoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py#L173)

( config: typing.Optional\[transformers.configuration\_utils.PretrainedConfig\] = Noneencoder: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Nonedecoder: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = None )

Parameters

-   **config** ([SpeechEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This class can be used to initialize a speech-sequence-to-text-sequence model with any pretrained speech autoencoding model as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function and the decoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream generative task, like summarization.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

Additionally, in [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://arxiv.org/abs/2104.06678) it is shown how leveraging large pretrained speech models for speech translation yields a significant performance improvement.

After such an Speech-Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

[SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) is a generic model class that will be instantiated as a transformer architecture with one of the base model classes of the library as encoder and another one as decoder when created with the :meth_~transformers.AutoModel.from\_pretrained_ class method for the encoder and :meth_~transformers.AutoModelForCausalLM.from\_pretrained_ class method for the decoder.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py#L442)

( inputs: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneinput\_values: typing.Optional\[torch.FloatTensor\] = Noneinput\_features: typing.Optional\[torch.FloatTensor\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [SpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import SpeechEncoderDecoderModel, AutoProcessor
>>> from datasets import load_dataset
>>> import torch

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")
>>> model = SpeechEncoderDecoderModel.from_pretrained("facebook/wav2vec2-xls-r-300m-en-to-15")

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> input_values = processor(ds[0]["audio"]["array"], return_tensors="pt").input_values
>>> 
>>> generated = model.generate(input_values)
>>> decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
>>> decoded
'Mr. Quilter ist der Apostel der Mittelschicht und wir freuen uns, sein Evangelium willkommen heißen zu können.'

>>> 
>>> labels = processor(text=ds[0]["text"], return_tensors="pt").input_ids

>>> loss = model(input_values, labels=labels).loss
>>> loss.backward()
```

#### from\_encoder\_decoder\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_encoder_decoder/modeling_speech_encoder_decoder.py#L287)

( encoder\_pretrained\_model\_name\_or\_path: str = Nonedecoder\_pretrained\_model\_name\_or\_path: str = None\*model\_args\*\*kwargs )

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model checkpoints.

The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train the model, you need to first set it back in training mode with `model.train()`.

Example:

```
>>> from transformers import SpeechEncoderDecoderModel

>>> 
>>> model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "facebook/wav2vec2-base-960h", "bert-base-uncased"
... )
>>> 
>>> model.save_pretrained("./wav2vec2bert")
>>> 
>>> model = SpeechEncoderDecoderModel.from_pretrained("./wav2vec2bert")
```

## FlaxSpeechEncoderDecoderModel

### class transformers.FlaxSpeechEncoderDecoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_encoder_decoder/modeling_flax_speech_encoder_decoder.py#L329)

( config: SpeechEncoderDecoderConfiginput\_shape: typing.Optional\[typing.Tuple\] = Noneseed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([SpeechEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.SpeechEncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

This class can be used to initialize a speech-sequence-to-text-sequence model with any pretrained speech autoencoding model as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function and the decoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream generative task, like summarization.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

Additionally, in [Large-Scale Self- and Semi-Supervised Learning for Speech Translation](https://arxiv.org/abs/2104.06678) it is shown how leveraging large pretrained speech models for speech translation yields a significant performance improvement.

After such an Speech-Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

[FlaxSpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.FlaxSpeechEncoderDecoderModel) is a generic model class that will be instantiated as a transformer architecture with the module (flax.nn.Module) of one of the base model classes of the library as encoder module and another one as decoder module when created with the :meth_~transformers.FlaxAutoModel.from\_pretrained_ class method for the encoder and :meth_~transformers.FlaxAutoModelForCausalLM.from\_pretrained_ class method for the decoder.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_encoder_decoder/modeling_flax_speech_encoder_decoder.py#L660)

( inputs: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falsefreeze\_feature\_encoder: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [FlaxSpeechEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/speech-encoder-decoder#transformers.FlaxSpeechEncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import FlaxSpeechEncoderDecoderModel, AutoTokenizer

>>> 
>>> model = FlaxSpeechEncoderDecoderModel.from_pretrained("patrickvonplaten/wav2vec2-2-bart-large")
>>> 
>>> tokenizer_output = AutoTokenizer.from_pretrained("facebook/bart-large")

>>> inputs = jnp.ones((2, 5000), dtype=jnp.float32)

>>> 
>>> model.config.decoder_start_token_id = model.decoder.config.bos_token_id
>>> model.config.pad_token_id = model.decoder.config.pad_token_id
>>> model.config.eos_token_id = model.decoder.config.eos_token_id

>>> outputs = model.generate(inputs)

```

#### from\_encoder\_decoder\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/speech_encoder_decoder/modeling_flax_speech_encoder_decoder.py#L782)

( encoder\_pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike, NoneType\] = Nonedecoder\_pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike, NoneType\] = None\*model\_args\*\*kwargs )

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model checkpoints.

Example:

```
>>> from transformers import FlaxSpeechEncoderDecoderModel

>>> 
>>> model = FlaxSpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "facebook/wav2vec2-large-lv60", "facebook/bart-large"
... )
>>> 
>>> model.save_pretrained("./wav2vec2-2-bart-large")
>>> 
>>> model = FlaxSpeechEncoderDecoderModel.from_pretrained("./wav2vec2-2-bart-large")
```