# Encoder Decoder Models

## Overview

The [EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the encoder and any pretrained autoregressive model as the decoder.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.

After such an [EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

An application of this architecture could be to leverage two pretrained [BertModel](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertModel) as the encoder and decoder for a summarization model as was shown in: [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) by Yang Liu and Mirella Lapata.

## Randomly initializing `EncoderDecoderModel` from model configurations.

[EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) can be randomly initialized from an encoder and a decoder config. In the following example, we show how to do this using the default [BertModel](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertModel) configuration for the encoder and the default `BertForCausalLM` configuration for the decoder.

```
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = EncoderDecoderModel(config=config)
```

## Initialising `EncoderDecoderModel` from a pretrained encoder and a pretrained decoder.

[EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) can be initialized from a pretrained encoder checkpoint and a pretrained decoder checkpoint. Note that any pretrained auto-encoding model, _e.g._ BERT, can serve as the encoder and both pretrained auto-encoding models, _e.g._ BERT, pretrained causal language models, _e.g._ GPT2, as well as the pretrained decoder part of sequence-to-sequence models, _e.g._ decoder of BART, can be used as the decoder. Depending on which architecture you choose as the decoder, the cross-attention layers might be randomly initialized. Initializing [EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) from a pretrained encoder and decoder checkpoint requires the model to be fine-tuned on a downstream task, as has been shown in [the _Warm-starting-encoder-decoder blog post_](https://huggingface.co/blog/warm-starting-encoder-decoder). To do so, the `EncoderDecoderModel` class provides a [EncoderDecoderModel.from\_encoder\_decoder\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel.from_encoder_decoder_pretrained) method.

```
>>> from transformers import EncoderDecoderModel, BertTokenizer

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
```

## Loading an existing `EncoderDecoderModel` checkpoint and perform inference.

To load fine-tuned checkpoints of the `EncoderDecoderModel` class, [EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) provides the `from_pretrained(...)` method just like any other model architecture in Transformers.

To perform inference, one uses the `generate` method, which allows to autoregressively generate text. This method supports various forms of decoding, such as greedy, beam search and multinomial sampling.

```
>>> from transformers import AutoTokenizer, EncoderDecoderModel

>>> 
>>> model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")
>>> tokenizer = AutoTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")

>>> 
>>> ARTICLE_TO_SUMMARIZE = (
...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
... )
>>> input_ids = tokenizer(ARTICLE_TO_SUMMARIZE, return_tensors="pt").input_ids

>>> 
>>> generated_ids = model.generate(input_ids)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
nearly 800 thousand customers were affected by the shutoffs. the aim is to reduce the risk of wildfires. nearly 800, 000 customers were expected to be affected by high winds amid dry conditions. pg & e said it scheduled the blackouts to last through at least midday tomorrow.
```

## Loading a PyTorch checkpoint into `TFEncoderDecoderModel`.

`TFEncoderDecoderModel.from_pretrained()` currently doesn’t support initializing the model from a pytorch checkpoint. Passing `from_pt=True` to this method will throw an exception. If there are only pytorch checkpoints for a particular encoder-decoder model, a workaround is:

```
>>> 
>>> from transformers import EncoderDecoderModel, TFEncoderDecoderModel

>>> _model = EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert-cnn_dailymail-fp16")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> 
>>> model.config = _model.config
```

## Training

Once the model is created, it can be fine-tuned similar to BART, T5 or any other encoder-decoder model. As you can see, only 2 inputs are required for the model in order to compute a loss: `input_ids` (which are the `input_ids` of the encoded input sequence) and `labels` (which are the `input_ids` of the encoded target sequence).

```
>>> from transformers import BertTokenizer, EncoderDecoderModel

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> input_ids = tokenizer(
...     "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side.During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was  finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft).Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.",
...     return_tensors="pt",
... ).input_ids

>>> labels = tokenizer(
...     "the eiffel tower surpassed the washington monument to become the tallest structure in the world. it was the first structure to reach a height of 300 metres in paris in 1930. it is now taller than the chrysler building by 5. 2 metres ( 17 ft ) and is the second tallest free - standing structure in paris.",
...     return_tensors="pt",
... ).input_ids

>>> 
>>> loss = model(input_ids=input_ids, labels=labels).loss
```

Detailed [colab](https://colab.research.google.com/drive/1WIk2bxglElfZewOHboPFNj8H44_VAyKE?usp=sharing#scrollTo=ZwQIEhKOrJpl) for training.

This model was contributed by [thomwolf](https://github.com/thomwolf). This model’s TensorFlow and Flax versions were contributed by [ydshieh](https://github.com/ydshieh).

## EncoderDecoderConfig

### class transformers.EncoderDecoderConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/configuration_encoder_decoder.py#L25)

( \*\*kwargs )

Parameters

-   **kwargs** (_optional_) — Dictionary of keyword arguments. Notably:
    
    -   **encoder** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig), _optional_) — An instance of a configuration object that defines the encoder config.
    -   **decoder** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig), _optional_) — An instance of a configuration object that defines the decoder config.
    

[EncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) is the configuration class to store the configuration of a [EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel). It is used to instantiate an Encoder Decoder model according to the specified arguments, defining the encoder and decoder configs.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

>>> 
>>> config_encoder = BertConfig()
>>> config_decoder = BertConfig()

>>> config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

>>> 
>>> model = EncoderDecoderModel(config=config)

>>> 
>>> config_encoder = model.config.encoder
>>> config_decoder = model.config.decoder
>>> 
>>> config_decoder.is_decoder = True
>>> config_decoder.add_cross_attention = True

>>> 
>>> model.save_pretrained("my-model")

>>> 
>>> encoder_decoder_config = EncoderDecoderConfig.from_pretrained("my-model")
>>> model = EncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
```

#### from\_encoder\_decoder\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/configuration_encoder_decoder.py#L90)

( encoder\_config: PretrainedConfigdecoder\_config: PretrainedConfig\*\*kwargs ) → [EncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)

An instance of a configuration object

Instantiate a [EncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig) (or a derived class) from a pre-trained encoder model configuration and decoder model configuration.

## EncoderDecoderModel

### class transformers.EncoderDecoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L170)

( config: typing.Optional\[transformers.configuration\_utils.PretrainedConfig\] = Noneencoder: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Nonedecoder: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = None )

Parameters

-   **config** ([EncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function and the decoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream generative task, like summarization.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

[EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) is a generic model class that will be instantiated as a transformer architecture with one of the base model classes of the library as encoder and another one as decoder when created with the :meth_~transformers.AutoModel.from\_pretrained_ class method for the encoder and :meth_~transformers.AutoModelForCausalLM.from\_pretrained_ class method for the decoder.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L539)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [EncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import EncoderDecoderModel, BertTokenizer
>>> import torch

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
...     "bert-base-uncased", "bert-base-uncased"
... )  

>>> 
>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id
>>> model.config.vocab_size = model.config.decoder.vocab_size

>>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
>>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
>>> outputs = model(input_ids=input_ids, labels=labels)
>>> loss, logits = outputs.loss, outputs.logits

>>> 
>>> model.save_pretrained("bert2bert")
>>> model = EncoderDecoderModel.from_pretrained("bert2bert")

>>> 
>>> generated = model.generate(input_ids)
```

#### from\_encoder\_decoder\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L389)

( encoder\_pretrained\_model\_name\_or\_path: str = Nonedecoder\_pretrained\_model\_name\_or\_path: str = None\*model\_args\*\*kwargs )

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model checkpoints.

The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train the model, you need to first set it back in training mode with `model.train()`.

Example:

```
>>> from transformers import EncoderDecoderModel

>>> 
>>> model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
>>> 
>>> model.save_pretrained("./bert2bert")
>>> 
>>> model = EncoderDecoderModel.from_pretrained("./bert2bert")
```

## TFEncoderDecoderModel

### class transformers.TFEncoderDecoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/modeling_tf_encoder_decoder.py#L193)

( \*args\*\*kwargs )

Parameters

-   **config** ([EncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function and the decoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream generative task, like summarization.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

[TFEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.TFEncoderDecoderModel) is a generic model class that will be instantiated as a transformer architecture with one of the base model classes of the library as encoder and another one as decoder when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method for the encoder and [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method for the decoder.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/modeling_tf_encoder_decoder.py#L469)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: np.ndarray | tf.Tensor | None = Nonepast\_key\_values: Tuple\[Tuple\[tf.Tensor\]\] | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Nonelabels: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False\*\*kwargs ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or `tuple(tf.Tensor)`

The [TFEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.TFEncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import TFEncoderDecoderModel, BertTokenizer

>>> 
>>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")

>>> tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

>>> 
>>> input_ids = tokenizer.encode(
...     "Hello, my dog is cute", add_special_tokens=True, return_tensors="tf"
... )  
>>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

>>> 
>>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
>>> loss, logits = outputs.loss, outputs.logits

>>> 
>>> model.save_pretrained("bert2gpt2")
>>> model = TFEncoderDecoderModel.from_pretrained("bert2gpt2")

>>> 
>>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.bos_token_id)
```

#### from\_encoder\_decoder\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/modeling_tf_encoder_decoder.py#L322)

( encoder\_pretrained\_model\_name\_or\_path: str = Nonedecoder\_pretrained\_model\_name\_or\_path: str = None\*model\_args\*\*kwargs )

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model checkpoints.

Example:

```
>>> from transformers import TFEncoderDecoderModel

>>> 
>>> model = TFEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")
>>> 
>>> model.save_pretrained("./bert2gpt2")
>>> 
>>> model = TFEncoderDecoderModel.from_pretrained("./bert2gpt2")
```

## FlaxEncoderDecoderModel

### class transformers.FlaxEncoderDecoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/modeling_flax_encoder_decoder.py#L302)

( config: EncoderDecoderConfiginput\_shape: typing.Optional\[typing.Tuple\] = Noneseed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([EncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.EncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

This class can be used to initialize a sequence-to-sequence model with any pretrained autoencoding model as the encoder and any pretrained autoregressive model as the decoder. The encoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function and the decoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream generative task, like summarization.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

After such an Encoder Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

[FlaxEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.FlaxEncoderDecoderModel) is a generic model class that will be instantiated as a transformer architecture with the module (flax.nn.Module) of one of the base model classes of the library as encoder module and another one as decoder module when created with the :meth_~transformers.FlaxAutoModel.from\_pretrained_ class method for the encoder and :meth_~transformers.FlaxAutoModelForCausalLM.from\_pretrained_ class method for the decoder.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/modeling_flax_encoder_decoder.py#L627)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [FlaxEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/encoder-decoder#transformers.FlaxEncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import FlaxEncoderDecoderModel, BertTokenizer, GPT2Tokenizer

>>> 
>>> model = FlaxEncoderDecoderModel.from_pretrained("patrickvonplaten/bert2gpt2-cnn_dailymail-fp16")
>>> 
>>> tokenizer_input = BertTokenizer.from_pretrained("bert-base-cased")
>>> tokenizer_output = GPT2Tokenizer.from_pretrained("gpt2")

>>> article = '''Sigma Alpha Epsilon is under fire for a video showing party-bound fraternity members
>>> singing a racist chant. SAE's national chapter suspended the students,
>>> but University of Oklahoma President David Boren took it a step further,
>>> saying the university's affiliation with the fraternity is permanently done.'''

>>> input_ids = tokenizer_input(article, add_special_tokens=True, return_tensors="np").input_ids

>>> 
>>> model.config.eos_token_id = model.config.decoder.eos_token_id
>>> model.config.pad_token_id = model.config.eos_token_id

>>> sequences = model.generate(input_ids, num_beams=4, max_length=12).sequences

>>> summary = tokenizer_output.batch_decode(sequences, skip_special_tokens=True)[0]
>>> assert summary == "SAS Alpha Epsilon suspended Sigma Alpha Epsilon members"
```

#### from\_encoder\_decoder\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/encoder_decoder/modeling_flax_encoder_decoder.py#L759)

( encoder\_pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike, NoneType\] = Nonedecoder\_pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike, NoneType\] = None\*model\_args\*\*kwargs )

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model checkpoints.

Example:

```
>>> from transformers import FlaxEncoderDecoderModel

>>> 
>>> model = FlaxEncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")
>>> 
>>> model.save_pretrained("./bert2gpt2")
>>> 
>>> model = FlaxEncoderDecoderModel.from_pretrained("./bert2gpt2")
```