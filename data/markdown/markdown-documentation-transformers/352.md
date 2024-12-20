# Vision Encoder Decoder Models

## Overview

The [VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) can be used to initialize an image-to-text model with any pretrained Transformer-based vision model as the encoder (_e.g._ [ViT](vit), [BEiT](beit), [DeiT](deit), [Swin](swin)) and any pretrained language model as the decoder (_e.g._ [RoBERTa](roberta), [GPT2](gpt2), [BERT](bert), [DistilBERT](distilbert)).

The effectiveness of initializing image-to-text-sequence models with pretrained checkpoints has been shown in (for example) [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) by Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei.

After such a [VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples below for more information).

An example application is image captioning, in which the encoder is used to encode the image, after which an autoregressive language model generates the caption. Another example is optical character recognition. Refer to [TrOCR](trocr), which is an instance of [VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel).

## Randomly initializing `VisionEncoderDecoderModel` from model configurations.

[VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) can be randomly initialized from an encoder and a decoder config. In the following example, we show how to do this using the default [ViTModel](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTModel) configuration for the encoder and the default `BertForCausalLM` configuration for the decoder.

```
>>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

>>> config_encoder = ViTConfig()
>>> config_decoder = BertConfig()

>>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = VisionEncoderDecoderModel(config=config)
```

## Initialising `VisionEncoderDecoderModel` from a pretrained encoder and a pretrained decoder.

[VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) can be initialized from a pretrained encoder checkpoint and a pretrained decoder checkpoint. Note that any pretrained Transformer-based vision model, _e.g._ [Swin](swin), can serve as the encoder and both pretrained auto-encoding models, _e.g._ BERT, pretrained causal language models, _e.g._ GPT2, as well as the pretrained decoder part of sequence-to-sequence models, _e.g._ decoder of BART, can be used as the decoder. Depending on which architecture you choose as the decoder, the cross-attention layers might be randomly initialized. Initializing [VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) from a pretrained encoder and decoder checkpoint requires the model to be fine-tuned on a downstream task, as has been shown in [the _Warm-starting-encoder-decoder blog post_](https://huggingface.co/blog/warm-starting-encoder-decoder). To do so, the `VisionEncoderDecoderModel` class provides a [VisionEncoderDecoderModel.from\_encoder\_decoder\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel.from_encoder_decoder_pretrained) method.

```
>>> from transformers import VisionEncoderDecoderModel

>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "microsoft/swin-base-patch4-window7-224-in22k", "bert-base-uncased"
... )
```

## Loading an existing `VisionEncoderDecoderModel` checkpoint and perform inference.

To load fine-tuned checkpoints of the `VisionEncoderDecoderModel` class, [VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) provides the `from_pretrained(...)` method just like any other model architecture in Transformers.

To perform inference, one uses the `generate` method, which allows to autoregressively generate text. This method supports various forms of decoding, such as greedy, beam search and multinomial sampling.

```
>>> import requests
>>> from PIL import Image

>>> from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

>>> 
>>> model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> 
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> 
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
a cat laying on a blanket next to a cat laying on a bed
```

## Loading a PyTorch checkpoint into `TFVisionEncoderDecoderModel`.

`TFVisionEncoderDecoderModel.from_pretrained()` currently doesn’t support initializing the model from a PyTorch checkpoint. Passing `from_pt=True` to this method will throw an exception. If there are only PyTorch checkpoints for a particular vision encoder-decoder model, a workaround is:

```
>>> from transformers import VisionEncoderDecoderModel, TFVisionEncoderDecoderModel

>>> _model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> 
>>> model.config = _model.config
```

## Training

Once the model is created, it can be fine-tuned similar to BART, T5 or any other encoder-decoder model on a dataset of (image, text) pairs. As you can see, only 2 inputs are required for the model in order to compute a loss: `pixel_values` (which are the images) and `labels` (which are the `input_ids` of the encoded target sequence).

```
>>> from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
>>> from datasets import load_dataset

>>> image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "bert-base-uncased"
... )

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> labels = tokenizer(
...     "an image of two cats chilling on a couch",
...     return_tensors="pt",
... ).input_ids

>>> 
>>> loss = model(pixel_values=pixel_values, labels=labels).loss
```

This model was contributed by [nielsr](https://github.com/nielsrogge). This model’s TensorFlow and Flax versions were contributed by [ydshieh](https://github.com/ydshieh).

## VisionEncoderDecoderConfig

### class transformers.VisionEncoderDecoderConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py#L33)

( \*\*kwargs )

Parameters

-   **kwargs** (_optional_) — Dictionary of keyword arguments. Notably:
    
    -   **encoder** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig), _optional_) — An instance of a configuration object that defines the encoder config.
    -   **decoder** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig), _optional_) — An instance of a configuration object that defines the decoder config.
    

[VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) is the configuration class to store the configuration of a [VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel). It is used to instantiate a Vision-Encoder-Text-Decoder model according to the specified arguments, defining the encoder and decoder configs.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

>>> 
>>> config_encoder = ViTConfig()
>>> config_decoder = BertConfig()

>>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

>>> 
>>> model = VisionEncoderDecoderModel(config=config)

>>> 
>>> config_encoder = model.config.encoder
>>> config_decoder = model.config.decoder
>>> 
>>> config_decoder.is_decoder = True
>>> config_decoder.add_cross_attention = True

>>> 
>>> model.save_pretrained("my-model")

>>> 
>>> encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained("my-model")
>>> model = VisionEncoderDecoderModel.from_pretrained("my-model", config=encoder_decoder_config)
```

#### from\_encoder\_decoder\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/configuration_vision_encoder_decoder.py#L99)

( encoder\_config: PretrainedConfigdecoder\_config: PretrainedConfig\*\*kwargs ) → [VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)

An instance of a configuration object

Instantiate a [VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig) (or a derived class) from a pre-trained encoder model configuration and decoder model configuration.

## VisionEncoderDecoderModel

### class transformers.VisionEncoderDecoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L151)

( config: typing.Optional\[transformers.configuration\_utils.PretrainedConfig\] = Noneencoder: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Nonedecoder: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = None )

Parameters

-   **config** ([VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function and the decoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream generative task, like image captioning.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical character recognition (OCR) yields a significant performance improvement.

After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

[VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) is a generic model class that will be instantiated as a transformer architecture with one of the base vision model classes of the library as encoder and another one as decoder when created with the :meth_~transformers.AutoModel.from\_pretrained_ class method for the encoder and :meth_~transformers.AutoModelForCausalLM.from\_pretrained_ class method for the decoder.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L519)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.BoolTensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [VisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoProcessor, VisionEncoderDecoderModel
>>> import requests
>>> from PIL import Image
>>> import torch

>>> processor = AutoProcessor.from_pretrained("microsoft/trocr-base-handwritten")
>>> model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

>>> 
>>> url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> 
>>> model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
>>> model.config.pad_token_id = processor.tokenizer.pad_token_id
>>> model.config.vocab_size = model.config.decoder.vocab_size

>>> pixel_values = processor(image, return_tensors="pt").pixel_values
>>> text = "hello world"
>>> labels = processor.tokenizer(text, return_tensors="pt").input_ids
>>> outputs = model(pixel_values=pixel_values, labels=labels)
>>> loss = outputs.loss

>>> 
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

#### from\_encoder\_decoder\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/modeling_vision_encoder_decoder.py#L365)

( encoder\_pretrained\_model\_name\_or\_path: str = Nonedecoder\_pretrained\_model\_name\_or\_path: str = None\*model\_args\*\*kwargs )

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model checkpoints.

The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train the model, you need to first set it back in training mode with `model.train()`.

Example:

```
>>> from transformers import VisionEncoderDecoderModel

>>> 
>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "bert-base-uncased"
... )
>>> 
>>> model.save_pretrained("./vit-bert")
>>> 
>>> model = VisionEncoderDecoderModel.from_pretrained("./vit-bert")
```

## TFVisionEncoderDecoderModel

### class transformers.TFVisionEncoderDecoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/modeling_tf_vision_encoder_decoder.py#L176)

( \*args\*\*kwargs )

Parameters

-   **config** ([VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function and the decoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream generative task, like image captioning.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical character recognition (OCR) yields a significant performance improvement.

After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

[TFVisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.TFVisionEncoderDecoderModel) is a generic model class that will be instantiated as a transformer architecture with one of the base vision model classes of the library as encoder and another one of the base model classes as decoder when created with the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method for the encoder and [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) class method for the decoder.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/modeling_tf_vision_encoder_decoder.py#L486)

( pixel\_values: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: Optional\[Union\[Tuple, TFBaseModelOutput\]\] = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Nonelabels: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False\*\*kwargs ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or `tuple(tf.Tensor)`

The [TFVisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.TFVisionEncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, AutoTokenizer, TFVisionEncoderDecoderModel
>>> from PIL import Image
>>> import requests

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> decoder_tokenizer = AutoTokenizer.from_pretrained("gpt2")

>>> 
>>> model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "gpt2"
... )

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> img = Image.open(requests.get(url, stream=True).raw)

>>> 
>>> pixel_values = image_processor(images=img, return_tensors="tf").pixel_values  
>>> decoder_input_ids = decoder_tokenizer("Linda Davis", return_tensors="tf").input_ids  
>>> outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)

>>> 
>>> outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
>>> loss, logits = outputs.loss, outputs.logits

>>> 
>>> model.save_pretrained("vit-gpt2")
>>> model = TFVisionEncoderDecoderModel.from_pretrained("vit-gpt2")

>>> 
>>> generated = model.generate(pixel_values, decoder_start_token_id=model.config.decoder.bos_token_id)
```

#### from\_encoder\_decoder\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/modeling_tf_vision_encoder_decoder.py#L338)

( encoder\_pretrained\_model\_name\_or\_path: str = Nonedecoder\_pretrained\_model\_name\_or\_path: str = None\*model\_args\*\*kwargs )

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model checkpoints.

Example:

```
>>> from transformers import TFVisionEncoderDecoderModel

>>> 
>>> model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "bert-base-uncased"
... )
>>> 
>>> model.save_pretrained("./vit-bert")
>>> 
>>> model = TFVisionEncoderDecoderModel.from_pretrained("./vit-bert")
```

## FlaxVisionEncoderDecoderModel

### class transformers.FlaxVisionEncoderDecoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/modeling_flax_vision_encoder_decoder.py#L268)

( config: VisionEncoderDecoderConfiginput\_shape: typing.Optional\[typing.Tuple\] = Noneseed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

This class can be used to initialize an image-to-text-sequence model with any pretrained vision autoencoding model as the encoder and any pretrained text autoregressive model as the decoder. The encoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function and the decoder is loaded via [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) function. Cross-attention layers are automatically added to the decoder and should be fine-tuned on a downstream generative task, like image captioning.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks was shown in [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461) by Sascha Rothe, Shashi Narayan, Aliaksei Severyn. Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

Additionally, in [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) it is shown how leveraging large pretrained vision models for optical character recognition (OCR) yields a significant performance improvement.

After such a Vision-Encoder-Text-Decoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

[FlaxVisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.FlaxVisionEncoderDecoderModel) is a generic model class that will be instantiated as a transformer architecture with the module (flax.nn.Module) of one of the base vision model classes of the library as encoder module and another one as decoder module when created with the :meth_~transformers.FlaxAutoModel.from\_pretrained_ class method for the encoder and :meth_~transformers.FlaxAutoModelForCausalLM.from\_pretrained_ class method for the decoder.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/modeling_flax_vision_encoder_decoder.py#L598)

( pixel\_values: Arraydecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [FlaxVisionEncoderDecoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.FlaxVisionEncoderDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import FlaxVisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

>>> 
>>> tokenizer_output = AutoTokenizer.from_pretrained("gpt2")

>>> 
>>> model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "gpt2"
... )

>>> pixel_values = image_processor(images=image, return_tensors="np").pixel_values

>>> 
>>> model.config.eos_token_id = model.config.decoder.eos_token_id
>>> model.config.pad_token_id = model.config.eos_token_id

>>> 
>>> sequences = model.generate(pixel_values, num_beams=4, max_length=12).sequences

>>> captions = tokenizer_output.batch_decode(sequences, skip_special_tokens=True)
```

#### from\_encoder\_decoder\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_encoder_decoder/modeling_flax_vision_encoder_decoder.py#L723)

( encoder\_pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike, NoneType\] = Nonedecoder\_pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike, NoneType\] = None\*model\_args\*\*kwargs )

Instantiate an encoder and a decoder from one or two base classes of the library from pretrained model checkpoints.

Example:

```
>>> from transformers import FlaxVisionEncoderDecoderModel

>>> 
>>> model = FlaxVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "gpt2"
... )
>>> 
>>> model.save_pretrained("./vit-gpt2")
>>> 
>>> model = FlaxVisionEncoderDecoderModel.from_pretrained("./vit-gpt2")
```