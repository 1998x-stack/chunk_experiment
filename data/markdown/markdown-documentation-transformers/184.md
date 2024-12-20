# Hubert

## Overview

Hubert was proposed in [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.

The abstract from the paper is the following:

_Self-supervised approaches for speech representation learning are challenged by three unique problems: (1) there are multiple sound units in each input utterance, (2) there is no lexicon of input sound units during the pre-training phase, and (3) sound units have variable lengths with no explicit segmentation. To deal with these three problems, we propose the Hidden-Unit BERT (HuBERT) approach for self-supervised speech representation learning, which utilizes an offline clustering step to provide aligned target labels for a BERT-like prediction loss. A key ingredient of our approach is applying the prediction loss over the masked regions only, which forces the model to learn a combined acoustic and language model over the continuous inputs. HuBERT relies primarily on the consistency of the unsupervised clustering step rather than the intrinsic quality of the assigned cluster labels. Starting with a simple k-means teacher of 100 clusters, and using two iterations of clustering, the HuBERT model either matches or improves upon the state-of-the-art wav2vec 2.0 performance on the Librispeech (960h) and Libri-light (60,000h) benchmarks with 10min, 1h, 10h, 100h, and 960h fine-tuning subsets. Using a 1B parameter model, HuBERT shows up to 19% and 13% relative WER reduction on the more challenging dev-other and test-other evaluation subsets._

Tips:

-   Hubert is a speech model that accepts a float array corresponding to the raw waveform of the speech signal.
-   Hubert model was fine-tuned using connectionist temporal classification (CTC) so the model output has to be decoded using [Wav2Vec2CTCTokenizer](/docs/transformers/v4.34.0/en/model_doc/wav2vec2#transformers.Wav2Vec2CTCTokenizer).

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).

## Documentation resources

-   [Audio classification task guide](../tasks/audio_classification)
-   [Automatic speech recognition task guide](../tasks/asr)

## HubertConfig

### class transformers.HubertConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/configuration_hubert.py#L32)

( vocab\_size = 32hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout = 0.1activation\_dropout = 0.1attention\_dropout = 0.1feat\_proj\_layer\_norm = Truefeat\_proj\_dropout = 0.0final\_dropout = 0.1layerdrop = 0.1initializer\_range = 0.02layer\_norm\_eps = 1e-05feat\_extract\_norm = 'group'feat\_extract\_activation = 'gelu'conv\_dim = (512, 512, 512, 512, 512, 512, 512)conv\_stride = (5, 2, 2, 2, 2, 2, 2)conv\_kernel = (10, 3, 3, 3, 3, 2, 2)conv\_bias = Falsenum\_conv\_pos\_embeddings = 128num\_conv\_pos\_embedding\_groups = 16do\_stable\_layer\_norm = Falseapply\_spec\_augment = Truemask\_time\_prob = 0.05mask\_time\_length = 10mask\_time\_min\_masks = 2mask\_feature\_prob = 0.0mask\_feature\_length = 10mask\_feature\_min\_masks = 0ctc\_loss\_reduction = 'sum'ctc\_zero\_infinity = Falseuse\_weighted\_layer\_sum = Falseclassifier\_proj\_size = 256pad\_token\_id = 0bos\_token\_id = 1eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [HubertModel](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertModel). It is used to instantiate an Hubert model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Hubert [facebook/hubert-base-ls960](https://huggingface.co/facebook/hubert-base-ls960) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import HubertModel, HubertConfig

>>> 
>>> configuration = HubertConfig()

>>> 
>>> model = HubertModel(configuration)

>>> 
>>> configuration = model.config
```

## HubertModel

### class transformers.HubertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_hubert.py#L991)

( config: HubertConfig )

Parameters

-   **config** ([HubertConfig](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Hubert Model transformer outputting raw hidden-states without any specific head on top. Hubert was proposed in [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_hubert.py#L1056)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Nonemask\_time\_indices: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

The [HubertModel](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, HubertModel
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
>>> model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  
>>> hidden_states = model(input_values).last_hidden_state
```

## HubertForCTC

### class transformers.HubertForCTC

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_hubert.py#L1135)

( configtarget\_lang: typing.Optional\[str\] = None )

Parameters

-   **config** ([HubertConfig](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Hubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC). Hubert was proposed in [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_hubert.py#L1207)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

The [HubertForCTC](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, HubertForCTC
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
>>> model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

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
22.68
```

## HubertForSequenceClassification

### class transformers.HubertForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_hubert.py#L1295)

( config )

Parameters

-   **config** ([HubertConfig](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Hubert Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB Keyword Spotting.

Hubert was proposed in [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_hubert.py#L1340)

( input\_values: typing.Optional\[torch.Tensor\]attention\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.Tensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [HubertForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoFeatureExtractor, HubertForSequenceClassification
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")
>>> model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks")

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
8.53
```

## TFHubertModel

### class transformers.TFHubertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_tf_hubert.py#L1279)

( \*args\*\*kwargs )

Parameters

-   **config** ([HubertConfig](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare TFHubert Model transformer outputing raw hidden-states without any specific head on top.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_tf_hubert.py#L1285)

( input\_values: tf.Tensorattention\_mask: tf.Tensor | None = Nonetoken\_type\_ids: tf.Tensor | None = Noneposition\_ids: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Noneinputs\_embeds: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutput) or `tuple(tf.Tensor)`

The [TFHubertModel](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.TFHubertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoProcessor, TFHubertModel
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
>>> model = TFHubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


>>> def map_to_array(batch):
...     speech, _ = sf.read(batch["file"])
...     batch["speech"] = speech
...     return batch


>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.map(map_to_array)

>>> input_values = processor(ds["speech"][0], return_tensors="tf").input_values  
>>> hidden_states = model(input_values).last_hidden_state
```

## TFHubertForCTC

### class transformers.TFHubertForCTC

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_tf_hubert.py#L1353)

( \*args\*\*kwargs )

Parameters

-   **config** ([HubertConfig](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.HubertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

TFHubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/hubert/modeling_tf_hubert.py#L1380)

( input\_values: tf.Tensorattention\_mask: tf.Tensor | None = Nonetoken\_type\_ids: tf.Tensor | None = Noneposition\_ids: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Noneinputs\_embeds: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Nonelabels: tf.Tensor | None = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFCausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutput) or `tuple(tf.Tensor)`

The [TFHubertForCTC](/docs/transformers/v4.34.0/en/model_doc/hubert#transformers.TFHubertForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import tensorflow as tf
>>> from transformers import AutoProcessor, TFHubertForCTC
>>> from datasets import load_dataset
>>> import soundfile as sf

>>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
>>> model = TFHubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")


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
>>> labels = processor(text=transcription, return_tensors="tf").input_values

>>> loss = model(input_values, labels=labels).loss
```