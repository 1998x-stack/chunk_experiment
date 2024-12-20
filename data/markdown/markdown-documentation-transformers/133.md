# CTRL

[![Models](https://img.shields.io/badge/All_model_pages-ctrl-blueviolet)](https://huggingface.co/models?filter=ctrl) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/tiny-ctrl)

## Overview

CTRL model was proposed in [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) by Nitish Shirish Keskar_, Bryan McCann_, Lav R. Varshney, Caiming Xiong and Richard Socher. It’s a causal (unidirectional) transformer pre-trained using language modeling on a very large corpus of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

The abstract from the paper is the following:

_Large-scale language models show promising text generation capabilities, but users cannot easily control particular aspects of the generated text. We release CTRL, a 1.63 billion-parameter conditional transformer language model, trained to condition on control codes that govern style, content, and task-specific behavior. Control codes were derived from structure that naturally co-occurs with raw text, preserving the advantages of unsupervised learning while providing more explicit control over text generation. These codes also allow CTRL to predict which parts of the training data are most likely given a sequence. This provides a potential method for analyzing large amounts of data via model-based source attribution._

Tips:

-   CTRL makes use of control codes to generate text: it requires generations to be started by certain words, sentences or links to generate coherent text. Refer to the [original implementation](https://github.com/salesforce/ctrl) for more information.
-   CTRL is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
-   CTRL was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. Leveraging this feature allows CTRL to generate syntactically coherent text as it can be observed in the _run\_generation.py_ example script.
-   The PyTorch models can take the `past_key_values` as input, which is the previously computed key/value attention pairs. TensorFlow models accepts `past` as input. Using the `past_key_values` value prevents the model from re-computing pre-computed values in the context of text generation. See the [`forward`](model_doc/ctrl#transformers.CTRLModel.forward) method for more information on the usage of this argument.

This model was contributed by [keskarnitishr](https://huggingface.co/keskarnitishr). The original code can be found [here](https://github.com/salesforce/ctrl).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Causal language modeling task guide](../tasks/language_modeling)

## CTRLConfig

### class transformers.CTRLConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/configuration_ctrl.py#L28)

( vocab\_size = 246534n\_positions = 256n\_embd = 1280dff = 8192n\_layer = 48n\_head = 16resid\_pdrop = 0.1embd\_pdrop = 0.1layer\_norm\_epsilon = 1e-06initializer\_range = 0.02use\_cache = True\*\*kwargs )

This is the configuration class to store the configuration of a [CTRLModel](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLModel) or a [TFCTRLModel](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.TFCTRLModel). It is used to instantiate a CTRL model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the [Salesforce/ctrl](https://huggingface.co/Salesforce/ctrl) architecture from SalesForce.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import CTRLConfig, CTRLModel

>>> 
>>> configuration = CTRLConfig()

>>> 
>>> model = CTRLModel(configuration)

>>> 
>>> configuration = model.config
```

## CTRLTokenizer

### class transformers.CTRLTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/tokenization_ctrl.py#L119)

( vocab\_filemerges\_fileunk\_token = '<unk>'\*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

Construct a CTRL tokenizer. Based on Byte-Pair-Encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/tokenization_ctrl.py#L226)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## CTRLModel

### class transformers.CTRLModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_ctrl.py#L319)

( config )

Parameters

-   **config** ([CTRLConfig](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_ctrl.py#L352)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

The [CTRLModel](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, CTRLModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLModel.from_pretrained("Salesforce/ctrl")

>>> 
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 5, 1280]
```

## CTRLLMHeadModel

### class transformers.CTRLLMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_ctrl.py#L512)

( config )

Parameters

-   **config** ([CTRLConfig](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_ctrl.py#L536)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

The [CTRLLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, CTRLLMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLLMHeadModel.from_pretrained("Salesforce/ctrl")

>>> 
>>> inputs = tokenizer("Wikipedia The llama is", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> sequence_ids = model.generate(inputs["input_ids"])
>>> sequences = tokenizer.batch_decode(sequence_ids)
>>> sequences
['Wikipedia The llama is a member of the family Bovidae. It is native to the Andes of Peru,']

>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> round(outputs.loss.item(), 2)
9.21

>>> list(outputs.logits.shape)
[1, 5, 246534]
```

## CTRLForSequenceClassification

### class transformers.CTRLForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_ctrl.py#L654)

( config )

Parameters

-   **config** ([CTRLConfig](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CTRL Model transformer with a sequence classification head on top (linear layer). [CTRLForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_ctrl.py#L664)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [CTRLForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, CTRLForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl")

>>> 
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'LABEL_0'
```

```
>>> import torch

>>> torch.manual_seed(42)
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

>>> labels = torch.tensor(1)
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
0.35
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, CTRLForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLForSequenceClassification.from_pretrained(
...     "Salesforce/ctrl", problem_type="multi_label_classification"
... )

>>> 
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'LABEL_0'
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

>>> num_labels = len(model.config.id2label)
>>> labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
...     torch.float
... )
>>> loss = model(**inputs, labels=labels).loss
>>> loss.backward()
```

## TFCTRLModel

### class transformers.TFCTRLModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_tf_ctrl.py#L523)

( \*args\*\*kwargs )

Parameters

-   **config** ([CTRLConfig](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_tf_ctrl.py#L528)

( input\_ids: TFModelInputType | None = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPast) or `tuple(tf.Tensor)`

The [TFCTRLModel](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.TFCTRLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFCTRLModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = TFCTRLModel.from_pretrained("Salesforce/ctrl")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFCTRLLMHeadModel

### class transformers.TFCTRLLMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_tf_ctrl.py#L596)

( \*args\*\*kwargs )

Parameters

-   **config** ([CTRLConfig](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_tf_ctrl.py#L648)

( input\_ids: TFModelInputType | None = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFCausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutputWithPast) or `tuple(tf.Tensor)`

The [TFCTRLLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.TFCTRLLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFCTRLLMHeadModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = TFCTRLLMHeadModel.from_pretrained("Salesforce/ctrl")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

## TFCTRLForSequenceClassification

### class transformers.TFCTRLForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_tf_ctrl.py#L729)

( \*args\*\*kwargs )

Parameters

-   **config** ([CTRLConfig](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.CTRLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CTRL Model transformer with a sequence classification head on top (linear layer).

[TFCTRLForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.TFCTRLForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-1, GPT-2) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/ctrl/modeling_tf_ctrl.py#L749)

( input\_ids: TFModelInputType | None = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonetoken\_type\_ids: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or `tuple(tf.Tensor)`

The [TFCTRLForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/ctrl#transformers.TFCTRLForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFCTRLForSequenceClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = TFCTRLForSequenceClassification.from_pretrained("Salesforce/ctrl")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

>>> logits = model(**inputs).logits

>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TFCTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

>>> labels = tf.constant(1)
>>> loss = model(**inputs, labels=labels).loss
```