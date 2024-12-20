# Blenderbot Small

Note that [BlenderbotSmallModel](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel) and [BlenderbotSmallForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForConditionalGeneration) are only used in combination with the checkpoint [facebook/blenderbot-90M](https://huggingface.co/facebook/blenderbot-90M). Larger Blenderbot checkpoints should instead be used with [BlenderbotModel](/docs/transformers/v4.34.0/en/model_doc/blenderbot#transformers.BlenderbotModel) and [BlenderbotForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blenderbot#transformers.BlenderbotForConditionalGeneration)

## Overview

The Blender chatbot model was proposed in [Recipes for building an open-domain chatbot](https://arxiv.org/pdf/2004.13637.pdf) Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu, Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.

The abstract of the paper is the following:

_Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that scaling neural models in the number of parameters and the size of the data they are trained on gives improved results, we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent persona. We show that large scale models can learn these skills when given appropriate training data and choice of generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing failure cases of our models._

Tips:

-   Blenderbot Small is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten). The authors’ code can be found [here](https://github.com/facebookresearch/ParlAI).

## Documentation resources

-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## BlenderbotSmallConfig

### class transformers.BlenderbotSmallConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/configuration_blenderbot_small.py#L36)

( vocab\_size = 50265max\_position\_embeddings = 512encoder\_layers = 8encoder\_ffn\_dim = 2048encoder\_attention\_heads = 16decoder\_layers = 8decoder\_ffn\_dim = 2048decoder\_attention\_heads = 16encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0use\_cache = Trueis\_encoder\_decoder = Trueactivation\_function = 'gelu'd\_model = 512dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02decoder\_start\_token\_id = 1scale\_embedding = Falsepad\_token\_id = 0bos\_token\_id = 1eos\_token\_id = 2forced\_eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [BlenderbotSmallModel](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel). It is used to instantiate an BlenderbotSmall model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BlenderbotSmall [facebook/blenderbot\_small-90M](https://huggingface.co/facebook/blenderbot_small-90M) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BlenderbotSmallConfig, BlenderbotSmallModel

>>> 
>>> configuration = BlenderbotSmallConfig()

>>> 
>>> model = BlenderbotSmallModel(configuration)

>>> 
>>> configuration = model.config
```

## BlenderbotSmallTokenizer

### class transformers.BlenderbotSmallTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/tokenization_blenderbot_small.py#L69)

( vocab\_filemerges\_filebos\_token = '\_\_start\_\_'eos\_token = '\_\_end\_\_'unk\_token = '\_\_unk\_\_'pad\_token = '\_\_null\_\_'\*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — File containing the vocabulary.
-   **merges\_file** (`str`) — Path to the merges file.
-   **bos\_token** (`str`, _optional_, defaults to `"__start__"`) — The beginning of sentence token.
-   **eos\_token** (`str`, _optional_, defaults to `"__end__"`) — The end of sentence token.
-   **unk\_token** (`str`, _optional_, defaults to `"__unk__"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"__pad__"`) — The token used for padding, for example when batching sequences of different lengths. \*\*kwargs — Additional keyword arguments passed along to [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)

Constructs a Blenderbot-90M tokenizer based on BPE (Byte-Pair-Encoding)

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to the superclass for more information regarding methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3325)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — The first tokenized sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — The second tokenized sequence.

The model input with special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens.

This implementation does not add special tokens and this method should be overridden in a subclass.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils.py#L904)

( token\_ids\_0: typing.Listtoken\_ids\_1: typing.Optional\[typing.List\] = Nonealready\_has\_special\_tokens: bool = False ) → A list of integers in the range \[0, 1\]

Parameters

-   **token\_ids\_0** (`List[int]`) — List of ids of the first sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — List of ids of the second sequence.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

Returns

A list of integers in the range \[0, 1\]

1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/tokenization_blenderbot_small.py#L210)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## BlenderbotSmallTokenizerFast

### class transformers.BlenderbotSmallTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/tokenization_blenderbot_small_fast.py#L52)

( vocab\_file = Nonemerges\_file = Noneunk\_token = '<|endoftext|>'bos\_token = '<|endoftext|>'eos\_token = '<|endoftext|>'add\_prefix\_space = Falsetrim\_offsets = True\*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.

Construct a “fast” BlenderbotSmall tokenizer (backed by HuggingFace’s _tokenizers_ library).

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/tokenization_blenderbot_small_fast.py#L98)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. BlenderbotSmall does not make use of token type ids, therefore a list of zeros is returned.

## BlenderbotSmallModel

### class transformers.BlenderbotSmallModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1099)

( config: BlenderbotSmallConfig )

Parameters

-   **config** ([BlenderbotSmallConfig](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare BlenderbotSmall Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1128)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Union\[typing.Tuple, transformers.modeling\_outputs.BaseModelOutput, NoneType\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [BlenderbotSmallModel](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BlenderbotSmallModel

>>> model = BlenderbotSmallModel.from_pretrained("facebook/blenderbot_small-90M")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

>>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
>>> decoder_inputs = tokenizer("Studies show that", return_tensors="pt")  
>>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 3, 512]
```

## BlenderbotSmallForConditionalGeneration

### class transformers.BlenderbotSmallForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1227)

( config: BlenderbotSmallConfig )

Parameters

-   **config** ([BlenderbotSmallConfig](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The BlenderbotSmall Model with a language modeling head. Can be used for summarization. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1267)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Union\[typing.Tuple, transformers.modeling\_outputs.BaseModelOutput, NoneType\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [BlenderbotSmallForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Conversation example:

```
>>> from transformers import AutoTokenizer, BlenderbotSmallForConditionalGeneration

>>> mname = "facebook/blenderbot_small-90M"
>>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
>>> tokenizer = AutoTokenizer.from_pretrained(mname)
>>> UTTERANCE = "My friends are cool but they eat too many carbs."
>>> print("Human: ", UTTERANCE)
Human:  My friends are cool but they eat too many carbs.

>>> inputs = tokenizer([UTTERANCE], return_tensors="pt")
>>> reply_ids = model.generate(**inputs)
>>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
Bot:  what kind of carbs do they eat? i don't know much about carbs.

>>> REPLY = "I'm not sure"
>>> print("Human: ", REPLY)
Human: I'm not sure

>>> NEXT_UTTERANCE = (
...     "My friends are cool but they eat too many carbs.__end__ __start__what kind of carbs do they eat? "
...     "i don't know much about carbs__end__ "
...     "__start__ I'm not sure."
... )
>>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors="pt")
>>> next_reply_ids = model.generate(**inputs)
>>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
Bot:  they eat a lot of carbs. carbs are high in fat, protein, and fats.
```

## BlenderbotSmallForCausalLM

### class transformers.BlenderbotSmallForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1404)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_blenderbot_small.py#L1437)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, BlenderbotSmallForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
>>> model = BlenderbotSmallForCausalLM.from_pretrained(
...     "facebook/blenderbot_small-90M", add_cross_attention=False
... )
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
>>> list(logits.shape) == expected_shape
True
```

## TFBlenderbotSmallModel

### class transformers.TFBlenderbotSmallModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_tf_blenderbot_small.py#L1130)

( \*args\*\*kwargs )

Parameters

-   **config** ([BlenderbotSmallConfig](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare BLENDERBOT\_SMALL Model outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_tf_blenderbot_small.py#L1142)

( input\_ids: tf.Tensor | None = Noneattention\_mask: tf.Tensor | None = Nonedecoder\_input\_ids: tf.Tensor | None = Nonedecoder\_attention\_mask: tf.Tensor | None = Nonedecoder\_position\_ids: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Nonedecoder\_head\_mask: tf.Tensor | None = Nonecross\_attn\_head\_mask: tf.Tensor | None = Noneencoder\_outputs: Optional\[Union\[Tuple, TFBaseModelOutput\]\] = Nonepast\_key\_values: List\[tf.Tensor\] | None = Noneinputs\_embeds: tf.Tensor | None = Nonedecoder\_inputs\_embeds: tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False\*\*kwargs ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput) or `tuple(tf.Tensor)`

The [TFBlenderbotSmallModel](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.TFBlenderbotSmallModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFBlenderbotSmallModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
>>> model = TFBlenderbotSmallModel.from_pretrained("facebook/blenderbot_small-90M")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFBlenderbotSmallForConditionalGeneration

### class transformers.TFBlenderbotSmallForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_tf_blenderbot_small.py#L1235)

( \*args\*\*kwargs )

Parameters

-   **config** ([BlenderbotSmallConfig](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The BLENDERBOT\_SMALL Model with a language modeling head. Can be used for summarization. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_tf_blenderbot_small.py#L1273)

( input\_ids: tf.Tensor | None = Noneattention\_mask: tf.Tensor | None = Nonedecoder\_input\_ids: tf.Tensor | None = Nonedecoder\_attention\_mask: tf.Tensor | None = Nonedecoder\_position\_ids: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Nonedecoder\_head\_mask: tf.Tensor | None = Nonecross\_attn\_head\_mask: tf.Tensor | None = Noneencoder\_outputs: Optional\[TFBaseModelOutput\] = Nonepast\_key\_values: List\[tf.Tensor\] | None = Noneinputs\_embeds: tf.Tensor | None = Nonedecoder\_inputs\_embeds: tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or `tuple(tf.Tensor)`

The [TFBlenderbotSmallForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.TFBlenderbotSmallForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Conversation example::

```
>>> from transformers import AutoTokenizer, TFBlenderbotSmallForConditionalGeneration

>>> mname = "facebook/blenderbot_small-90M"
>>> model = BlenderbotSmallForConditionalGeneration.from_pretrained(mname)
>>> tokenizer = AutoTokenizer.from_pretrained(mname)

>>> UTTERANCE = "My friends are cool but they eat too many carbs."
>>> print("Human: ", UTTERANCE)
>>> inputs = tokenizer([UTTERANCE], return_tensors="tf")

>>> reply_ids = model.generate(**inputs)
>>> print("Bot: ", tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0])
what kind of carbs do they eat? i don't know much about carbs.

>>> REPLY = "I'm not sure"
>>> print("Human: ", REPLY)
>>> NEXT_UTTERANCE = (
...     "My friends are cool but they eat too many carbs.</s> "
...     "<s>what kind of carbs do they eat? i don't know much about carbs.</s> "
...     "<s>I'm not sure."
... )

>>> inputs = tokenizer([NEXT_UTTERANCE], return_tensors="tf")
>>> inputs.pop("token_type_ids")
>>> next_reply_ids = model.generate(**inputs)
>>> print("Bot: ", tokenizer.batch_decode(next_reply_ids, skip_special_tokens=True)[0])
```

## FlaxBlenderbotSmallModel

### class transformers.FlaxBlenderbotSmallModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1213)

( config: BlenderbotSmallConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BlenderbotSmallConfig](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare BlenderbotSmall Model transformer outputting raw hidden-states without any specific head on top. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1149)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxBlenderbotSmallModel

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
>>> model = FlaxBlenderbotSmallModel.from_pretrained("facebook/blenderbot_small-90M")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L970)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

>>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1033)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

>>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> last_decoder_hidden_states = outputs.last_hidden_state
```

## FlaxBlenderbotForConditionalGeneration

### class transformers.FlaxBlenderbotSmallForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1299)

( config: BlenderbotSmallConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BlenderbotSmallConfig](/docs/transformers/v4.34.0/en/model_doc/blenderbot-small#transformers.BlenderbotSmallConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The BLENDERBOT\_SMALL Model with a language modeling head. Can be used for summarization. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1149)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The `FlaxBlenderbotSmallPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```
>>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

>>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

>>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="np")

>>> 
>>> summary_ids = model.generate(inputs["input_ids"]).sequences
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
```

Mask filling example:

```
>>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
>>> TXT = "My friends are <mask> but they eat too many carbs."

>>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
>>> input_ids = tokenizer([TXT], return_tensors="np")["input_ids"]
>>> logits = model(input_ids).logits

>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = jax.nn.softmax(logits[0, masked_index], axis=0)
>>> values, predictions = jax.lax.top_k(probs)

>>> tokenizer.decode(predictions).split()
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L970)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

>>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/blenderbot_small/modeling_flax_blenderbot_small.py#L1303)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonedeterministic: bool = Trueparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoTokenizer, FlaxBlenderbotSmallForConditionalGeneration

>>> model = FlaxBlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> logits = outputs.logits
```