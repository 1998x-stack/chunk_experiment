# Pegasus

[![Models](https://img.shields.io/badge/All_model_pages-pegasus-blueviolet)](https://huggingface.co/models?filter=pegasus) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/pegasus_paraphrase)

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=sshleifer&labels=&template=bug-report.md&title) and assign @patrickvonplaten.

## Overview

The Pegasus model was proposed in [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/pdf/1912.08777.pdf) by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu on Dec 18, 2019.

According to the abstract,

-   Pegasus’ pretraining task is intentionally similar to summarization: important sentences are removed/masked from an input document and are generated together as one output sequence from the remaining sentences, similar to an extractive summary.
-   Pegasus achieves SOTA summarization performance on all 12 downstream tasks, as measured by ROUGE and human eval.

This model was contributed by [sshleifer](https://huggingface.co/sshleifer). The Authors’ code can be found [here](https://github.com/google-research/pegasus).

Tips:

-   Sequence-to-sequence model with the same encoder-decoder model architecture as BART. Pegasus is pre-trained jointly on two self-supervised objective functions: Masked Language Modeling (MLM) and a novel summarization specific pretraining objective, called Gap Sentence Generation (GSG).
    
    -   MLM: encoder input tokens are randomly replaced by a mask tokens and have to be predicted by the encoder (like in BERT)
    -   GSG: whole encoder input sentences are replaced by a second mask token and fed to the decoder, but which has a causal mask to hide the future words like a regular auto-regressive transformer decoder.

## Checkpoints

All the [checkpoints](https://huggingface.co/models?search=pegasus) are fine-tuned for summarization, besides _pegasus-large_, whence the other checkpoints are fine-tuned:

-   Each checkpoint is 2.2 GB on disk and 568M parameters.
-   FP16 is not supported (help/ideas on this appreciated!).
-   Summarizing xsum in fp32 takes about 400ms/sample, with default parameters on a v100 GPU.
-   Full replication results and correctly pre-processed data can be found in this [Issue](https://github.com/huggingface/transformers/issues/6844#issue-689259666).
-   [Distilled checkpoints](https://huggingface.co/models?search=distill-pegasus) are described in this [paper](https://arxiv.org/abs/2010.13002).

### Examples

-   [Script](https://github.com/huggingface/transformers/tree/main/examples/research_projects/seq2seq-distillation/finetune_pegasus_xsum.sh) to fine-tune pegasus on the XSUM dataset. Data download instructions at [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md).
-   FP16 is not supported (help/ideas on this appreciated!).
-   The adafactor optimizer is recommended for pegasus fine-tuning.

## Implementation Notes

-   All models are transformer encoder-decoders with 16 layers in each component.
    
-   The implementation is completely inherited from [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration)
    
-   Some key configuration differences:
    
    -   static, sinusoidal position embeddings
    -   the model starts generating with pad\_token\_id (which has 0 token\_embedding) as the prefix.
    -   more beams are used (`num_beams=8`)
-   All pretrained pegasus checkpoints are the same besides three attributes: `tokenizer.model_max_length` (maximum input size), `max_length` (the maximum number of tokens to generate) and `length_penalty`.
    
-   The code to convert checkpoints trained in the author’s [repo](https://github.com/google-research/pegasus) can be found in `convert_pegasus_tf_to_pytorch.py`.
    

## Usage Example

```
>>> from transformers import PegasusForConditionalGeneration, PegasusTokenizer
>>> import torch

>>> src_text = [
...     """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
... ]

... model_name = "google/pegasus-xsum"
... device = "cuda" if torch.cuda.is_available() else "cpu"
... tokenizer = PegasusTokenizer.from_pretrained(model_name)
... model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
... batch = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt").to(device)
... translated = model.generate(**batch)
... tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
... assert (
...     tgt_text[0]
...     == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
... )
```

## Documentation resources

-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## PegasusConfig

### class transformers.PegasusConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/configuration_pegasus.py#L29)

( vocab\_size = 50265max\_position\_embeddings = 1024encoder\_layers = 12encoder\_ffn\_dim = 4096encoder\_attention\_heads = 16decoder\_layers = 12decoder\_ffn\_dim = 4096decoder\_attention\_heads = 16encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0use\_cache = Trueis\_encoder\_decoder = Trueactivation\_function = 'gelu'd\_model = 1024dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02decoder\_start\_token\_id = 0scale\_embedding = Falsepad\_token\_id = 0eos\_token\_id = 1forced\_eos\_token\_id = 1\*\*kwargs )

This is the configuration class to store the configuration of a [PegasusModel](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.PegasusModel). It is used to instantiate an PEGASUS model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the PEGASUS [google/pegasus-large](https://huggingface.co/google/pegasus-large) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import PegasusConfig, PegasusModel

>>> 
>>> configuration = PegasusConfig()

>>> 
>>> model = PegasusModel(configuration)

>>> 
>>> configuration = model.config
```

## PegasusTokenizer

warning: `add_tokens` does not work at the moment.

### class transformers.PegasusTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/tokenization_pegasus.py#L42)

( vocab\_filepad\_token = '<pad>'eos\_token = '</s>'unk\_token = '<unk>'mask\_token = '<mask\_2>'mask\_token\_sent = '<mask\_1>'additional\_special\_tokens = Noneoffset = 103sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None\*\*kwargs )

Construct a PEGASUS tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/tokenization_pegasus.py#L250)

( token\_ids\_0token\_ids\_1 = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequences for sequence classification tasks by concatenating and adding special tokens. A PEGASUS sequence has the following format, where `X` represents the sequence:

-   single sequence: `X </s>`
-   pair of sequences: `A B </s>` (not intended use)

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a separator.

Converts a sequence of tokens (string) in a single string.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/tokenization_pegasus.py#L239)

( token\_ids\_0: typing.Listtoken\_ids\_1: typing.Optional\[typing.List\] = Nonealready\_has\_special\_tokens: bool = False )

Get list where entries are \[1\] if a token is \[eos\] or \[pad\] else 0.

#### num\_special\_tokens\_to\_add

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/tokenization_pegasus.py#L229)

( pair = False )

Just EOS

## PegasusTokenizerFast

### class transformers.PegasusTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/tokenization_pegasus_fast.py#L51)

( vocab\_file = Nonetokenizer\_file = Nonepad\_token = '<pad>'eos\_token = '</s>'unk\_token = '<unk>'mask\_token = '<mask\_2>'mask\_token\_sent = '<mask\_1>'additional\_special\_tokens = Noneoffset = 103\*\*kwargs )

Construct a “fast” PEGASUS tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/tokenization_pegasus_fast.py#L183)

( token\_ids\_0token\_ids\_1 = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence by adding eos to the end. no bos token is added to the front.

-   single sequence: `X </s>`
-   pair of sequences: `A B </s>` (not intended use)

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/tokenization_pegasus_fast.py#L172)

( token\_ids\_0: typing.Listtoken\_ids\_1: typing.Optional\[typing.List\] = Nonealready\_has\_special\_tokens: bool = False )

Get list where entries are \[1\] if a token is \[eos\] or \[pad\] else 0.

## PegasusModel

### class transformers.PegasusModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_pegasus.py#L1159)

( config: PegasusConfig )

Parameters

-   **config** ([PegasusConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.PegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare PEGASUS Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_pegasus.py#L1211)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [PegasusModel](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.PegasusModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, PegasusModel

>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
>>> model = PegasusModel.from_pretrained("google/pegasus-large")

>>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
>>> decoder_inputs = tokenizer("Studies show that", return_tensors="pt")
>>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 4, 1024]
```

## PegasusForConditionalGeneration

### class transformers.PegasusForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_pegasus.py#L1310)

( config: PegasusConfig )

Parameters

-   **config** ([PegasusConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.PegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The PEGASUS Model with a language modeling head. Can be used for summarization. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_pegasus.py#L1373)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.Tensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [PegasusForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.PegasusForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```
>>> from transformers import AutoTokenizer, PegasusForConditionalGeneration

>>> model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

>>> ARTICLE_TO_SUMMARIZE = (
...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
... )
>>> inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="pt")

>>> 
>>> summary_ids = model.generate(inputs["input_ids"])
>>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"California's largest electricity provider has turned off power to hundreds of thousands of customers."
```

## PegasusForCausalLM

### class transformers.PegasusForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_pegasus.py#L1513)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_pegasus.py#L1568)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, PegasusForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
>>> model = PegasusForCausalLM.from_pretrained("google/pegasus-large", add_cross_attention=False)
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
>>> list(logits.shape) == expected_shape
True
```

## TFPegasusModel

### class transformers.TFPegasusModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_tf_pegasus.py#L1166)

( \*args\*\*kwargs )

Parameters

-   **config** ([PegasusConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.PegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare PEGASUS Model outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_tf_pegasus.py#L1178)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_position\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Nonecross\_attn\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: Optional\[Union\[Tuple, TFBaseModelOutput\]\] = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False\*\*kwargs ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput) or `tuple(tf.Tensor)`

The [TFPegasusModel](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.TFPegasusModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFPegasusModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
>>> model = TFPegasusModel.from_pretrained("google/pegasus-large")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFPegasusForConditionalGeneration

### class transformers.TFPegasusForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_tf_pegasus.py#L1271)

( \*args\*\*kwargs )

Parameters

-   **config** ([PegasusConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.PegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The PEGASUS Model with a language modeling head. Can be used for summarization. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_tf_pegasus.py#L1309)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_position\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Nonecross\_attn\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: Optional\[TFBaseModelOutput\] = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or `tuple(tf.Tensor)`

The [TFPegasusForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.TFPegasusForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```
>>> from transformers import AutoTokenizer, TFPegasusForConditionalGeneration

>>> model = TFPegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

>>> ARTICLE_TO_SUMMARIZE = (
...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
... )
>>> inputs = tokenizer(ARTICLE_TO_SUMMARIZE, max_length=1024, return_tensors="tf")

>>> 
>>> summary_ids = model.generate(input_ids)
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
```

## FlaxPegasusModel

### class transformers.FlaxPegasusModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_flax_pegasus.py#L1223)

( config: PegasusConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([PegasusConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.PegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare Pegasus Model transformer outputting raw hidden-states without any specific head on top. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_flax_pegasus.py#L1158)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The `FlaxPegasusPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxPegasusModel

>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
>>> model = FlaxPegasusModel.from_pretrained("google/pegasus-large")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_flax_pegasus.py#L981)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

>>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_flax_pegasus.py#L1044)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

>>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> last_decoder_hidden_states = outputs.last_hidden_state
```

## FlaxPegasusForConditionalGeneration

### class transformers.FlaxPegasusForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_flax_pegasus.py#L1308)

( config: PegasusConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([PegasusConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus#transformers.PegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The PEGASUS Model with a language modeling head. Can be used for summarization. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_flax_pegasus.py#L1158)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The `FlaxPegasusPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```python
>>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

>>> model = FlaxPegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
>>> tokenizer = AutoTokenizer.from_pretrained('google/pegasus-large')

>>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='np')

>>> 
>>> summary_ids = model.generate(inputs['input_ids']).sequences
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
```

Mask filling example:

```
>>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
>>> TXT = "My friends are <mask> but they eat too many carbs."

>>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
>>> input_ids = tokenizer([TXT], return_tensors="np")["input_ids"]
>>> logits = model(input_ids).logits

>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = jax.nn.softmax(logits[0, masked_index], axis=0)
>>> values, predictions = jax.lax.top_k(probs)

>>> tokenizer.decode(predictions).split()
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_flax_pegasus.py#L981)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

>>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus/modeling_flax_pegasus.py#L1312)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonedeterministic: bool = Trueparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoTokenizer, FlaxPegasusForConditionalGeneration

>>> model = FlaxPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="np")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> logits = outputs.logits
```