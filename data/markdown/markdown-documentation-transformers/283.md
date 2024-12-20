# Transformer XL

[![Models](https://img.shields.io/badge/All_model_pages-transfo--xl-blueviolet)](https://huggingface.co/models?filter=transfo-xl) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/transfo-xl-wt103)

## Overview

The Transformer-XL model was proposed in [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov. It’s a causal (uni-directional) transformer with relative positioning (sinusoïdal) embeddings which can reuse previously computed hidden-states to attend to longer context (memory). This model also uses adaptive softmax inputs and outputs (tied).

The abstract from the paper is the following:

_Transformers have a potential of learning longer-term dependency, but are limited by a fixed-length context in the setting of language modeling. We propose a novel neural architecture Transformer-XL that enables learning dependency beyond a fixed length without disrupting temporal coherence. It consists of a segment-level recurrence mechanism and a novel positional encoding scheme. Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem. As a result, Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation. Notably, we improve the state-of-the-art results of bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn Treebank (without finetuning). When trained only on WikiText-103, Transformer-XL manages to generate reasonably coherent, novel text articles with thousands of tokens._

Tips:

-   Transformer-XL uses relative sinusoidal positional embeddings. Padding can be done on the left or on the right. The original implementation trains on SQuAD with padding on the left, therefore the padding defaults are set to left.
-   Transformer-XL is one of the few models that has no sequence length limit.
-   Same as a regular GPT model, but introduces a recurrence mechanism for two consecutive segments (similar to a regular RNNs with two consecutive inputs). In this context, a segment is a number of consecutive tokens (for instance 512) that may span across multiple documents, and segments are fed in order to the model.
-   Basically, the hidden states of the previous segment are concatenated to the current input to compute the attention scores. This allows the model to pay attention to information that was in the previous segment as well as the current one. By stacking multiple attention layers, the receptive field can be increased to multiple previous segments.
-   This changes the positional embeddings to positional relative embeddings (as the regular positional embeddings would give the same results in the current input and the current hidden state at a given position) and needs to make some adjustments in the way attention scores are computed.

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://github.com/kimiyoung/transformer-xl).

TransformerXL does **not** work with _torch.nn.DataParallel_ due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035)

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Causal language modeling task guide](../tasks/language_modeling)

## TransfoXLConfig

### class transformers.TransfoXLConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/configuration_transfo_xl.py#L29)

( vocab\_size = 267735 cutoffs = \[20000, 40000, 200000\] d\_model = 1024 d\_embed = 1024 n\_head = 16 d\_head = 64 d\_inner = 4096 div\_val = 4 pre\_lnorm = False n\_layer = 18 mem\_len = 1600 clamp\_len = 1000 same\_length = True proj\_share\_all\_but\_first = True attn\_type = 0 sample\_softmax = -1 adaptive = True dropout = 0.1 dropatt = 0.0 untie\_r = True init = 'normal' init\_range = 0.01 proj\_init\_std = 0.01 init\_std = 0.02 layer\_norm\_epsilon = 1e-05 eos\_token\_id = 0 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 267735) — Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [TransfoXLModel](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLModel) or [TFTransfoXLModel](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TFTransfoXLModel).
-   **cutoffs** (`List[int]`, _optional_, defaults to `[20000, 40000, 200000]`) — Cutoffs for the adaptive softmax.
-   **d\_model** (`int`, _optional_, defaults to 1024) — Dimensionality of the model’s hidden states.
-   **d\_embed** (`int`, _optional_, defaults to 1024) — Dimensionality of the embeddings
-   **n\_head** (`int`, _optional_, defaults to 16) — Number of attention heads for each attention layer in the Transformer encoder.
-   **d\_head** (`int`, _optional_, defaults to 64) — Dimensionality of the model’s heads.
-   **d\_inner** (`int`, _optional_, defaults to 4096) — Inner dimension in FF
-   **div\_val** (`int`, _optional_, defaults to 4) — Divident value for adapative input and softmax
-   **pre\_lnorm** (`boolean`, _optional_, defaults to `False`) — Whether or not to apply LayerNorm to the input instead of the output in the blocks.
-   **n\_layer** (`int`, _optional_, defaults to 18) — Number of hidden layers in the Transformer encoder.
-   **mem\_len** (`int`, _optional_, defaults to 1600) — Length of the retained previous heads.
-   **clamp\_len** (`int`, _optional_, defaults to 1000) — Use the same pos embeddings after clamp\_len.
-   **same\_length** (`boolean`, _optional_, defaults to `True`) — Whether or not to use the same attn length for all tokens
-   **proj\_share\_all\_but\_first** (`boolean`, _optional_, defaults to `True`) — True to share all but first projs, False not to share.
-   **attn\_type** (`int`, _optional_, defaults to 0) — Attention type. 0 for Transformer-XL, 1 for Shaw et al, 2 for Vaswani et al, 3 for Al Rfou et al.
-   **sample\_softmax** (`int`, _optional_, defaults to -1) — Number of samples in the sampled softmax.
-   **adaptive** (`boolean`, _optional_, defaults to `True`) — Whether or not to use adaptive softmax.
-   **dropout** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **dropatt** (`float`, _optional_, defaults to 0) — The dropout ratio for the attention probabilities.
-   **untie\_r** (`boolean`, _optional_, defaults to `True`) — Whether ot not to untie relative position biases.
-   **init** (`str`, _optional_, defaults to `"normal"`) — Parameter initializer to use.
-   **init\_range** (`float`, _optional_, defaults to 0.01) — Parameters initialized by U(-init\_range, init\_range).
-   **proj\_init\_std** (`float`, _optional_, defaults to 0.01) — Parameters initialized by N(0, init\_std)
-   **init\_std** (`float`, _optional_, defaults to 0.02) — Parameters initialized by N(0, init\_std)
-   **layer\_norm\_epsilon** (`float`, _optional_, defaults to 1e-5) — The epsilon to use in the layer normalization layers

This is the configuration class to store the configuration of a [TransfoXLModel](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLModel) or a [TFTransfoXLModel](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TFTransfoXLModel). It is used to instantiate a Transformer-XL model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the TransfoXL [transfo-xl-wt103](https://huggingface.co/transfo-xl-wt103) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import TransfoXLConfig, TransfoXLModel

>>> 
>>> configuration = TransfoXLConfig()

>>> 
>>> model = TransfoXLModel(configuration)

>>> 
>>> configuration = model.config
```

## TransfoXLTokenizer

### class transformers.TransfoXLTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/tokenization_transfo_xl.py#L123)

( special = None min\_freq = 0 max\_size = None lower\_case = False delimiter = None vocab\_file = None pretrained\_vocab\_file: str = None never\_split = None unk\_token = '<unk>' eos\_token = '<eos>' additional\_special\_tokens = \['<formula>'\] language = 'en' \*\*kwargs )

Parameters

-   **special** (`List[str]`, _optional_) — A list of special tokens (to be treated by the original implementation of this tokenizer).
-   **min\_freq** (`int`, _optional_, defaults to 0) — The minimum number of times a token has to be present in order to be kept in the vocabulary (otherwise it will be mapped to `unk_token`).
-   **max\_size** (`int`, _optional_) — The maximum size of the vocabulary. If left unset, it will default to the size of the vocabulary found after excluding the tokens according to the `min_freq` rule.
-   **lower\_case** (`bool`, _optional_, defaults to `False`) — Whether or not to lowercase the input when tokenizing.
-   **delimiter** (`str`, _optional_) — The delimiter used between tokens.
-   **vocab\_file** (`str`, _optional_) — File containing the vocabulary (from the original implementation).
-   **pretrained\_vocab\_file** (`str`, _optional_) — File containing the vocabulary as saved with the `save_pretrained()` method.
-   **never\_split** (`List[str]`, _optional_) — List of tokens that should never be split. If no list is specified, will simply use the existing special tokens.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **eos\_token** (`str`, _optional_, defaults to `"<eos>"`) — The end of sequence token.
-   **additional\_special\_tokens** (`List[str]`, _optional_, defaults to `["<formula>"]`) — A list of additional special tokens (for the HuggingFace functionality).
-   **language** (`str`, _optional_, defaults to `"en"`) — The language of this tokenizer (used for mose preprocessing).

Construct a Transformer-XL tokenizer adapted from Vocab class in [the original code](https://github.com/kimiyoung/transformer-xl). The Transformer-XL tokenizer is a word-level tokenizer (no sub-word tokenization).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/tokenization_transfo_xl.py#L313)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None )

## TransfoXL specific outputs

### class transformers.models.transfo\_xl.modeling\_transfo\_xl.TransfoXLModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L600)

( last\_hidden\_state: FloatTensor mems: typing.List\[torch.FloatTensor\] = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **mems** (`List[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

### class transformers.models.transfo\_xl.modeling\_transfo\_xl.TransfoXLLMHeadModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L665)

( losses: typing.Optional\[torch.FloatTensor\] = None prediction\_scores: FloatTensor = None mems: typing.List\[torch.FloatTensor\] = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None loss: typing.Optional\[torch.FloatTensor\] = None )

Parameters

-   **losses** (`torch.FloatTensor` of shape _(batch\_size, sequence\_length-1)_, _optional_, returned when `labels` is provided) — Language modeling losses (not reduced).
-   **prediction\_scores** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
-   **mems** (`List[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **loss** (`torch.FloatTensor` of shape `()`, _optional_, returned when `labels` is provided) — Reduced language modeling loss.

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

### class transformers.models.transfo\_xl.modeling\_tf\_transfo\_xl.TFTransfoXLModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_tf_transfo_xl.py#L673)

( last\_hidden\_state: tf.Tensor = None mems: List\[tf.Tensor\] = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **mems** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

### class transformers.models.transfo\_xl.modeling\_tf\_transfo\_xl.TFTransfoXLLMHeadModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_tf_transfo_xl.py#L704)

( prediction\_scores: tf.Tensor = None mems: List\[tf.Tensor\] = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **losses** (`tf.Tensor` of shape _(batch\_size, sequence\_length-1)_, _optional_, returned when `labels` is provided) — Language modeling losses (not reduced).
-   **prediction\_scores** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
-   **mems** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for model’s outputs that may also contain a past key/values (to speed up sequential decoding).

## TransfoXLModel

### class transformers.TransfoXLModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L764)

( config )

Parameters

-   **config** ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Bert Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L867)

( input\_ids: typing.Optional\[torch.LongTensor\] = None mems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.transfo\_xl.modeling\_transfo\_xl.TransfoXLModelOutput](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.models.transfo_xl.modeling_transfo_xl.TransfoXLModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **mems** (`List[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems given to this model should not be passed as `input_ids` as they have already been computed.
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.models.transfo\_xl.modeling\_transfo\_xl.TransfoXLModelOutput](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.models.transfo_xl.modeling_transfo_xl.TransfoXLModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **mems** (`List[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TransfoXLModel](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TransfoXLModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
>>> model = TransfoXLModel.from_pretrained("transfo-xl-wt103")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TransfoXLLMHeadModel

### class transformers.TransfoXLLMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L1004)

( config )

Parameters

-   **config** ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive input embeddings)

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L1060)

( input\_ids: typing.Optional\[torch.LongTensor\] = None mems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.transfo\_xl.modeling\_transfo\_xl.TransfoXLLMHeadModelOutput](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.models.transfo_xl.modeling_transfo_xl.TransfoXLLMHeadModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **mems** (`List[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems given to this model should not be passed as `input_ids` as they have already been computed.
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

A [transformers.models.transfo\_xl.modeling\_transfo\_xl.TransfoXLLMHeadModelOutput](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.models.transfo_xl.modeling_transfo_xl.TransfoXLLMHeadModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) and inputs.

-   **losses** (`torch.FloatTensor` of shape _(batch\_size, sequence\_length-1)_, _optional_, returned when `labels` is provided) — Language modeling losses (not reduced).
    
-   **prediction\_scores** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
    
-   **mems** (`List[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **loss** (`torch.FloatTensor` of shape `()`, _optional_, returned when `labels` is provided) Reduced language modeling loss.
    

The [TransfoXLLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, TransfoXLLMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
>>> model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## TransfoXLForSequenceClassification

### class transformers.TransfoXLForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L1192)

( config )

Parameters

-   **config** ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Transformer-XL Model transformer with a sequence classification head on top (linear layer).

[TransfoXLForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-1) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L1201)

( input\_ids: typing.Optional\[torch.LongTensor\] = None mems: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.models.transfo_xl.modeling_transfo_xl.TransfoXLSequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **mems** (`List[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems given to this model should not be passed as `input_ids` as they have already been computed.
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

`transformers.models.transfo_xl.modeling_transfo_xl.TransfoXLSequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.models.transfo_xl.modeling_transfo_xl.TransfoXLSequenceClassifierOutputWithPast` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **mems** (`List[torch.FloatTensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TransfoXLForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, TransfoXLForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
>>> model = TransfoXLForSequenceClassification.from_pretrained("transfo-xl-wt103")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TransfoXLForSequenceClassification.from_pretrained("transfo-xl-wt103", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, TransfoXLForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
>>> model = TransfoXLForSequenceClassification.from_pretrained("transfo-xl-wt103", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TransfoXLForSequenceClassification.from_pretrained(
...     "transfo-xl-wt103", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## TFTransfoXLModel

### class transformers.TFTransfoXLModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_tf_transfo_xl.py#L855)

( \*args \*\*kwargs )

Parameters

-   **config** ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Bert Model transformer outputting raw hidden-states without any specific head on top.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_tf_transfo_xl.py#L860)

( input\_ids: TFModelInputType | None = None mems: List\[tf.Tensor\] | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: bool | None = None output\_hidden\_states: bool | None = None return\_dict: bool | None = None training: bool = False ) → [transformers.models.transfo\_xl.modeling\_tf\_transfo\_xl.TFTransfoXLModelOutput](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **mems** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems given to this model should not be passed as `input_ids` as they have already been computed.
-   **head\_mask** (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

A [transformers.models.transfo\_xl.modeling\_tf\_transfo\_xl.TFTransfoXLModelOutput](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **mems** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFTransfoXLModel](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TFTransfoXLModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFTransfoXLModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
>>> model = TFTransfoXLModel.from_pretrained("transfo-xl-wt103")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFTransfoXLLMHeadModel

### class transformers.TFTransfoXLLMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_tf_transfo_xl.py#L899)

( \*args \*\*kwargs )

Parameters

-   **config** ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Transformer-XL Model with a language modeling head on top (adaptive softmax with weights tied to the adaptive input embeddings)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_tf_transfo_xl.py#L928)

( input\_ids: TFModelInputType | None = None mems: List\[tf.Tensor\] | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: bool | None = None output\_hidden\_states: bool | None = None return\_dict: bool | None = None labels: np.ndarray | tf.Tensor | None = None training: bool = False ) → [transformers.models.transfo\_xl.modeling\_tf\_transfo\_xl.TFTransfoXLLMHeadModelOutput](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLLMHeadModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **mems** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems given to this model should not be passed as `input_ids` as they have already been computed.
-   **head\_mask** (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

A [transformers.models.transfo\_xl.modeling\_tf\_transfo\_xl.TFTransfoXLLMHeadModelOutput](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLLMHeadModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) and inputs.

-   **losses** (`tf.Tensor` of shape _(batch\_size, sequence\_length-1)_, _optional_, returned when `labels` is provided) — Language modeling losses (not reduced).
    
-   **prediction\_scores** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token after SoftMax).
    
-   **mems** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFTransfoXLLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TFTransfoXLLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFTransfoXLLMHeadModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
>>> model = TFTransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

## TFTransfoXLForSequenceClassification

### class transformers.TFTransfoXLForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_tf_transfo_xl.py#L1006)

( \*args \*\*kwargs )

Parameters

-   **config** ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Transfo XL Model transformer with a sequence classification head on top (linear layer).

[TFTransfoXLForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TFTransfoXLForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-1,GPT-2) do.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_tf_transfo_xl.py#L1026)

( input\_ids: TFModelInputType | None = None mems: List\[tf.Tensor\] | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → `transformers.models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLSequenceClassifierOutputWithPast` or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **mems** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `mems` output below). Can be used to speed up sequential decoding. The token ids which have their mems given to this model should not be passed as `input_ids` as they have already been computed.
-   **head\_mask** (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the cross entropy classification loss. Indices should be in `[0, ..., config.vocab_size - 1]`.

Returns

`transformers.models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLSequenceClassifierOutputWithPast` or `tuple(tf.Tensor)`

A `transformers.models.transfo_xl.modeling_tf_transfo_xl.TFTransfoXLSequenceClassifierOutputWithPast` or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([TransfoXLConfig](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TransfoXLConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **mems** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks). Can be used (see `mems` input) to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFTransfoXLForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/transfo-xl#transformers.TFTransfoXLForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFTransfoXLForSequenceClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
>>> model = TFTransfoXLForSequenceClassification.from_pretrained("transfo-xl-wt103")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

>>> logits = model(**inputs).logits

>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TFTransfoXLForSequenceClassification.from_pretrained("transfo-xl-wt103", num_labels=num_labels)

>>> labels = tf.constant(1)
>>> loss = model(**inputs, labels=labels).loss
```

## Internal Layers

### class transformers.AdaptiveEmbedding

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_transfo_xl.py#L399)

( n\_token d\_embed d\_proj cutoffs div\_val = 1 sample\_softmax = False )

### class transformers.TFAdaptiveEmbedding

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/transfo_xl/modeling_tf_transfo_xl.py#L344)

( \*args \*\*kwargs )