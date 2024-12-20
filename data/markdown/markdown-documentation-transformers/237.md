# PEGASUS-X

## Overview

The PEGASUS-X model was proposed in [Investigating Efficiently Extending Transformers for Long Input Summarization](https://arxiv.org/abs/2208.04347) by Jason Phang, Yao Zhao and Peter J. Liu.

PEGASUS-X (PEGASUS eXtended) extends the PEGASUS models for long input summarization through additional long input pretraining and using staggered block-local attention with global tokens in the encoder.

The abstract from the paper is the following:

_While large pretrained Transformer models have proven highly capable at tackling natural language tasks, handling long sequence inputs continues to be a significant challenge. One such task is long input summarization, where inputs are longer than the maximum input context of most pretrained models. Through an extensive set of experiments, we investigate what model architectural changes and pretraining paradigms can most efficiently adapt a pretrained Transformer for long input summarization. We find that a staggered, block-local Transformer with global encoder tokens strikes a good balance of performance and efficiency, and that an additional pretraining phase on long sequences meaningfully improves downstream summarization performance. Based on our findings, we introduce PEGASUS-X, an extension of the PEGASUS model with additional long input pretraining to handle inputs of up to 16K tokens. PEGASUS-X achieves strong performance on long input summarization tasks comparable with much larger models while adding few additional parameters and not requiring model parallelism to train._

Tips:

-   PEGASUS-X uses the same tokenizer as PEGASUS.

This model was contributed by \[zphang\](<[https://huggingface.co/zphang](https://huggingface.co/zphang)). The original code can be found [here](https://github.com/google-research/pegasus).

## Documentation resources

-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## PegasusXConfig

### class transformers.PegasusXConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus_x/configuration_pegasus_x.py#L30)

( vocab\_size = 96103 max\_position\_embeddings = 16384 encoder\_layers = 16 encoder\_ffn\_dim = 4096 encoder\_attention\_heads = 16 decoder\_layers = 16 decoder\_ffn\_dim = 4096 decoder\_attention\_heads = 16 encoder\_layerdrop = 0.0 decoder\_layerdrop = 0.0 use\_cache = True is\_encoder\_decoder = True activation\_function = 'gelu' d\_model = 1024 dropout = 0.1 attention\_dropout = 0.0 activation\_dropout = 0.0 init\_std = 0.02 decoder\_start\_token\_id = 0 scale\_embedding = True pad\_token\_id = 0 eos\_token\_id = 1 forced\_eos\_token\_id = 1 num\_global\_tokens = 32 block\_size = 512 stagger\_local\_blocks = True \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 96103) — Vocabulary size of the PEGASUS-X model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [PegasusXModel](/docs/transformers/v4.34.0/en/model_doc/pegasus_x#transformers.PegasusXModel).
-   **d\_model** (`int`, _optional_, defaults to 1024) — Dimension of the layers and the pooler layer.
-   **encoder\_layers** (`int`, _optional_, defaults to 16) — Number of encoder layers.
-   **decoder\_layers** (`int`, _optional_, defaults to 16) — Number of decoder layers.
-   **encoder\_attention\_heads** (`int`, _optional_, defaults to 16) — Number of attention heads for each attention layer in the Transformer encoder.
-   **decoder\_attention\_heads** (`int`, _optional_, defaults to 16) — Number of attention heads for each attention layer in the Transformer decoder.
-   **decoder\_ffn\_dim** (`int`, _optional_, defaults to 4096) — Dimension of the “intermediate” (often named feed-forward) layer in decoder.
-   **encoder\_ffn\_dim** (`int`, _optional_, defaults to 4096) — Dimension of the “intermediate” (often named feed-forward) layer in decoder.
-   **activation\_function** (`str` or `function`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **dropout** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **activation\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for activations inside the fully connected layer.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 16384) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **init\_std** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **encoder\_layerdrop** (`float`, _optional_, defaults to 0.0) — The LayerDrop probability for the encoder. See the \[LayerDrop paper\](see [https://arxiv.org/abs/1909.11556](https://arxiv.org/abs/1909.11556)) for more details.
-   **decoder\_layerdrop** (`float`, _optional_, defaults to 0.0) — The LayerDrop probability for the decoder. See the \[LayerDrop paper\](see [https://arxiv.org/abs/1909.11556](https://arxiv.org/abs/1909.11556)) for more details.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models)
-   **forced\_eos\_token\_id** (`int`, _optional_, defaults to 1) — The id of the token to force as the last generated token when `max_length` is reached. Usually set to `eos_token_id`.
-   **num\_global\_tokens** (`int`, _optional_, defaults to 128) — Number of global tokens to use for the encoder
-   **block\_size** (`int`, _optional_, defaults to 512) — Block size for encoder local attention. Sequence length should be an exact multiple of block size. block\_size must be a multiple of 2 if stagger\_local\_block is True
-   **stagger\_local\_block** (`bool`, _optional_, defaults to `True`) — Whether to stagger every other local attention by half a block

This is the configuration class to store the configuration of a [PegasusXModel](/docs/transformers/v4.34.0/en/model_doc/pegasus_x#transformers.PegasusXModel). It is used to instantiate a PEGASUS-X model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the PEGASUS-X [google/pegasus-x-large](https://huggingface.co/google/pegasus-x-large) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import PegasusXConfig, PegasusXModel

>>> 
>>> configuration = PegasusXConfig()

>>> 
>>> model = PegasusXModel(configuration)

>>> 
>>> configuration = model.config
```

## PegasusXModel

### class transformers.PegasusXModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus_x/modeling_pegasus_x.py#L1397)

( config: PegasusXConfig )

Parameters

-   **config** ([PegasusXConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus_x#transformers.PegasusXConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare PEGASUS-X Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus_x/modeling_pegasus_x.py#L1449)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None decoder\_input\_ids: typing.Optional\[torch.Tensor\] = None decoder\_attention\_mask: typing.Optional\[torch.Tensor\] = None encoder\_outputs: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None past\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None decoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    PEGASUS-X uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
-   **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
-   **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, _optional_) — Tuple consists of (`last_hidden_state`, _optional_: `hidden_states`, _optional_: `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs\_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_): Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
-   **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PegasusXConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus_x#transformers.PegasusXConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [PegasusXModel](/docs/transformers/v4.34.0/en/model_doc/pegasus_x#transformers.PegasusXModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, PegasusModel

>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")
>>> model = PegasusModel.from_pretrained("google/pegasus-x-large")

>>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
>>> decoder_inputs = tokenizer("Studies show that", return_tensors="pt")
>>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 4, 1024]
```

## PegasusXForConditionalGeneration

### class transformers.PegasusXForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus_x/modeling_pegasus_x.py#L1540)

( config: PegasusXConfig )

Parameters

-   **config** ([PegasusXConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus_x#transformers.PegasusXConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The PEGASUS-X for conditional generation (e.g. summarization). This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/pegasus_x/modeling_pegasus_x.py#L1587)

( input\_ids: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None decoder\_input\_ids: typing.Optional\[torch.Tensor\] = None decoder\_attention\_mask: typing.Optional\[torch.Tensor\] = None encoder\_outputs: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None past\_key\_values: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None decoder\_inputs\_embeds: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    PEGASUS-X uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
-   **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
-   **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, _optional_) — Tuple consists of (`last_hidden_state`, _optional_: `hidden_states`, _optional_: `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs\_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_): Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
-   **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PegasusXConfig](/docs/transformers/v4.34.0/en/model_doc/pegasus_x#transformers.PegasusXConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [PegasusXForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/pegasus_x#transformers.PegasusXForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```
>>> from transformers import AutoTokenizer, PegasusXForConditionalGeneration

>>> model = PegasusXForConditionalGeneration.from_pretrained("google/pegasus-x-base")
>>> tokenizer = AutoTokenizer.from_pretrained("google/pegasus-x-large")

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