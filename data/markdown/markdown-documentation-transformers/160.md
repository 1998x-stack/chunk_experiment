# Falcon

## Overview

Falcon is a class of causal decoder-only models built by [TII](https://www.tii.ae/). The largest Falcon checkpoints have been trained on >=1T tokens of text, with a particular emphasis on the [RefinedWeb](https://arxiv.org/abs/2306.01116) corpus. They are made available under the Apache 2.0 license.

Falcon’s architecture is modern and optimized for inference, with multi-query attention and support for efficient attention variants like `FlashAttention`. Both ‘base’ models trained only as causal language models as well as ‘instruct’ models that have received further fine-tuning are available.

Falcon models are (as of 2023) some of the largest and most powerful open-source language models, and consistently rank highly in the [OpenLLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

## Converting custom checkpoints

Falcon models were initially added to the Hugging Face Hub as custom code checkpoints. However, Falcon is now fully supported in the Transformers library. If you fine-tuned a model from a custom code checkpoint, we recommend converting your checkpoint to the new in-library format, as this should give significant improvements to stability and performance, especially for generation, as well as removing the need to use `trust_remote_code=True`!

You can convert custom code checkpoints to full Transformers checkpoints using the `convert_custom_code_checkpoint.py` script located in the [Falcon model directory](https://github.com/huggingface/transformers/tree/main/src/transformers/models/falcon) of the Transformers library. To use this script, simply call it with `python convert_custom_code_checkpoint.py --checkpoint_dir my_model`. This will convert your checkpoint in-place, and you can immediately load it from the directory afterwards with e.g. `from_pretrained()`. If your model hasn’t been uploaded to the Hub, we recommend making a backup before attempting the conversion, just in case!

## FalconConfig

### class transformers.FalconConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/configuration_falcon.py#L28)

( vocab\_size = 65024 hidden\_size = 4544 num\_hidden\_layers = 32 num\_attention\_heads = 71 layer\_norm\_epsilon = 1e-05 initializer\_range = 0.02 use\_cache = True hidden\_dropout = 0.0 attention\_dropout = 0.0 num\_kv\_heads = None alibi = False new\_decoder\_architecture = False multi\_query = True parallel\_attn = True bias = False max\_position\_embeddings = 2048 rope\_theta = 10000.0 rope\_scaling = None bos\_token\_id = 11 eos\_token\_id = 11 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 65024) — Vocabulary size of the Falcon model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [FalconModel](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconModel)
-   **hidden\_size** (`int`, _optional_, defaults to 4544) — Dimension of the hidden representations.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 32) — Number of hidden layers in the Transformer decoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 71) — Number of attention heads for each attention layer in the Transformer encoder.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.
-   **layer\_norm\_epsilon** (`float`, _optional_, defaults to 1e-5) — The epsilon used by the layer normalization layers.
-   **hidden\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout probability for MLP layers.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout probability for attention layers.
-   **num\_kv\_heads** (`int`, _optional_) — Number of key-value heads to use per attention layer. If unset, defaults to the same value as `num_attention_heads`.
-   **alibi** (`bool`, _optional_, defaults to `False`) — Whether to use ALiBi positional biases during self-attention.
-   **new\_decoder\_architecture** (`bool`, _optional_, defaults to `False`) — Whether to use the new (Falcon-40B) decoder architecture. If `True`, the `multi_query` and `parallel_attn` arguments are ignored, as the new decoder always uses parallel attention.
-   **multi\_query** (`bool`, _optional_, defaults to `True`) — Whether to use multi-query attention in the decoder. Ignored when `new_decoder_architecture` is `True`.
-   **parallel\_attn** (`bool`, _optional_, defaults to `True`) — Whether to compute attention in parallel with the feedforward layer. If False, they are consecutive instead, as in the original Transformer architecture. Ignored when `new_decoder_architecture` is `True`.
-   **bias** (`bool`, _optional_, defaults to `False`) — Whether to use bias on Linear layers.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 2048) — The maximum sequence length that this model might ever be used with, when `alibi` is `False`. Pretrained Falcon models with RoPE support up to 2048 tokens.
-   **rope\_theta** (`float`, _optional_, defaults to 10000.0) — The base period of the RoPE embeddings.
-   **rope\_scaling** (`Dict`, _optional_) — Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don’t update `max_position_embeddings` to the expected new maximum. See the following thread for more information on how these scaling strategies behave: [https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically\_scaled\_rope\_further\_increases/](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/). This is an experimental feature, subject to breaking API changes in future versions.
-   **bos\_token\_id** (`int`, _optional_, defaults to 11) — The id of the “beginning-of-sequence” token.
-   **eos\_token\_id** (`int`, _optional_, defaults to 11) — The id of the “end-of-sequence” token.

This is the configuration class to store the configuration of a [FalconModel](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconModel). It is used to instantiate a Falcon model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the [tiiuae/falcon-7b](https://huggingface.co/tiiuae/falcon-7b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```python
>>> from transformers import FalconModel, FalconConfig

>>> 
>>> configuration = FalconConfig(num_hidden_layers=2)

>>> 
>>> model = FalconModel(configuration)

>>> 
>>> configuration = model.config
```

## FalconModel

### class transformers.FalconModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L987)

( config: FalconConfig )

Parameters

-   **config** ([FalconConfig](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Falcon Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1046)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.LongTensor\] = None inputs\_embeds: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.num_hidden_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
    
    Each element of `past_key_values` is a tuple (past\_key, past\_value):
    
    -   past\_key: \[batch\_size \* num\_heads, head\_dim, kv\_length\]
    -   past\_value: \[batch\_size \* num\_heads, kv\_length, head\_dim\]
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see `past_key_values`).
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([FalconConfig](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    

The [FalconModel](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FalconModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
>>> model = FalconModel.from_pretrained("Rocketknight1/falcon-rw-1b")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## FalconForCausalLM

### class transformers.FalconForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1206)

( config: FalconConfig )

Parameters

-   **config** ([FalconConfig](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Falcon Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1250)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.num_hidden_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
    
    Each element of `past_key_values` is a tuple (past\_key, past\_value):
    
    -   past\_key: \[batch\_size \* num\_heads, head\_dim, kv\_length\]
    -   past\_value: \[batch\_size \* num\_heads, kv\_length, head\_dim\]
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see `past_key_values`).
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([FalconConfig](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Cross attentions weights after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `torch.FloatTensor` tuples of length `config.n_layers`, with each tuple containing the cached key, value states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting. Only relevant if `config.is_decoder = True`.
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    

The [FalconForCausalLM](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, FalconForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
>>> model = FalconForCausalLM.from_pretrained("Rocketknight1/falcon-rw-1b")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## FalconForSequenceClassification

### class transformers.FalconForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1359)

( config: FalconConfig )

Parameters

-   **config** ([FalconConfig](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Falcon Model transformer with a sequence classification head on top (linear layer).

[FalconForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-1) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1369)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.num_hidden_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
    
    Each element of `past_key_values` is a tuple (past\_key, past\_value):
    
    -   past\_key: \[batch\_size \* num\_heads, head\_dim, kv\_length\]
    -   past\_value: \[batch\_size \* num\_heads, kv\_length, head\_dim\]
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see `past_key_values`).
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

Returns

`transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

A `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([FalconConfig](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [FalconForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, FalconForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
>>> model = FalconForSequenceClassification.from_pretrained("Rocketknight1/falcon-rw-1b")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = FalconForSequenceClassification.from_pretrained("Rocketknight1/falcon-rw-1b", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, FalconForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
>>> model = FalconForSequenceClassification.from_pretrained("Rocketknight1/falcon-rw-1b", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = FalconForSequenceClassification.from_pretrained(
...     "Rocketknight1/falcon-rw-1b", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## FalconForTokenClassification

### class transformers.FalconForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1475)

( config: FalconConfig )

Parameters

-   **config** ([FalconConfig](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Falcon Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1493)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Union\[typing.Tuple\[typing.Tuple\[torch.Tensor, torch.Tensor\], ...\], NoneType\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None inputs\_embeds: typing.Optional\[torch.Tensor\] = None labels: typing.Optional\[torch.Tensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.num_hidden_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
    
    Each element of `past_key_values` is a tuple (past\_key, past\_value):
    
    -   past\_key: \[batch\_size \* num\_heads, head\_dim, kv\_length\]
    -   past\_value: \[batch\_size \* num\_heads, kv\_length, head\_dim\]
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see `past_key_values`).
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([FalconConfig](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [FalconForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FalconForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Rocketknight1/falcon-rw-1b")
>>> model = FalconForTokenClassification.from_pretrained("Rocketknight1/falcon-rw-1b")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> 
>>> 
>>> 
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
```

## FalconForQuestionAnswering

### class transformers.FalconForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1564)

( config )

Parameters

-   **config** ([FalconConfig](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Falcon Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/falcon/modeling_falcon.py#L1573)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None start\_positions: typing.Optional\[torch.LongTensor\] = None end\_positions: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.num_hidden_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
    
    Each element of `past_key_values` is a tuple (past\_key, past\_value):
    
    -   past\_key: \[batch\_size \* num\_heads, head\_dim, kv\_length\]
    -   past\_value: \[batch\_size \* num\_heads, kv\_length, head\_dim\]
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see `past_key_values`).
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **start\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the start of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss.
-   **end\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the end of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss.

The [FalconForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/falcon#transformers.FalconForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.