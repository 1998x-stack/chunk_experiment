# Mistral

## Overview

Mistral-7B-v0.1 is Mistral AI’s first Large Language Model (LLM).

## Model Details

Mistral-7B-v0.1 is a decoder-based LM with the following architectural choices:

-   Sliding Window Attention - Trained with 8k context length and fixed cache size, with a theoretical attention span of 128K tokens
-   GQA (Grouped Query Attention) - allowing faster inference and lower cache size.
-   Byte-fallback BPE tokenizer - ensures that characters are never mapped to out of vocabulary tokens.

We also provide an instruction fine-tuned model: `Mistral-7B-Instruct-v0.1` which can be used for chat-based inference.

For more details please read our [release blog post](https://mistral.ai/news/announcing-mistral-7b-v0.1/)

## License

Both `Mistral-7B-v0.1` and `Mistral-7B-Instruct-v0.1` are released under the Apache 2.0 license.

## Usage

`Mistral-7B-v0.1` and `Mistral-7B-Instruct-v0.1` can be found on the [Huggingface Hub](https://huggingface.co/mistralai)

These ready-to-use checkpoints can be downloaded and used via the HuggingFace Hub:

```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" 

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected outupt"
```

Raw weights for `Mistral-7B-v0.1` and `Mistral-7B-Instruct-v0.1` can be downloaded from:

| Model Name | Checkpoint |
| --- | --- |
| `Mistral-7B-v0.1` | [Raw Checkpoint](https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar) |
| `Mistral-7B-Instruct-v0.1` | [Raw Checkpoint](https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-instruct-v0.1.tar) |

To use these raw checkpoints with HuggingFace you can use the `convert_mistral_weights_to_hf.py` script to convert them to the HuggingFace format:

```
python src/transformers/models/mistral/convert_mistral_weights_to_hf.py \
    --input_dir /path/to/downloaded/mistral/weights --model_size 7B --output_dir /output/path
```

You can then load the converted model from the `output/path`:

```
from transformers import MistralForCausalLM, LlamaTokenzier

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = MistralForCausalLM.from_pretrained("/output/path")
```

## Combining Mistral and Flash Attention 2

First, make sure to install the latest version of Flash Attention 2 to include the sliding window attention feature.

```
pip install -U flash-attn --no-build-isolation
```

Make also sure that you have a hardware that is compatible with Flash-Attention 2. Read more about it in the official documentation of [`flash-attn`](https://github.com/Dao-AILab/flash-attention) repository. Make also sure to load your model in half-precision (e.g. `torch.float16`)

To load and run a model using Flash Attention 2, refer to the snippet below:

```
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> device = "cuda" 

>>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, use_flash_attention_2=True)
>>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

>>> prompt = "My favourite condiment is"

>>> model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
>>> model.to(device)

>>> generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
>>> tokenizer.batch_decode(generated_ids)[0]
"The expected outupt"
```

### Expected speedups

Below is a expected speedup diagram that compares pure inference time between the native implementation in transformers using `mistralai/Mistral-7B-v0.1` checkpoint and the Flash Attention 2 version of the model.

![](https://huggingface.co/datasets/ybelkada/documentation-images/resolve/main/mistral-7b-inference-large-seqlen.png)

### Sliding window Attention

The current implementation supports the sliding window attention mechanism and memory efficient cache management. To enable sliding window attention, just make sure to have a `flash-attn` version that is compatible with sliding window attention (`>=2.3.0`).

The Flash Attention-2 model uses also a more memory efficient cache slicing mechanism - as recommended per the official implementation of Mistral model that use rolling cache mechanism we keep the cache size fixed (`self.config.sliding_window`), support batched generation only for `padding_side="left"` and use the absolute position of the current token to compute the positional embedding.

## The Mistral Team

Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.

## MistralConfig

### class transformers.MistralConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/configuration_mistral.py#L29)

( vocab\_size = 32000 hidden\_size = 4096 intermediate\_size = 14336 num\_hidden\_layers = 32 num\_attention\_heads = 32 num\_key\_value\_heads = 8 hidden\_act = 'silu' max\_position\_embeddings = 131072 initializer\_range = 0.02 rms\_norm\_eps = 1e-06 use\_cache = True pad\_token\_id = None bos\_token\_id = 1 eos\_token\_id = 2 tie\_word\_embeddings = False rope\_theta = 10000.0 sliding\_window = 4096 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 32000) — Vocabulary size of the Mistral model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [MistralModel](/docs/transformers/v4.34.0/en/model_doc/mistral#transformers.MistralModel)
-   **hidden\_size** (`int`, _optional_, defaults to 4096) — Dimension of the hidden representations.
-   **intermediate\_size** (`int`, _optional_, defaults to 14336) — Dimension of the MLP representations.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 32) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 32) — Number of attention heads for each attention layer in the Transformer encoder.
-   **num\_key\_value\_heads** (`int`, _optional_, defaults to 8) — This is the number of key\_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details checkout [this paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to` 8\`.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"silu"`) — The non-linear activation function (function or string) in the decoder.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 4096_32) — The maximum sequence length that this model might ever be used with. Mistral’s sliding window attention allows sequence of up to 4096_32 tokens.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **rms\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the rms normalization layers.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.
-   **tie\_word\_embeddings(`bool`,** _optional_, defaults to `False`) — Whether to tie weight embeddings
-   **rope\_theta** (`float`, _optional_, defaults to 10000.0) — The base period of the RoPE embeddings.
-   **sliding\_window** (`int`, _optional_, defaults to 4096) — Sliding window attention window size. If not specified, will default to `4096`.
    
    Example —
    

This is the configuration class to store the configuration of a [MistralModel](/docs/transformers/v4.34.0/en/model_doc/mistral#transformers.MistralModel). It is used to instantiate an Mistral model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Mistral-7B-v0.1 or Mistral-7B-Instruct-v0.1.

[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

```
>>> from transformers import MistralModel, MistralConfig

>>> 
>>> configuration = MistralConfig()

>>> 
>>> model = MistralModel(configuration)

>>> 
>>> configuration = model.config
```

## MistralModel

### class transformers.MistralModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py#L762)

( config: MistralConfig )

Parameters

-   **config** ([MistralConfig](/docs/transformers/v4.34.0/en/model_doc/mistral#transformers.MistralConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights. config — MistralConfig

The bare Mistral Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Transformer decoder consisting of _config.num\_hidden\_layers_ layers. Each layer is a `MistralDecoderLayer`

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py#L815)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
    If you want to change padding behavior, you should read `modeling_opt._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

The [MistralModel](/docs/transformers/v4.34.0/en/model_doc/mistral#transformers.MistralModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## MistralForCausalLM

### class transformers.MistralForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py#L967)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py#L997)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
    If you want to change padding behavior, you should read `modeling_opt._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
    
    Args — labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_): Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
    

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MistralConfig](/docs/transformers/v4.34.0/en/model_doc/mistral#transformers.MistralConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [MistralForCausalLM](/docs/transformers/v4.34.0/en/model_doc/mistral#transformers.MistralForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, MistralForCausalLM

>>> model = MistralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
>>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> 
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

## MistralForSequenceClassification

### class transformers.MistralForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py#L1142)

( config )

Parameters

-   **config** ([MistralConfig](/docs/transformers/v4.34.0/en/model_doc/mistral#transformers.MistralConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Mistral Model transformer with a sequence classification head on top (linear layer).

[MistralForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/mistral#transformers.MistralForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mistral/modeling_mistral.py#L1158)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
    If you want to change padding behavior, you should read `modeling_opt._prepare_decoder_attention_mask` and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

The [MistralForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/mistral#transformers.MistralForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.