# CPMAnt

## Overview

CPM-Ant is an open-source Chinese pre-trained language model (PLM) with 10B parameters. It is also the first milestone of the live training process of CPM-Live. The training process is cost-effective and environment-friendly. CPM-Ant also achieves promising results with delta tuning on the CUGE benchmark. Besides the full model, we also provide various compressed versions to meet the requirements of different hardware configurations. [See more](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live)

Tips:

This model was contributed by [OpenBMB](https://huggingface.co/openbmb). The original code can be found [here](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live).

⚙️ Training & Inference

-   A tutorial on [CPM-Live](https://github.com/OpenBMB/CPM-Live/tree/cpm-ant/cpm-live).

## CpmAntConfig

### class transformers.CpmAntConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cpmant/configuration_cpmant.py#L29)

( vocab\_size: int = 30720 hidden\_size: int = 4096 num\_attention\_heads: int = 32 dim\_head: int = 128 dim\_ff: int = 10240 num\_hidden\_layers: int = 48 dropout\_p: int = 0.0 position\_bias\_num\_buckets: int = 512 position\_bias\_max\_distance: int = 2048 eps: int = 1e-06 init\_std: float = 1.0 prompt\_types: int = 32 prompt\_length: int = 32 segment\_types: int = 32 use\_cache: bool = True \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 30720) — Vocabulary size of the CPMAnt model. Defines the number of different tokens that can be represented by the `input` passed when calling [CpmAntModel](/docs/transformers/v4.34.0/en/model_doc/cpmant#transformers.CpmAntModel).
-   **hidden\_size** (`int`, _optional_, defaults to 4096) — Dimension of the encoder layers.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 32) — Number of attention heads in the Transformer encoder.
-   **dim\_head** (`int`, _optional_, defaults to 128) — Dimension of attention heads for each attention layer in the Transformer encoder.
-   **dim\_ff** (`int`, _optional_, defaults to 10240) — Dimension of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 48) — Number of layers of the Transformer encoder.
-   **dropout\_p** (`float`, _optional_, defaults to 0.1) — The dropout probabilitiy for all fully connected layers in the embeddings, encoder.
-   **position\_bias\_num\_buckets** (`int`, _optional_, defaults to 512) — The number of position\_bias buckets.
-   **position\_bias\_max\_distance** (`int`, _optional_, defaults to 2048) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **eps** (`float`, _optional_, defaults to 1e-6) — The epsilon used by the layer normalization layers.
-   **prompt\_types** (`int`, _optional_, defaults to 32) — The type of prompt.
-   **prompt\_length** (`int`, _optional_, defaults to 32) — The length of prompt.
-   **segment\_types** (`int`, _optional_, defaults to 32) — The type of segment.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether to use cache.
-   **init\_std** (`float`, _optional_, defaults to 1.0) — Initialize parameters with std = init\_std.

This is the configuration class to store the configuration of a [CpmAntModel](/docs/transformers/v4.34.0/en/model_doc/cpmant#transformers.CpmAntModel). It is used to instantiate an CPMAnt model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the CPMAnt [openbmb/cpm-ant-10b](https://huggingface.co/openbmb/cpm-ant-10b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CpmAntModel, CpmAntConfig

>>> 
>>> configuration = CpmAntConfig()

>>> 
>>> model = CpmAntModel(configuration)

>>> 
>>> configuration = model.config
```

## CpmAntTokenizer

### class transformers.CpmAntTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cpmant/tokenization_cpmant.py#L88)

( vocab\_file bod\_token = '<d>' eod\_token = '</d>' bos\_token = '<s>' eos\_token = '</s>' pad\_token = '<pad>' unk\_token = '<unk>' line\_token = '</n>' space\_token = '</\_>' padding\_side = 'left' \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **bod\_token** (`str`, _optional_, defaults to `"<d>"`) — The beginning of document token.
-   **eod\_token** (`str`, _optional_, defaults to `"</d>"`) — The end of document token.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token.
-   **line\_token** (`str`, _optional_, defaults to `"</n>"`) — The line token.
-   **space\_token** (`str`, _optional_, defaults to `"</_>"`) — The space token.

Construct a CPMAnt tokenizer. Based on byte-level Byte-Pair-Encoding.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cpmant/tokenization_cpmant.py#L236)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.List\[int\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — The first tokenized sequence that special tokens will be added.
-   **token\_ids\_1** (`List[int]`) — The optional second tokenized sequence that special tokens will be added.

The model input with special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A CPMAnt sequence has the following format:

-   single sequence: `[BOS] Sequence`.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cpmant/tokenization_cpmant.py#L254)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## CpmAntModel

### class transformers.CpmAntModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cpmant/modeling_cpmant.py#L603)

( config: CpmAntConfig )

The bare CPMAnt Model outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Parameters config ([~CpmAntConfig](/docs/transformers/v4.34.0/en/model_doc/cpmant#transformers.CpmAntConfig)): Model configuration class with all the parameters of the Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cpmant/modeling_cpmant.py#L641)

( input\_ids: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None use\_cache: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None \*\*kwargs ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.Tensor` of shape `(batch_size, seq_len)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using `CPMAntTokenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CpmAntConfig](/docs/transformers/v4.34.0/en/model_doc/cpmant#transformers.CpmAntConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CpmAntModel](/docs/transformers/v4.34.0/en/model_doc/cpmant#transformers.CpmAntModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, CpmAntModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("openbmb/cpm-ant-10b")
>>> model = CpmAntModel.from_pretrained("openbmb/cpm-ant-10b")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## CpmAntForCausalLM

### class transformers.CpmAntForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cpmant/modeling_cpmant.py#L750)

( config: CpmAntConfig )

The CPMAnt Model with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

Parameters config ([~CpmAntConfig](/docs/transformers/v4.34.0/en/model_doc/cpmant#transformers.CpmAntConfig)): Model configuration class with all the parameters of the Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/cpmant/modeling_cpmant.py#L763)

( input\_ids: typing.Optional\[torch.Tensor\] = None past\_key\_values: typing.Union\[typing.List\[typing.Tuple\[torch.Tensor, torch.Tensor\]\], NoneType\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None \*\*kwargs ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.Tensor` of shape `(batch_size, seq_len)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using `CPMAntTokenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
    
    Args — input\_ids (`torch.Tensor` of shape `(batch_size, seq_len)`): Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using `CPMAntTokenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids) past\_key\_values (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`): Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding. use\_cache (`bool`, _optional_): If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`). output\_attentions (`bool`, _optional_): Whether or not to return the attentions tensors of all attention layers. output\_hidden\_states (`bool`, _optional_): Whether or not to return the hidden states of all layers. labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_): Labels for computing the masked language modeling loss. return\_dict (`bool`, _optional_): Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. attention\_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_): CPMAnt will process attention mask automatically, this parameter is a dummy parameter for text-generation pipeline.
    
    Example —
    
-   **Text** Generation with CpmAntForCausalLM. —

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CpmAntConfig](/docs/transformers/v4.34.0/en/model_doc/cpmant#transformers.CpmAntConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CpmAntForCausalLM](/docs/transformers/v4.34.0/en/model_doc/cpmant#transformers.CpmAntForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, CpmAntForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("openbmb/cpm-ant-10b")
>>> model = CpmAntForCausalLM.from_pretrained("openbmb/cpm-ant-10b")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```