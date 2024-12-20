# Custom Layers and Utilities

This page lists all the custom layers used by the library, as well as the utility functions it provides for modeling.

Most of those are only useful if you are studying the code of the models in the library.

## Pytorch custom modules

### class transformers.Conv1D

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pytorch_utils.py#L86)

( nfnx )

Parameters

-   **nf** (`int`) — The number of output features.
-   **nx** (`int`) — The number of input features.

1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

Basically works like a linear layer but the weights are transposed.

### class transformers.modeling\_utils.PoolerStartLogits

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L3971)

( config: PretrainedConfig )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The config used by the model, will be used to grab the `hidden_size` of the model.

Compute SQuAD start logits from sequence hidden states.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L3984)

( hidden\_states: FloatTensorp\_mask: typing.Optional\[torch.FloatTensor\] = None ) → `torch.FloatTensor`

Parameters

-   **hidden\_states** (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`) — The final hidden states of the model.
-   **p\_mask** (`torch.FloatTensor` of shape `(batch_size, seq_len)`, _optional_) — Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token should be masked.

Returns

`torch.FloatTensor`

The start logits for SQuAD.

### class transformers.modeling\_utils.PoolerEndLogits

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L4009)

( config: PretrainedConfig )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps` to use.

Compute SQuAD end logits from sequence hidden states.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L4026)

( hidden\_states: FloatTensorstart\_states: typing.Optional\[torch.FloatTensor\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Nonep\_mask: typing.Optional\[torch.FloatTensor\] = None ) → `torch.FloatTensor`

Parameters

-   **hidden\_states** (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`) — The final hidden states of the model.
-   **start\_states** (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, _optional_) — The hidden states of the first tokens for the labeled span.
-   **start\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — The position of the first token for the labeled span.
-   **p\_mask** (`torch.FloatTensor` of shape `(batch_size, seq_len)`, _optional_) — Mask for tokens at invalid position, such as query and special symbols (PAD, SEP, CLS). 1.0 means token should be masked.

Returns

`torch.FloatTensor`

The end logits for SQuAD.

One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides `start_states`.

### class transformers.modeling\_utils.PoolerAnswerClass

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L4078)

( config )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The config used by the model, will be used to grab the `hidden_size` of the model.

Compute SQuAD 2.0 answer class from classification and start tokens hidden states.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L4093)

( hidden\_states: FloatTensorstart\_states: typing.Optional\[torch.FloatTensor\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Nonecls\_index: typing.Optional\[torch.LongTensor\] = None ) → `torch.FloatTensor`

Parameters

-   **hidden\_states** (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`) — The final hidden states of the model.
-   **start\_states** (`torch.FloatTensor` of shape `(batch_size, seq_len, hidden_size)`, _optional_) — The hidden states of the first tokens for the labeled span.
-   **start\_positions** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — The position of the first token for the labeled span.
-   **cls\_index** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Position of the CLS token for each sentence in the batch. If `None`, takes the last token.

Returns

`torch.FloatTensor`

The SQuAD 2.0 answer class.

One of `start_states` or `start_positions` should be not `None`. If both are set, `start_positions` overrides `start_states`.

### class transformers.modeling\_utils.SquadHeadOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L4144)

( loss: typing.Optional\[torch.FloatTensor\] = Nonestart\_top\_log\_probs: typing.Optional\[torch.FloatTensor\] = Nonestart\_top\_index: typing.Optional\[torch.LongTensor\] = Noneend\_top\_log\_probs: typing.Optional\[torch.FloatTensor\] = Noneend\_top\_index: typing.Optional\[torch.LongTensor\] = Nonecls\_logits: typing.Optional\[torch.FloatTensor\] = None )

Base class for outputs of question answering models using a [SQuADHead](/docs/transformers/v4.34.0/en/internal/modeling_utils#transformers.modeling_utils.SQuADHead).

### class transformers.modeling\_utils.SQuADHead

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L4174)

( config )

Parameters

-   **config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The config used by the model, will be used to grab the `hidden_size` of the model and the `layer_norm_eps` to use.

A SQuAD head inspired by XLNet.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L4193)

( hidden\_states: FloatTensorstart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Nonecls\_index: typing.Optional\[torch.LongTensor\] = Noneis\_impossible: typing.Optional\[torch.LongTensor\] = Nonep\_mask: typing.Optional\[torch.FloatTensor\] = Nonereturn\_dict: bool = False ) → [transformers.modeling\_utils.SquadHeadOutput](/docs/transformers/v4.34.0/en/internal/modeling_utils#transformers.modeling_utils.SquadHeadOutput) or `tuple(torch.FloatTensor)`

### class transformers.modeling\_utils.SequenceSummary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L4291)

( config: PretrainedConfig )

Compute a single vector summary of a sequence hidden states.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_utils.py#L4346)

( hidden\_states: FloatTensorcls\_index: typing.Optional\[torch.LongTensor\] = None ) → `torch.FloatTensor`

Parameters

-   **hidden\_states** (`torch.FloatTensor` of shape `[batch_size, seq_len, hidden_size]`) — The hidden states of the last layer.
-   **cls\_index** (`torch.LongTensor` of shape `[batch_size]` or `[batch_size, ...]` where … are optional leading dimensions of `hidden_states`, _optional_) — Used if `summary_type == "cls_index"` and takes the last token of the sequence as classification token.

Returns

`torch.FloatTensor`

The summary of the sequence hidden states.

Compute a single vector summary of a sequence hidden states.

## PyTorch Helper Functions

#### transformers.apply\_chunking\_to\_forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pytorch_utils.py#L168)

( forward\_fn: typing.Callable\[..., torch.Tensor\]chunk\_size: intchunk\_dim: int\*input\_tensors ) → `torch.Tensor`

Parameters

-   **forward\_fn** (`Callable[..., torch.Tensor]`) — The forward function of the model.
-   **chunk\_size** (`int`) — The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
-   **chunk\_dim** (`int`) — The dimension over which the `input_tensors` should be chunked.
-   **input\_tensors** (`Tuple[torch.Tensor]`) — The input tensors of `forward_fn` which will be chunked

A tensor with the same shape as the `forward_fn` would have given if applied\`.

This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly applying `forward_fn` to `input_tensors`.

Examples:

```
def forward_chunk(self, hidden_states):
    hidden_states = self.decoder(hidden_states)
    return hidden_states



def forward(self, hidden_states):
    return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
```

#### transformers.pytorch\_utils.find\_pruneable\_heads\_and\_indices

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pytorch_utils.py#L243)

( heads: typing.List\[int\]n\_heads: inthead\_size: intalready\_pruned\_heads: typing.Set\[int\] ) → `Tuple[Set[int], torch.LongTensor]`

Parameters

-   **heads** (`List[int]`) — List of the indices of heads to prune.
-   **n\_heads** (`int`) — The number of heads in the model.
-   **head\_size** (`int`) — The size of each head.
-   **already\_pruned\_heads** (`Set[int]`) — A set of already pruned heads.

Returns

`Tuple[Set[int], torch.LongTensor]`

A tuple with the indices of heads to prune taking `already_pruned_heads` into account and the indices of rows/columns to keep in the layer weight.

Finds the heads and their indices taking `already_pruned_heads` into account.

#### transformers.prune\_layer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pytorch_utils.py#L144)

( layer: typing.Union\[torch.nn.modules.linear.Linear, transformers.pytorch\_utils.Conv1D\]index: LongTensordim: typing.Optional\[int\] = None ) → `torch.nn.Linear` or [Conv1D](/docs/transformers/v4.34.0/en/internal/modeling_utils#transformers.Conv1D)

Parameters

-   **layer** (`Union[torch.nn.Linear, Conv1D]`) — The layer to prune.
-   **index** (`torch.LongTensor`) — The indices to keep in the layer.
-   **dim** (`int`, _optional_) — The dimension on which to keep the indices.

Returns

`torch.nn.Linear` or [Conv1D](/docs/transformers/v4.34.0/en/internal/modeling_utils#transformers.Conv1D)

The pruned layer as a new layer with `requires_grad=True`.

Prune a Conv1D or linear layer to keep only entries in index.

Used to remove heads.

#### transformers.pytorch\_utils.prune\_conv1d\_layer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pytorch_utils.py#L111)

( layer: Conv1Dindex: LongTensordim: int = 1 ) → [Conv1D](/docs/transformers/v4.34.0/en/internal/modeling_utils#transformers.Conv1D)

Parameters

-   **layer** ([Conv1D](/docs/transformers/v4.34.0/en/internal/modeling_utils#transformers.Conv1D)) — The layer to prune.
-   **index** (`torch.LongTensor`) — The indices to keep in the layer.
-   **dim** (`int`, _optional_, defaults to 1) — The dimension on which to keep the indices.

The pruned layer as a new layer with `requires_grad=True`.

Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.

Used to remove heads.

#### transformers.pytorch\_utils.prune\_linear\_layer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pytorch_utils.py#L52)

( layer: Linearindex: LongTensordim: int = 0 ) → `torch.nn.Linear`

Parameters

-   **layer** (`torch.nn.Linear`) — The layer to prune.
-   **index** (`torch.LongTensor`) — The indices to keep in the layer.
-   **dim** (`int`, _optional_, defaults to 0) — The dimension on which to keep the indices.

The pruned layer as a new layer with `requires_grad=True`.

Prune a linear layer to keep only entries in index.

Used to remove heads.

## TensorFlow custom layers

### class transformers.modeling\_tf\_utils.TFConv1D

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L3185)

( \*args\*\*kwargs )

Parameters

-   **nf** (`int`) — The number of output features.
-   **nx** (`int`) — The number of input features.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation to use to initialize the weights.
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional keyword arguments passed along to the `__init__` of `tf.keras.layers.Layer`.

1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

Basically works like a linear layer but the weights are transposed.

### class transformers.TFSequenceSummary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L3328)

( \*args\*\*kwargs )

Compute a single vector summary of a sequence hidden states.

## TensorFlow loss functions

### class transformers.modeling\_tf\_utils.TFCausalLanguageModelingLoss

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L192)

( )

Loss function suitable for causal language modeling (CLM), that is, the task of guessing the next token.

Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

### class transformers.modeling\_tf\_utils.TFMaskedLanguageModelingLoss

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L311)

( )

Loss function suitable for masked language modeling (MLM), that is, the task of guessing the masked tokens.

Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

### class transformers.modeling\_tf\_utils.TFMultipleChoiceLoss

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L301)

( )

Loss function suitable for multiple choice tasks.

### class transformers.modeling\_tf\_utils.TFQuestionAnsweringLoss

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L223)

( )

Loss function suitable for question answering.

### class transformers.modeling\_tf\_utils.TFSequenceClassificationLoss

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L282)

( )

Loss function suitable for sequence classification.

### class transformers.modeling\_tf\_utils.TFTokenClassificationLoss

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L238)

( )

Loss function suitable for token classification.

Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

## TensorFlow Helper Functions

#### transformers.modeling\_tf\_utils.get\_initializer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L3444)

( initializer\_range: float = 0.02 ) → `tf.keras.initializers.TruncatedNormal`

Parameters

-   **initializer\_range** (_float_, defaults to 0.02) — Standard deviation of the initializer range.

Returns

`tf.keras.initializers.TruncatedNormal`

The truncated normal initializer.

Creates a `tf.keras.initializers.TruncatedNormal` with the given range.

#### transformers.modeling\_tf\_utils.keras\_serializable

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_utils.py#L127)

( )

Parameters

-   **cls** (a `tf.keras.layers.Layers subclass`) — Typically a `TF.MainLayer` class in this project, in general must accept a `config` argument to its initializer.

Decorate a Keras Layer class to support Keras serialization.

This is done by:

1.  Adding a `transformers_config` dict to the Keras config dictionary in `get_config` (called by Keras at serialization time.
2.  Wrapping `__init__` to accept that `transformers_config` dict (passed by Keras at deserialization time) and convert it to a config object for the actual layer initializer.
3.  Registering the class as a custom object in Keras (if the Tensorflow version supports this), so that it does not need to be supplied in `custom_objects` in the call to `tf.keras.models.load_model`.

#### transformers.shape\_list

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tf_utils.py#L26)

( tensor: typing.Union\[tensorflow.python.framework.ops.Tensor, numpy.ndarray\] ) → `List[int]`

Parameters

-   **tensor** (`tf.Tensor` or `np.ndarray`) — The tensor we want the shape of.

The shape of the tensor as a list.

Deal with dynamic shape in tensorflow cleanly.