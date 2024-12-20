# Utilities for Generation

This page lists all the utility functions used by [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate), [greedy\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.greedy_search), [contrastive\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.contrastive_search), [sample()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.sample), [beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search), [beam\_sample()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample), [group\_beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.group_beam_search), and [constrained\_beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.constrained_beam_search).

Most of those are only useful if you are studying the code of the generate methods in the library.

## Generate Outputs

The output of [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate) is an instance of a subclass of [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput). This output is a data structure containing all the information returned by [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate), but that can also be used as tuple or dictionary.

Here’s an example:

```
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
```

The `generation_output` object is a [GreedySearchDecoderOnlyOutput](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.GreedySearchDecoderOnlyOutput), as we can see in the documentation of that class below, it means it has the following attributes:

-   `sequences`: the generated sequences of tokens
-   `scores` (optional): the prediction scores of the language modelling head, for each generation step
-   `hidden_states` (optional): the hidden states of the model, for each generation step
-   `attentions` (optional): the attention weights of the model, for each generation step

Here we have the `scores` since we passed along `output_scores=True`, but we don’t have `hidden_states` and `attentions` because we didn’t pass `output_hidden_states=True` or `output_attentions=True`.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you will get `None`. Here for instance `generation_output.scores` are all the generated prediction scores of the language modeling head, and `generation_output.attentions` is `None`.

When using our `generation_output` object as a tuple, it only keeps the attributes that don’t have `None` values. Here, for instance, it has two elements, `loss` then `logits`, so

will return the tuple `(generation_output.sequences, generation_output.scores)` for instance.

When using our `generation_output` object as a dictionary, it only keeps the attributes that don’t have `None` values. Here, for instance, it has two keys that are `sequences` and `scores`.

We document here all output types.

### PyTorch

### class transformers.generation.GreedySearchEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L184)

( sequences: LongTensor = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention weights of the decoder (respectively the encoder) can be accessed via the encoder\_attentions and the encoder\_hidden\_states attributes (respectively the decoder\_attentions and the decoder\_hidden\_states attributes)

### class transformers.generation.GreedySearchDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L88)

( sequences: LongTensor = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of decoder-only generation models using greedy search.

### class transformers.generation.SampleEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L255)

( sequences: LongTensor = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of encoder-decoder generation models using sampling. Hidden states and attention weights of the decoder (respectively the encoder) can be accessed via the encoder\_attentions and the encoder\_hidden\_states attributes (respectively the decoder\_attentions and the decoder\_hidden\_states attributes)

### class transformers.generation.SampleDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L226)

( sequences: LongTensor = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of decoder-only generation models using sampling.

### class transformers.generation.BeamSearchEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L333)

( sequences: LongTensor = Nonesequences\_scores: typing.Optional\[torch.FloatTensor\] = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonebeam\_indices: typing.Optional\[torch.LongTensor\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights of the decoder (respectively the encoder) can be accessed via the encoder\_attentions and the encoder\_hidden\_states attributes (respectively the decoder\_attentions and the decoder\_hidden\_states attributes)

### class transformers.generation.BeamSearchDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L298)

( sequences: LongTensor = Nonesequences\_scores: typing.Optional\[torch.FloatTensor\] = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonebeam\_indices: typing.Optional\[torch.LongTensor\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of decoder-only generation models using beam search.

### class transformers.generation.BeamSampleEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L418)

( sequences: LongTensor = Nonesequences\_scores: typing.Optional\[torch.FloatTensor\] = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonebeam\_indices: typing.Optional\[torch.LongTensor\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of encoder-decoder generation models using beam sampling. Hidden states and attention weights of the decoder (respectively the encoder) can be accessed via the encoder\_attentions and the encoder\_hidden\_states attributes (respectively the decoder\_attentions and the decoder\_hidden\_states attributes)

### class transformers.generation.BeamSampleDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L383)

( sequences: LongTensor = Nonesequences\_scores: typing.Optional\[torch.FloatTensor\] = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonebeam\_indices: typing.Optional\[torch.LongTensor\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of decoder-only generation models using beam sample.

### class transformers.generation.ContrastiveSearchEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L116)

( sequences: LongTensor = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of decoder-only generation models using contrastive search.

### class transformers.generation.ContrastiveSearchDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L155)

( sequences: LongTensor = Nonescores: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None )

Base class for outputs of decoder-only generation models using contrastive search.

### TensorFlow

### class transformers.generation.TFGreedySearchEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L86)

( sequences: Tensor = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of encoder-decoder generation models using greedy search. Hidden states and attention weights of the decoder (respectively the encoder) can be accessed via the encoder\_attentions and the encoder\_hidden\_states attributes (respectively the decoder\_attentions and the decoder\_hidden\_states attributes)

### class transformers.generation.TFGreedySearchDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L58)

( sequences: Tensor = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of decoder-only generation models using greedy search.

### class transformers.generation.TFSampleEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L156)

( sequences: Tensor = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of encoder-decoder generation models using sampling. Hidden states and attention weights of the decoder (respectively the encoder) can be accessed via the encoder\_attentions and the encoder\_hidden\_states attributes (respectively the decoder\_attentions and the decoder\_hidden\_states attributes)

### class transformers.generation.TFSampleDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L128)

( sequences: Tensor = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of decoder-only generation models using sampling.

### class transformers.generation.TFBeamSearchEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L233)

( sequences: Tensor = Nonesequences\_scores: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Nonebeam\_indices: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of encoder-decoder generation models using beam search. Hidden states and attention weights of the decoder (respectively the encoder) can be accessed via the encoder\_attentions and the encoder\_hidden\_states attributes (respectively the decoder\_attentions and the decoder\_hidden\_states attributes)

### class transformers.generation.TFBeamSearchDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L198)

( sequences: Tensor = Nonesequences\_scores: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Nonebeam\_indices: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of decoder-only generation models using beam search.

### class transformers.generation.TFBeamSampleEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L318)

( sequences: Tensor = Nonesequences\_scores: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Nonebeam\_indices: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of encoder-decoder generation models using beam sampling. Hidden states and attention weights of the decoder (respectively the encoder) can be accessed via the encoder\_attentions and the encoder\_hidden\_states attributes (respectively the decoder\_attentions and the decoder\_hidden\_states attributes)

### class transformers.generation.TFBeamSampleDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L283)

( sequences: Tensor = Nonesequences\_scores: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Nonebeam\_indices: typing.Optional\[tensorflow.python.framework.ops.Tensor\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of decoder-only generation models using beam sample.

### class transformers.generation.TFContrastiveSearchEncoderDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L394)

( sequences: Tensor = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of encoder-decoder generation models using contrastive search. Hidden states and attention weights of the decoder (respectively the encoder) can be accessed via the encoder\_attentions and the encoder\_hidden\_states attributes (respectively the decoder\_attentions and the decoder\_hidden\_states attributes)

### class transformers.generation.TFContrastiveSearchDecoderOnlyOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L367)

( sequences: Tensor = Nonescores: typing.Optional\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = Nonehidden\_states: typing.Optional\[typing.Tuple\[typing.Tuple\[tensorflow.python.framework.ops.Tensor\]\]\] = None )

Base class for outputs of decoder-only generation models using contrastive search.

### FLAX

### class transformers.generation.FlaxSampleOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_utils.py#L69)

( sequences: Array = None )

Parameters

-   **sequences** (`jnp.ndarray` of shape `(batch_size, max_length)`) — The generated sequences.

Flax Base class for outputs of decoder-only generation models using sampling.

“Returns a new object replacing the specified fields with new values.

### class transformers.generation.FlaxGreedySearchOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_utils.py#L55)

( sequences: Array = None )

Parameters

-   **sequences** (`jnp.ndarray` of shape `(batch_size, max_length)`) — The generated sequences.

Flax Base class for outputs of decoder-only generation models using greedy search.

“Returns a new object replacing the specified fields with new values.

### class transformers.generation.FlaxBeamSearchOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_utils.py#L83)

( sequences: Array = Nonescores: Array = None )

Parameters

-   **sequences** (`jnp.ndarray` of shape `(batch_size, max_length)`) — The generated sequences.
-   **scores** (`jnp.ndarray` of shape `(batch_size,)`) — The scores (log probabilities) of the generated sequences.

Flax Base class for outputs of decoder-only generation models using greedy search.

“Returns a new object replacing the specified fields with new values.

## LogitsProcessor

A [LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) can be used to modify the prediction scores of a language model head for generation.

### PyTorch

### class transformers.AlternatingCodebooksLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1553)

( input\_start\_len: intsemantic\_vocab\_size: intcodebook\_size: int )

Parameters

-   **input\_start\_len** (`int`) — The length of the initial input sequence.
-   **semantic\_vocab\_size** (`int`) — Vocabulary size of the semantic part, i.e number of tokens associated to the semantic vocabulary.
-   **codebook\_size** (`int`) — Number of tokens associated to the codebook.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) enforcing alternated generation between the two codebooks of `Bark`’s fine submodel.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1574)

( input\_ids: LongTensorscores: FloatTensor )

### class transformers.ClassifierFreeGuidanceLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1513)

( guidance\_scale )

Parameters

-   **guidance\_scale** (float) — The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality.

Logits processor for classifier free guidance (CFG). The scores are split over the batch dimension, where the first half correspond to the conditional logits (predicted from the input prompt) and the second half correspond to the unconditional logits (predicted from an empty or ‘null’ prompt). The processor computes a weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.

See [the paper](https://arxiv.org/abs/2306.05284) for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1537)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.EncoderNoRepeatNGramLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L763)

( encoder\_ngram\_size: intencoder\_input\_ids: LongTensor )

Parameters

-   **encoder\_ngram\_size** (`int`) — All ngrams of size `ngram_size` can only occur within the encoder input ids.
-   **encoder\_input\_ids** (`int`) — The encoder\_input\_ids that should not be repeated within the decoder ids.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that enforces no repetition of encoder input ids n-grams for the decoder ids. See [ParlAI](https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1350).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L786)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.EncoderRepetitionPenaltyLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L323)

( penalty: floatencoder\_input\_ids: LongTensor )

Parameters

-   **hallucination\_penalty** (`float`) — The parameter for hallucination penalty. 1.0 means no penalty.
-   **encoder\_input\_ids** (`torch.LongTensor`) — The encoder\_input\_ids that should be repeated within the decoder ids.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) enforcing an exponential penalty on tokens that are not in the original input.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L341)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.EpsilonLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L493)

( epsilon: floatfilter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **epsilon** (`float`) — If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
-   **filter\_value** (`float`, _optional_, defaults to `-float("Inf")`) — All filtered values will be set to this float value.
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimum number of tokens that cannot be filtered.

[LogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper) that performs epsilon-sampling, i.e. restricting to tokens with `prob >= epsilon`. Takes the largest min\_tokens\_to\_keep tokens if no tokens satisfy this constraint. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more information.

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(0)
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> 
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

>>> 
>>> 
>>> 
>>> outputs = model.generate(**inputs, do_sample=True, epsilon_cutoff=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L546)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.EtaLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L560)

( epsilon: floatfilter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **epsilon** (`float`) — A float value in the range (0, 1). Hyperparameter used to calculate the dynamic cutoff value, `eta`. The suggested values from the paper ranges from 3e-4 to 4e-3 depending on the size of the model.
-   **filter\_value** (`float`, _optional_, defaults to `-float("Inf")`) — All values that are found to be below the dynamic cutoff value, `eta`, are set to this float value. This parameter is useful when logits need to be modified for very low probability tokens that should be excluded from generation entirely.
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Specifies the minimum number of tokens that must be kept for generation, regardless of their probabilities. For example, if `min_tokens_to_keep` is set to 1, at least one token will always be kept for generation, even if all tokens have probabilities below the cutoff `eta`.

[LogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper) that performs eta-sampling, a technique to filter out tokens with probabilities below a dynamic cutoff value, `eta`, which is calculated based on a combination of the hyperparameter `epsilon` and the entropy of the token probabilities, i.e. `eta := min(epsilon, sqrt(epsilon * e^-entropy(probabilities)))`. Takes the largest min\_tokens\_to\_keep tokens if no tokens satisfy this constraint. It addresses the issue of poor quality in long samples of text generated by neural language models leading to more coherent and fluent text. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more information. Note: `do_sample` must be set to `True` for this `LogitsWarper` to work.

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(0)
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> 
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

>>> 
>>> 
>>> 
>>> outputs = model.generate(**inputs, do_sample=True, eta_cutoff=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L623)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.ExponentialDecayLengthPenalty

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1298)

( exponential\_decay\_length\_penalty: typing.Tuple\[int, float\]eos\_token\_id: typing.Union\[int, typing.List\[int\]\]input\_ids\_seq\_length: int )

Parameters

-   **exponential\_decay\_length\_penalty** (`tuple(int, float)`) — This tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay
-   **eos\_token\_id** (`Union[int, List[int]]`) — The id of the _end-of-sequence_ token. Optionally, use a list to set multiple _end-of-sequence_ tokens.
-   **input\_ids\_seq\_length** (`int`) — The length of the input sequence.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that exponentially increases the score of the `eos_token_id` after `start_index` has been reached. This allows generating shorter sequences without having a hard cutoff, allowing the `eos_token` to be predicted in a meaningful position.

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(1)
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")

>>> text = "Just wanted to let you know, I"
>>> inputs = tokenizer(text, return_tensors="pt")

>>> 
>>> 
>>> outputs = model.generate(**inputs, do_sample=True, temperature=0.9, max_length=30, pad_token_id=50256)
>>> print(tokenizer.batch_decode(outputs)[0])
Just wanted to let you know, I'm not even a lawyer. I'm a man. I have no real knowledge of politics. I'm a

>>> # Generate sequences with exponential penalty, we add the exponential_decay_length_penalty=(start_index, decay_factor)
>>> # We see that instead of cutting at max_tokens, the output comes to an end before (at 25 tokens) and with more meaning
>>> # What happens is that starting from `start_index` the EOS token score will be increased by decay_factor exponentially
>>> outputs = model.generate(
...     **inputs,
...     do_sample=True,
...     temperature=0.9,
...     max_length=30,
...     pad_token_id=50256,
...     exponential_decay_length_penalty=(15, 1.6),
... )
>>> print(tokenizer.batch_decode(outputs)[0])
Just wanted to let you know, I've got a very cool t-shirt educating people on how to use the Internet<|endoftext|>

>>> 
>>> outputs = model.generate(
...     **inputs,
...     do_sample=True,
...     temperature=0.9,
...     max_length=30,
...     pad_token_id=50256,
...     exponential_decay_length_penalty=(15, 1.05),
... )
>>> print(tokenizer.batch_decode(outputs)[0])
Just wanted to let you know, I've been working on it for about 6 months and now it's in Alpha.<|endoftext|>
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1371)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.ForcedBOSTokenLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1230)

( bos\_token\_id: int )

Parameters

-   **bos\_token\_id** (`int`) — The id of the token to force as the first generated token.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that enforces the specified token as the first generated token.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1242)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.ForcedEOSTokenLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1252)

( max\_length: inteos\_token\_id: typing.Union\[int, typing.List\[int\]\] )

Parameters

-   **max\_length** (`int`) — The maximum length of the sequence to be generated.
-   **eos\_token\_id** (`Union[int, List[int]]`) — The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a list to set multiple _end-of-sequence_ tokens.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that enforces the specified token as the last generated token when `max_length` is reached.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1270)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.ForceTokensLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1428)

( force\_token\_map: typing.List\[typing.List\[int\]\] )

This processor takes a list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. The processor will set their log probs to `inf` so that they are sampled at their corresponding index.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1436)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.HammingDiversityLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1067)

( diversity\_penalty: floatnum\_beams: intnum\_beam\_groups: int )

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that enforces diverse beam search.

Note that this logits processor is only effective for [PreTrainedModel.group\_beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.group_beam_search). See [Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models](https://arxiv.org/pdf/1610.02424.pdf) for more details.

Diverse beam search can be particularly useful in scenarios where a variety of different outputs is desired, rather than multiple similar sequences. It allows the model to explore different generation paths and provides a broader coverage of possible outputs.

This logits processor can be resource-intensive, especially when using large models or long sequences.

Traditional beam search often generates very similar sequences across different beams. `HammingDiversityLogitsProcessor` addresses this by penalizing beams that generate tokens already chosen by other beams in the same time step.

How It Works:

-   **Grouping Beams**: Beams are divided into groups. Each group selects tokens independently of the others.
-   **Penalizing Repeated Tokens**: If a beam in a group selects a token already chosen by another group in the same step, a penalty is applied to that token’s score.
-   **Promoting Diversity**: This penalty discourages beams within a group from selecting the same tokens as beams in other groups.

Benefits:

-   **Diverse Outputs**: Produces a variety of different sequences.
-   **Exploration**: Allows the model to explore different paths.

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> import torch

>>> 
>>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

>>> 
>>> text = "The Solar System is a gravitationally bound system comprising the Sun and the objects that orbit it, either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud."
>>> inputs = tokenizer("summarize: " + text, return_tensors="pt")

>>> 
>>> outputs_diverse = model.generate(
...     **inputs,
...     num_beam_groups=2,
...     diversity_penalty=10.0,
...     max_length=100,
...     num_beams=4,
...     num_return_sequences=2,
... )
>>> summaries_diverse = tokenizer.batch_decode(outputs_diverse, skip_special_tokens=True)

>>> 
>>> outputs_non_diverse = model.generate(
...     **inputs,
...     max_length=100,
...     num_beams=4,
...     num_return_sequences=2,
... )
>>> summary_non_diverse = tokenizer.batch_decode(outputs_non_diverse, skip_special_tokens=True)

>>> 
>>> print(summary_non_diverse)
['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
'the Solar System formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.']

>>> print(summaries_diverse)
['the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets.',
'the solar system formed 4.6 billion years ago from the collapse of a giant interstellar molecular cloud. of the objects that orbit the Sun directly, the largest are the eight planets. the rest of the objects are smaller objects, such as the five dwarf planets and small solar system bodies.']
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1184)

( input\_ids: LongTensorscores: FloatTensorcurrent\_tokens: LongTensorbeam\_group\_idx: int ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search
-   **current\_tokens** (`torch.LongTensor` of shape `(batch_size)`) — Indices of input sequence tokens in the vocabulary, corresponding to the tokens selected by the other beam groups in the current generation step.
-   **beam\_group\_idx** (`int`) — The index of the beam group currently being processed.

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.InfNanRemoveLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1281)

( )

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that removes all `nan` and `inf` values to avoid the generation method to fail. Note that using the logits processor should only be used if necessary since it can slow down the generation method.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1287)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.LogitNormalization

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1382)

( )

[LogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper) and [LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) for normalizing the scores using log-softmax. It’s important to normalize the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in this library doesn’t do it (it only does it before, but they may need re-normalization) but it still supposes that the scores are normalized when comparing the hypotheses.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1390)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

Abstract base class for all logit processors that can be applied during generation.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L47)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.LogitsProcessorList

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L64)

( iterable = () )

This class can be used to create a list of [LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) or [LogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper) to subsequently process a `scores` input tensor. This class inherits from list and adds a specific _**call**_ method to apply each [LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) or [LogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper) to the inputs.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L71)

( input\_ids: LongTensorscores: FloatTensor\*\*kwargs ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional kwargs that are specific to a logits processor.

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

Abstract base class for all logit warpers that can be applied during generation with multinomial sampling.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L57)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.MinLengthLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L101)

( min\_length: inteos\_token\_id: typing.Union\[int, typing.List\[int\]\] )

Parameters

-   **min\_length** (`int`) — The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
-   **eos\_token\_id** (`Union[int, List[int]]`) — The id of the _end-of-sequence_ token. Optionally, use a list to set multiple _end-of-sequence_ tokens.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) enforcing a min-length by setting EOS probability to 0.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L124)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.MinNewTokensLengthLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L133)

( prompt\_length\_to\_skip: intmin\_new\_tokens: inteos\_token\_id: typing.Union\[int, typing.List\[int\]\] )

Parameters

-   **prompt\_length\_to\_skip** (`int`) — The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the input length.
-   **min\_new\_tokens** (`int`) — The minimum _new_ tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
-   **eos\_token\_id** (`Union[int, List[int]]`) — The id of the _end-of-sequence_ token. Optionally, use a list to set multiple _end-of-sequence_ tokens.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0. Note that for decoder-only models, such as Llama2, `min_length` will compute the length of `prompt + newly generated tokens` whereas for other models it will behave as `min_new_tokens`, that is, taking only into account the newly generated ones.

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> model.config.pad_token_id = model.config.eos_token_id
>>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

>>> 
>>> outputs = model.generate(**inputs, min_new_tokens=30)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Hugging Face Company is a company that has been working on a new product for the past year.

>>> 
>>> 
>>> outputs = model.generate(**inputs, eos_token_id=1664)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Hugging Face Company is a company

>>> 
>>> 
>>> outputs = model.generate(**inputs, min_new_tokens=2, eos_token_id=1664)
>>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Hugging Face Company is a new company
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L195)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.NoBadWordsLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L953)

( bad\_words\_ids: typing.List\[typing.List\[int\]\]eos\_token\_id: typing.Union\[int, typing.List\[int\]\] )

Parameters

-   **bad\_words\_ids** (`List[List[int]]`) — List of list of token ids that are not allowed to be generated.
-   **eos\_token\_id** (`Union[int, List[int]]`) — The id of the _end-of-sequence_ token. Optionally, use a list to set multiple _end-of-sequence_ tokens.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that enforces that specified sequences will never be selected.

In order to get the token ids of the words that should not appear in the generated text, make sure to set `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers’ prefixing behaviours come from `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> inputs = tokenizer(["In a word, the cake is a"], return_tensors="pt")

>>> output_ids = model.generate(inputs["input_ids"], max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
>>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
In a word, the cake is a bit of a mess.

>>> 
>>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


>>> def get_tokens_as_list(word_list):
...     "Converts a sequence of words into a list of tokens"
...     tokens_list = []
...     for word in word_list:
...         tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
...         tokens_list.append(tokenized_word)
...     return tokens_list


>>> bad_words_ids = get_tokens_as_list(word_list=["mess"])
>>> output_ids = model.generate(
...     inputs["input_ids"], max_new_tokens=5, bad_words_ids=bad_words_ids, pad_token_id=tokenizer.eos_token_id
... )
>>> print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
In a word, the cake is a bit of a surprise.
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L876)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.NoRepeatNGramLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L706)

( ngram\_size: int )

Parameters

-   **ngram\_size** (`int`) — All ngrams of size `ngram_size` can only occur once.

N-grams are groups of “n” consecutive words, characters, or tokens taken from a sequence of text. Given the sentence: “She runs fast”, the bi-grams (n=2) would be (“she”, “runs”) and (“runs”, “fast”). In text generation, avoiding repetitions of word sequences provides a more diverse output. This [LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) enforces no repetition of n-grams by setting the scores of banned tokens to negative infinity which eliminates those tokens from consideration when further processing the scores. [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

Use n-gram penalties with care. For instance, penalizing 2-grams (bigrams) in an article about the city of New York might lead to undesirable outcomes where the city’s name appears only once in the entire text. [Reference](https://huggingface.co/blog/how-to-generate)

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
>>> inputs = tokenizer(["Today I"], return_tensors="pt")

>>> output = model.generate(**inputs)
>>> print(tokenizer.decode(output[0], skip_special_tokens=True))
Today I’m not sure if I’m going to be able to do it.

>>> 
>>> output = model.generate(**inputs, no_repeat_ngram_size=2)
>>> print(tokenizer.decode(output[0], skip_special_tokens=True))
Today I’m not sure if I can get a better understanding of the nature of this issue
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L752)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.PrefixConstrainedLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1040)

( prefix\_allowed\_tokens\_fn: typing.Callable\[\[int, torch.Tensor\], typing.List\[int\]\]num\_beams: int )

Parameters

-   **prefix\_allowed\_tokens\_fn** (`Callable[[int, torch.Tensor], List[int]]`) — This function constraints the beam search to allowed tokens only at each step. This function takes 2 arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID `batch_id`.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that enforces constrained generation and is useful for prefix-conditioned constrained generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1057)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.RepetitionPenaltyLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L270)

( penalty: float )

Parameters

-   **repetition\_penalty** (`float`) — The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that prevents the repetition of previous tokens through an exponential penalty. This technique shares some similarities with coverage mechanisms and other aimed at reducing repetition. During the text generation process, the probability distribution for the next token is determined using a formula that incorporates token scores based on their occurrence in the generated sequence. Tokens with higher scores are more likely to be selected. The formula can be seen in the original [paper](https://arxiv.org/pdf/1909.05858.pdf). According to the paper a penalty of around 1.2 yields a good balance between truthful generation and lack of repetition.

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> 
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
>>> inputs = tokenizer(["I'm not going to"], return_tensors="pt")

>>> 
>>> summary_ids = model.generate(**inputs)
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
I'm not going to be able to do that. I'm going to be able to do that

>>> 
>>> penalized_ids = model.generate(**inputs, repetition_penalty=1.1)
>>> print(tokenizer.batch_decode(penalized_ids, skip_special_tokens=True)[0])
I'm not going to be able to do that. I'll just have to go out and play
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L312)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.SequenceBiasLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L805)

( sequence\_bias: typing.Dict\[typing.Tuple\[int\], float\] )

Parameters

-   **sequence\_bias** (`Dict[Tuple[int], float]`) — Dictionary that maps a sequence of tokens to its bias term. Positive biases increase the odds of the sequence being selected, while negative biases do the opposite. If a sequence has a length of 1, its bias will always be applied. Otherwise, the bias will only be applied if the sequence in question is about to be completed (in the token selection step after this processor is applied).

[LogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor) that applies an additive bias on sequences. The bias is applied to the last token of a sequence when the next generated token can complete it. Consequently, to take the most of biasing sequences with more than one token, consider using beam methods (to gracefully work around partially completed sequences that have a negative bias) and applying the bias to their prefixes (to ensure the bias is applied earlier).

In order to get the token ids of the sequences that you want to bias, make sure to set `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers’ prefixing behaviours come from `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> inputs = tokenizer(["The full name of Donald is Donald"], return_tensors="pt")

>>> summary_ids = model.generate(inputs["input_ids"], max_new_tokens=4)
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald J. Trump Jr

>>> 
>>> tokenizer_with_prefix_space = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)


>>> def get_tokens_as_tuple(word):
...     return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])


>>> 
>>> sequence_bias = {get_tokens_as_tuple("Trump"): -10.0}
>>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, sequence_bias=sequence_bias)
>>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald J. Donald,

>>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
>>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald Rumsfeld,

>>> 
>>> sequence_bias = {get_tokens_as_tuple("Donald Duck"): 10.0}
>>> biased_ids = model.generate(inputs["input_ids"], max_new_tokens=4, num_beams=4, sequence_bias=sequence_bias)
>>> print(tokenizer.batch_decode(biased_ids, skip_special_tokens=True)[0])
The full name of Donald is Donald Duck.
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L876)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.SuppressTokensAtBeginLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1396)

( begin\_suppress\_tokensbegin\_index )

[SuppressTokensAtBeginLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.SuppressTokensAtBeginLogitsProcessor) supresses a list of tokens as soon as the `generate` function starts generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` at not sampled at the begining of the generation.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1407)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.SuppressTokensLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1415)

( suppress\_tokens )

This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so that they are not sampled.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1422)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.TemperatureLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L205)

( temperature: float )

Parameters

-   **temperature** (`float`) — Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely token.

[LogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper) for temperature (exponential scaling output probability distribution), which effectively means that it can control the randomness of the predicted tokens.

Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won’t have any effect.

Examples:

```
>>> import torch
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(0)  

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> model.config.pad_token_id = model.config.eos_token_id
>>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

>>> 
>>> generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
>>> outputs = model.generate(**inputs, **generate_kwargs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Hugging Face Company is a joint venture between GEO Group, one of',
'Hugging Face Company is not an exact science – but what we believe does']

>>> 
>>> generate_kwargs["temperature"] = 0.0001
>>> outputs = model.generate(**inputs, **generate_kwargs)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
['Hugging Face Company is a company that has been around for over 20 years',
'Hugging Face Company is a company that has been around for over 20 years']
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L264)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.TopKLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L415)

( top\_k: intfilter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **top\_k** (`int`) — The number of highest probability vocabulary tokens to keep for top-k-filtering.
-   **filter\_value** (`float`, _optional_, defaults to `-float("Inf")`) — All filtered values will be set to this float value.
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimum number of tokens that cannot be filtered.

[LogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper) that performs top-k, i.e. restricting to the k highest probability elements.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L435)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.TopPLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L352)

( top\_p: floatfilter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **top\_p** (`float`) — If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
-   **filter\_value** (`float`, _optional_, defaults to `-float("Inf")`) — All filtered values will be set to this float value.
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimum number of tokens that cannot be filtered.

[LogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper) that performs top-p, i.e. restricting to top tokens summing to prob\_cut\_off <= prob\_cut\_off.

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

>>> set_seed(0)
>>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

>>> inputs = tokenizer("A sequence: 1, 2", return_tensors="pt")

>>> 
>>> outputs = model.generate(**inputs, do_sample=True)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 0, 2, 2. 2, 2, 2, 2

>>> 
>>> 
>>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
>>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
```

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L399)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.TypicalLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L444)

( mass: float = 0.9filter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **mass** (`float`) — Value of typical\_p between 0 and 1 inclusive, defaults to 0.9.
-   **filter\_value** (`float`, _optional_, defaults to `-float("Inf")`) — All filtered values will be set to this float value.
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimum number of tokens that cannot be filtered.

[LogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper) that performs typical decoding. See [Typical Decoding for Natural Language Generation](https://arxiv.org/abs/2202.00666) for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L469)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### class transformers.UnbatchedClassifierFreeGuidanceLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1589)

( guidance\_scale: floatmodelunconditional\_ids: typing.Optional\[torch.LongTensor\] = Noneunconditional\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = True )

Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`. The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

See [the paper](https://arxiv.org/abs/2306.17806) for more information.

Examples:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="pt")
>>> out = model.generate(inputs["input_ids"], guidance_scale=1.5)
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

>>> 
>>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
>>> out = model.generate(inputs["input_ids"], guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

>>> 
>>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
>>> out = model.generate(inputs["input_ids"], guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
>>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
"Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
```

### class transformers.WhisperTimeStampLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1446)

( generate\_config )

Parameters

-   **generate\_config** (`GenerateConfig`) — The generate config used to generate the output. The following parameters are required: eos\_token\_id (`int`, _optional_, defaults to 50257): The id of the _end-of-sequence_ token. no\_timestamps\_token\_id (`int`, _optional_, defaults to 50363): The id of the `"<|notimestamps|>"` token. max\_initial\_timestamp\_index (`int`, _optional_, defaults to 1): Used to set the maximum value of the initial timestamp. This is used to prevent the model from predicting timestamps that are too far in the future.

Whisper specific Processor. This processor can be used to force a list of tokens. The processor will set their log probs to `inf` so that they are sampled at their corresponding index.

See [the paper](https://arxiv.org/abs/2212.04356) for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/logits_process.py#L1475)

( input\_ids: LongTensorscores: FloatTensor ) → `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search

Returns

`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

### TensorFlow

### class transformers.TFForcedBOSTokenLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L448)

( bos\_token\_id: int )

Parameters

-   **bos\_token\_id** (`int`) — The id of the token to force as the first generated token.

[TFLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsProcessor) that enforces the specified token as the first generated token.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L462)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFForcedEOSTokenLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L478)

( max\_length: inteos\_token\_id: int )

Parameters

-   **max\_length** (`int`) — The maximum length of the sequence to be generated.
-   **eos\_token\_id** (`int`) — The id of the token to force as the last generated token when `max_length` is reached.

[TFLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsProcessor) that enforces the specified token as the last generated token when `max_length` is reached.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L495)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFForceTokensLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L551)

( force\_token\_map: typing.List\[typing.List\[int\]\] )

This processor takes a list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. The processor will set their log probs to `0` and all other tokens to `-inf` so that they are sampled at their corresponding index.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L567)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L53)

( )

Abstract base class for all logit processors that can be applied during generation.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L56)

( input\_ids: Tensorscores: Tensorcur\_len: int ) → `tf.Tensor` of shape `(batch_size, config.vocab_size)`

TF method for processing logits.

### class transformers.TFLogitsProcessorList

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L75)

( iterable = () )

This class can be used to create a list of [TFLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsProcessor) to subsequently process a `scores` input tensor. This class inherits from list and adds a specific _**call**_ method to apply each [TFLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsProcessor) to the inputs.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L82)

( input\_ids: Tensorscores: Tensorcur\_len: int\*\*kwargs ) → `tf.Tensor` of shape `(batch_size, config.vocab_size)`

Abstract base class for all logit warpers that can be applied during generation with multinomial sampling.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L67)

( input\_ids: Tensorscores: Tensorcur\_len: int ) → `tf.Tensor` of shape `(batch_size, config.vocab_size)`

TF method for warping logits.

### class transformers.TFMinLengthLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L202)

( min\_length: inteos\_token\_id: int )

Parameters

-   **min\_length** (`int`) — The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
-   **eos\_token\_id** (`int`) — The id of the _end-of-sequence_ token.

[TFLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsProcessor) enforcing a min-length by setting EOS probability to 0.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L228)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFNoBadWordsLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L288)

( bad\_words\_ids: typing.List\[typing.List\[int\]\]eos\_token\_id: int )

Parameters

-   **bad\_words\_ids** (`List[List[int]]`) — List of list of token ids that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, make sure to set `add_prefix_space=True` when initializing the tokenizer, and use `tokenizer(bad_words, add_special_tokens=False).input_ids`. The `add_prefix_space` argument is only supported for some slow tokenizers, as fast tokenizers’ prefixing behaviours come from `pre tokenizers`. Read more [here](https://huggingface.co/docs/tokenizers/api/pre-tokenizers).
-   **eos\_token\_id** (`int`) — The id of the _end-of-sequence_ token.

[TFLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsProcessor) that enforces that specified sequences will never be sampled.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L367)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFNoRepeatNGramLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L388)

( ngram\_size: int )

Parameters

-   **ngram\_size** (`int`) — All ngrams of size `ngram_size` can only occur once.

[TFLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsProcessor) that enforces no repetition of n-grams. See [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L427)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFRepetitionPenaltyLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L238)

( penalty: float )

Parameters

-   **repetition\_penalty** (`float`) — The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.

[TFLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsProcessor) enforcing an exponential penalty on repeated sequences.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L280)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFSuppressTokensAtBeginLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L511)

( begin\_suppress\_tokensbegin\_index )

[TFSuppressTokensAtBeginLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFSuppressTokensAtBeginLogitsProcessor) suppresses a list of tokens as soon as the `generate` function starts generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` at not sampled at the begining of the generation.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L522)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFSuppressTokensLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L535)

( suppress\_tokens )

This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so that they are not sampled.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L542)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFTemperatureLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L98)

( temperature: float )

Parameters

-   **temperature** (`float`) — The value used to module the logits distribution.

[TFLogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsWarper) for temperature (exponential scaling output probability distribution).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L113)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFTopKLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L118)

( top\_k: intfilter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **top\_k** (`int`) — The number of highest probability vocabulary tokens to keep for top-k-filtering.
-   **filter\_value** (`float`, _optional_, defaults to `-float("Inf")`) — All filtered values will be set to this float value.
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimum number of tokens that cannot be filtered.

[TFLogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsWarper) that performs top-k, i.e. restricting to the k highest probability elements.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L138)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### class transformers.TFTopPLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L146)

( top\_p: floatfilter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **top\_p** (`float`) — If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
-   **filter\_value** (`float`, _optional_, defaults to `-float("Inf")`) — All filtered values will be set to this float value.
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimum number of tokens that cannot be filtered.

[TFLogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.TFLogitsWarper) that performs top-p, i.e. restricting to top tokens summing to <= prob\_cut\_off.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_logits_process.py#L170)

( input\_ids: Tensorscores: Tensorcur\_len: int )

### FLAX

### class transformers.FlaxForcedBOSTokenLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L194)

( bos\_token\_id: int )

Parameters

-   **bos\_token\_id** (`int`) — The id of the token to force as the first generated token.

[FlaxLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsProcessor) that enforces the specified token as the first generated token.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L206)

( input\_ids: Arrayscores: Arraycur\_len: int )

### class transformers.FlaxForcedEOSTokenLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L216)

( max\_length: inteos\_token\_id: int )

Parameters

-   **max\_length** (`int`) — The maximum length of the sequence to be generated.
-   **eos\_token\_id** (`int`) — The id of the token to force as the last generated token when `max_length` is reached.

[FlaxLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsProcessor) that enforces the specified token as the last generated token when `max_length` is reached.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L231)

( input\_ids: Arrayscores: Arraycur\_len: int )

### class transformers.FlaxForceTokensLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L315)

( force\_token\_map )

Parameters

-   **force\_token\_map** (`list`) — Map giving token ids and indices where they will be forced to be sampled.

[FlaxLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsProcessor) that takes a list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. The processor will set their log probs to 0 and all other tokens to `-inf` so that they are sampled at their corresponding index.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L337)

( input\_ids: Arrayscores: Arraycur\_len: int )

### class transformers.FlaxLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L50)

( )

Abstract base class for all logit processors that can be applied during generation.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L53)

( input\_ids: Arrayscores: Array ) → `jnp.ndarray` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`jnp.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **scores** (`jnp.ndarray` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional logits processor specific kwargs.

Returns

`jnp.ndarray` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

Flax method for processing logits.

### class transformers.FlaxLogitsProcessorList

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L72)

( iterable = () )

This class can be used to create a list of [FlaxLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsProcessor) or [FlaxLogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsWarper) to subsequently process a `scores` input tensor. This class inherits from list and adds a specific _**call**_ method to apply each [FlaxLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsProcessor) or [FlaxLogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsWarper) to the inputs.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L79)

( input\_ids: Arrayscores: Arraycur\_len: int\*\*kwargs ) → `jnp.ndarray` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`jnp.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **scores** (`jnp.ndarray` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional logits processor specific kwargs.

Returns

`jnp.ndarray` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

Abstract base class for all logit warpers that can be applied during generation with multinomial sampling.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L64)

( input\_ids: Arrayscores: Array ) → `jnp.ndarray` of shape `(batch_size, config.vocab_size)`

Parameters

-   **input\_ids** (`jnp.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **scores** (`jnp.ndarray` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam search or log softmax for each vocabulary token when using beam search
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional logits processor specific kwargs.

Returns

`jnp.ndarray` of shape `(batch_size, config.vocab_size)`

The processed prediction scores.

Flax method for warping logits.

### class transformers.FlaxMinLengthLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L241)

( min\_length: inteos\_token\_id: int )

Parameters

-   **min\_length** (`int`) — The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
-   **eos\_token\_id** (`int`) — The id of the _end-of-sequence_ token.

[FlaxLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsProcessor) enforcing a min-length by setting EOS probability to 0.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L262)

( input\_ids: Arrayscores: Arraycur\_len: int )

### class transformers.FlaxSuppressTokensAtBeginLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L271)

( begin\_suppress\_tokensbegin\_index )

Parameters

-   **begin\_suppress\_tokens** (`List[int]`) — Tokens to not sample.
-   **begin\_index** (`int`) — Index where the tokens are suppressed.

[FlaxLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsProcessor) supressing a list of tokens as soon as the `generate` function starts generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are not sampled at the begining of the generation.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L288)

( input\_idsscorescur\_len: int )

### class transformers.FlaxSuppressTokensLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L296)

( suppress\_tokens: list )

Parameters

-   **suppress\_tokens** (`list`) — Tokens to not sample.

[FlaxLogitsProcessor](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsProcessor) suppressing a list of tokens at each decoding step. The processor will set their log probs to be `-inf` so they are not sampled.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L309)

( input\_ids: Arrayscores: Arraycur\_len: int )

### class transformers.FlaxTemperatureLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L95)

( temperature: float )

Parameters

-   **temperature** (`float`) — The value used to module the logits distribution.

[FlaxLogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsWarper) for temperature (exponential scaling output probability distribution).

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L110)

( input\_ids: Arrayscores: Arraycur\_len: int )

### class transformers.FlaxTopKLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L159)

( top\_k: intfilter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **top\_k** (`int`) — The number of highest probability vocabulary tokens to keep for top-k-filtering.
-   **filter\_value** (`float`, _optional_, defaults to `-float("Inf")`) — All filtered values will be set to this float value.
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimum number of tokens that cannot be filtered.

[FlaxLogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsWarper) that performs top-k, i.e. restricting to the k highest probability elements.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L179)

( input\_ids: Arrayscores: Arraycur\_len: int )

### class transformers.FlaxTopPLogitsWarper

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L115)

( top\_p: floatfilter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **top\_p** (`float`) — If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
-   **filter\_value** (`float`, _optional_, defaults to `-float("Inf")`) — All filtered values will be set to this float value.
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimum number of tokens that cannot be filtered.

[FlaxLogitsWarper](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.FlaxLogitsWarper) that performs top-p, i.e. restricting to top tokens summing to prob\_cut\_off <= prob\_cut\_off.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L139)

( input\_ids: Arrayscores: Arraycur\_len: int )

### class transformers.FlaxWhisperTimeStampLogitsProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/flax_logits_process.py#L363)

( generate\_configmodel\_configdecoder\_input\_length )

Parameters

-   **generate\_config** (`GenerateConfig`) — The generate config used to generate the output. The following parameters are required: eos\_token\_id (`int`, _optional_, defaults to 50257): The id of the _end-of-sequence_ token. no\_timestamps\_token\_id (`int`, _optional_, defaults to 50363): The id of the `"<|notimestamps|>"` token. max\_initial\_timestamp\_index (`int`, _optional_, defaults to 1): Used to set the maximum value of the initial timestamp. This is used to prevent the model from predicting timestamps that are too far in the future.

Whisper specific Processor. This processor can be used to force a list of tokens. The processor will set their log probs to `inf` so that they are sampled at their corresponding index.

## StoppingCriteria

A [StoppingCriteria](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteria) can be used to change when to stop generation (other than EOS token). Please note that this is exclusivelly available to our PyTorch implementations.

Abstract base class for all stopping criteria that can be applied during generation.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/stopping_criteria.py#L39)

( input\_ids: LongTensorscores: FloatTensor\*\*kwargs )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax or scores for each vocabulary token after SoftMax.
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional stopping criteria specific kwargs.

### class transformers.StoppingCriteriaList

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/stopping_criteria.py#L124)

( iterable = () )

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/stopping_criteria.py#L125)

( input\_ids: LongTensorscores: FloatTensor\*\*kwargs )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax or scores for each vocabulary token after SoftMax.
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional stopping criteria specific kwargs.

### class transformers.MaxLengthCriteria

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/stopping_criteria.py#L44)

( max\_length: intmax\_position\_embeddings: typing.Optional\[int\] = None )

Parameters

-   **max\_length** (`int`) — The maximum length that the output sequence can have in number of tokens.
-   **max\_position\_embeddings** (`int`, `optional`) — The maximum model length, as defined by the model’s `config.max_position_embeddings` attribute.

This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep in mind for decoder-only type of transformers, this will include the initial prompted tokens.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/stopping_criteria.py#L60)

( input\_ids: LongTensorscores: FloatTensor\*\*kwargs )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax or scores for each vocabulary token after SoftMax.
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional stopping criteria specific kwargs.

### class transformers.MaxTimeCriteria

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/stopping_criteria.py#L102)

( max\_time: floatinitial\_timestamp: typing.Optional\[float\] = None )

Parameters

-   **max\_time** (`float`) — The maximum allowed time in seconds for the generation.
-   **initial\_time** (`float`, _optional_, defaults to `time.time()`) — The start of the generation allowed time.

This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the time will start being counted when you initialize this function. You can override this by passing an `initial_time`.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/stopping_criteria.py#L119)

( input\_ids: LongTensorscores: FloatTensor\*\*kwargs )

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **scores** (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`) — Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax or scores for each vocabulary token after SoftMax.
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional stopping criteria specific kwargs.

## Constraints

A [Constraint](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.Constraint) can be used to force the generation to include specific tokens or sequences in the output. Please note that this is exclusivelly available to our PyTorch implementations.

Abstract base class for all constraints that can be applied during generation. It must define how the constraint can be satisfied.

All classes that inherit Constraint must follow the requirement that

```
completed = False
while not completed:
    _, completed = constraint.update(constraint.advance())
```

will always terminate (halt).

#### advance

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_constraints.py#L48)

( ) → token\_ids(`torch.tensor`)

Returns

token\_ids(`torch.tensor`)

Must be a tensor of a list of indexable tokens, not some integer.

When called, returns the token that would take this constraint one step closer to being fulfilled.

#### copy

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_constraints.py#L113)

( stateful = False ) → constraint(`Constraint`)

Returns

constraint(`Constraint`)

The same constraint as the one being called from.

Creates a new instance of this constraint.

Reads in a token and returns whether it creates progress.

Returns the number of remaining steps of `advance()` in order to complete this constraint.

Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of a constraint is abrupted by an unwanted token.

Tests whether this constraint has been properly defined.

#### update

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_constraints.py#L69)

( token\_id: int ) → stepped(`bool`)

Whether this constraint has become one step closer to being fulfuilled. completed(`bool`): Whether this constraint has been completely fulfilled by this token being generated. reset (`bool`): Whether this constraint has reset its progress by this token being generated.

Reads in a token and returns booleans that indicate the progress made by it. This function will update the state of this object unlikes `does_advance(self, token_id: int)`.

This isn’t to test whether a certain token will advance the progress; it’s to update its state as if it has been generated. This becomes important if token\_id != desired token (refer to else statement in PhrasalConstraint)

### class transformers.PhrasalConstraint

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_constraints.py#L129)

( token\_ids: typing.List\[int\] )

Parameters

-   **token\_ids** (`List[int]`) — The id of the token that must be generated by the output.

[Constraint](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.Constraint) enforcing that an ordered sequence of tokens is included in the output.

### class transformers.DisjunctiveConstraint

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_constraints.py#L261)

( nested\_token\_ids: typing.List\[typing.List\[int\]\] )

Parameters

-   **nested\_token\_ids** (`List[List[int]]`) — a list of words, where each word is a list of ids. This constraint
-   **is** fulfilled by generating just one from the list of words. —

A special [Constraint](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.Constraint) that is fulfilled by fulfilling just one of several constraints.

### class transformers.ConstraintListState

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_constraints.py#L350)

( constraints: typing.List\[transformers.generation.beam\_constraints.Constraint\] )

Parameters

-   **constraints** (`List[Constraint]`) — A list of [Constraint](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.Constraint) objects that must be fulfilled by the beam scorer.

A class for beam scorers to track its progress through a list of constraints.

The list of tokens to generate such that we can make progress. By “list” we don’t mean the list of token that will fully fulfill a constraint.

Given constraints `c_i = {t_ij | j == # of tokens}`, If we’re not in the middle of progressing through a specific constraint `c_i`, we return:

`[t_k1 for k in indices of unfulfilled constraints]`

If we are in the middle of a constraint, then we return: `[t_ij]`, where `i` is the index of the inprogress constraint, `j` is the next step for the constraint.

Though we don’t care which constraint is fulfilled first, if we are in the progress of fulfilling a constraint, that’s the only one we’ll return.

#### reset

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_constraints.py#L417)

( token\_ids: typing.Optional\[typing.List\[int\]\] )

token\_ids: the tokens generated thus far to reset the state of the progress through constraints.

## BeamSearch

Abstract base class for all beam scorers that are used for [beam\_search()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search) and [beam\_sample()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample).

#### process

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_search.py#L97)

( input\_ids: LongTensornext\_scores: FloatTensornext\_tokens: LongTensornext\_indices: LongTensor\*\*kwargs ) → `UserDict`

#### finalize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_search.py#L109)

( input\_ids: LongTensornext\_scores: FloatTensornext\_tokens: LongTensornext\_indices: LongTensormax\_length: int\*\*kwargs ) → `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`

### class transformers.BeamSearchScorer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_search.py#L123)

( batch\_size: intnum\_beams: intdevice: devicelength\_penalty: typing.Optional\[float\] = 1.0do\_early\_stopping: typing.Union\[bool, str, NoneType\] = Falsenum\_beam\_hyps\_to\_keep: typing.Optional\[int\] = 1num\_beam\_groups: typing.Optional\[int\] = 1max\_length: typing.Optional\[int\] = None )

[BeamScorer](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.BeamScorer) implementing standard beam search decoding.

Adapted in part from [Facebook’s XLM beam search code](https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529).

Reference for the diverse beam search algorithm and implementation [Ashwin Kalyan’s DBS implementation](https://github.com/ashwinkalyan/dbs/blob/master/dbs/beam_utils.lua)

#### process

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_search.py#L215)

( input\_ids: LongTensornext\_scores: FloatTensornext\_tokens: LongTensornext\_indices: LongTensorpad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Nonebeam\_indices: typing.Optional\[torch.LongTensor\] = Nonegroup\_index: typing.Optional\[int\] = 0 )

#### finalize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_search.py#L315)

( input\_ids: LongTensorfinal\_beam\_scores: FloatTensorfinal\_beam\_tokens: LongTensorfinal\_beam\_indices: LongTensormax\_length: intpad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Nonebeam\_indices: typing.Optional\[torch.LongTensor\] = None )

### class transformers.ConstrainedBeamSearchScorer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_search.py#L410)

( batch\_size: intnum\_beams: intconstraints: typing.List\[transformers.generation.beam\_constraints.Constraint\]device: devicelength\_penalty: typing.Optional\[float\] = 1.0do\_early\_stopping: typing.Union\[bool, str, NoneType\] = Falsenum\_beam\_hyps\_to\_keep: typing.Optional\[int\] = 1num\_beam\_groups: typing.Optional\[int\] = 1max\_length: typing.Optional\[int\] = None )

[BeamScorer](/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.BeamScorer) implementing constrained beam search decoding.

#### process

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_search.py#L504)

( input\_ids: LongTensornext\_scores: FloatTensornext\_tokens: LongTensornext\_indices: LongTensorscores\_for\_all\_vocab: FloatTensorpad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Nonebeam\_indices: typing.Optional\[torch.LongTensor\] = None ) → `UserDict`

#### finalize

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/beam_search.py#L798)

( input\_ids: LongTensorfinal\_beam\_scores: FloatTensorfinal\_beam\_tokens: LongTensorfinal\_beam\_indices: LongTensormax\_length: intpad\_token\_id: typing.Optional\[int\] = Noneeos\_token\_id: typing.Union\[int, typing.List\[int\], NoneType\] = Nonebeam\_indices: typing.Optional\[torch.LongTensor\] = None )

## Utilities

#### transformers.top\_k\_top\_p\_filtering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/utils.py#L4706)

( logits: FloatTensortop\_k: int = 0top\_p: float = 1.0filter\_value: float = -infmin\_tokens\_to\_keep: int = 1 )

Parameters

-   **top\_k** (`int`, _optional_, defaults to 0) — If > 0, only keep the top k tokens with highest probability (top-k filtering)
-   **top\_p** (`float`, _optional_, defaults to 1.0) — If < 1.0, only keep the top tokens with cumulative probability >= top\_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. ([http://arxiv.org/abs/1904.09751](http://arxiv.org/abs/1904.09751))
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimumber of tokens we keep per batch example in the output.

Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

From: [https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317](https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317)

#### transformers.tf\_top\_k\_top\_p\_filtering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/tf_utils.py#L3063)

( logitstop\_k = 0top\_p = 1.0filter\_value = -infmin\_tokens\_to\_keep = 1 )

Parameters

-   **top\_k** (`int`, _optional_, defaults to 0) — If > 0, only keep the top k tokens with highest probability (top-k filtering)
-   **top\_p** (`float`, _optional_, defaults to 1.0) — If < 1.0, only keep the top tokens with cumulative probability >= top\_p (nucleus filtering). Nucleus filtering is described in Holtzman et al. ([http://arxiv.org/abs/1904.09751](http://arxiv.org/abs/1904.09751))
-   **min\_tokens\_to\_keep** (`int`, _optional_, defaults to 1) — Minimumber of tokens we keep per batch example in the output.

Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

From: [https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317](https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317)

## Streamers

### class transformers.TextStreamer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/streamers.py#L38)

( tokenizer: AutoTokenizerskip\_prompt: bool = False\*\*decode\_kwargs )

Parameters

-   **tokenizer** (`AutoTokenizer`) — The tokenized used to decode the tokens.
-   **skip\_prompt** (`bool`, _optional_, defaults to `False`) — Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
-   **decode\_kwargs** (`dict`, _optional_) — Additional keyword arguments to pass to the tokenizer’s `decode` method.

Simple text streamer that prints the token(s) to stdout as soon as entire words are formed.

The API for the streamer classes is still under development and may change in the future.

Examples:

```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

>>> tok = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextStreamer(tok)

>>> 
>>> _ = model.generate(**inputs, streamer=streamer, max_new_tokens=20)
An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,
```

Flushes any remaining cache and prints a newline to stdout.

#### on\_finalized\_text

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/streamers.py#L130)

( text: strstream\_end: bool = False )

Prints the new text to stdout. If the stream is ending, also prints a newline.

Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.

### class transformers.TextIteratorStreamer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/streamers.py#L159)

( tokenizer: AutoTokenizerskip\_prompt: bool = Falsetimeout: typing.Optional\[float\] = None\*\*decode\_kwargs )

Parameters

-   **tokenizer** (`AutoTokenizer`) — The tokenized used to decode the tokens.
-   **skip\_prompt** (`bool`, _optional_, defaults to `False`) — Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
-   **timeout** (`float`, _optional_) — The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions in `.generate()`, when it is called in a separate thread.
-   **decode\_kwargs** (`dict`, _optional_) — Additional keyword arguments to pass to the tokenizer’s `decode` method.

Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is useful for applications that benefit from acessing the generated text in a non-blocking way (e.g. in an interactive Gradio demo).

The API for the streamer classes is still under development and may change in the future.

Examples:

```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
>>> from threading import Thread

>>> tok = AutoTokenizer.from_pretrained("gpt2")
>>> model = AutoModelForCausalLM.from_pretrained("gpt2")
>>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
>>> streamer = TextIteratorStreamer(tok)

>>> 
>>> generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
>>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
>>> thread.start()
>>> generated_text = ""
>>> for new_text in streamer:
...     generated_text += new_text
>>> generated_text
'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
```

#### on\_finalized\_text

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/generation/streamers.py#L213)

( text: strstream\_end: bool = False )

Put the new text in the queue. If the stream is ending, also put a stop signal in the queue.