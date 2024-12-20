# Perceiver

## Overview

The Perceiver IO model was proposed in [Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795) by Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, João Carreira.

Perceiver IO is a generalization of [Perceiver](https://arxiv.org/abs/2103.03206) to handle arbitrary outputs in addition to arbitrary inputs. The original Perceiver only produced a single classification label. In addition to classification labels, Perceiver IO can produce (for example) language, optical flow, and multimodal videos with audio. This is done using the same building blocks as the original Perceiver. The computational complexity of Perceiver IO is linear in the input and output size and the bulk of the processing occurs in the latent space, allowing us to process inputs and outputs that are much larger than can be handled by standard Transformers. This means, for example, Perceiver IO can do BERT-style masked language modeling directly using bytes instead of tokenized inputs.

The abstract from the paper is the following:

_The recently-proposed Perceiver model obtains good results on several domains (images, audio, multimodal, point clouds) while scaling linearly in compute and memory with the input size. While the Perceiver supports many kinds of inputs, it can only produce very simple outputs such as class scores. Perceiver IO overcomes this limitation without sacrificing the original’s appealing properties by learning to flexibly query the model’s latent space to produce outputs of arbitrary size and semantics. Perceiver IO still decouples model depth from data size and still scales linearly with data size, but now with respect to both input and output sizes. The full Perceiver IO model achieves strong results on tasks with highly structured output spaces, such as natural language and visual understanding, StarCraft II, and multi-task and multi-modal domains. As highlights, Perceiver IO matches a Transformer-based BERT baseline on the GLUE language benchmark without the need for input tokenization and achieves state-of-the-art performance on Sintel optical flow estimation._

Here’s a TLDR explaining how Perceiver works:

The main problem with the self-attention mechanism of the Transformer is that the time and memory requirements scale quadratically with the sequence length. Hence, models like BERT and RoBERTa are limited to a max sequence length of 512 tokens. Perceiver aims to solve this issue by, instead of performing self-attention on the inputs, perform it on a set of latent variables, and only use the inputs for cross-attention. In this way, the time and memory requirements don’t depend on the length of the inputs anymore, as one uses a fixed amount of latent variables, like 256 or 512. These are randomly initialized, after which they are trained end-to-end using backpropagation.

Internally, [PerceiverModel](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverModel) will create the latents, which is a tensor of shape `(batch_size, num_latents, d_latents)`. One must provide `inputs` (which could be text, images, audio, you name it!) to the model, which it will use to perform cross-attention with the latents. The output of the Perceiver encoder is a tensor of the same shape. One can then, similar to BERT, convert the last hidden states of the latents to classification logits by averaging along the sequence dimension, and placing a linear layer on top of that to project the `d_latents` to `num_labels`.

This was the idea of the original Perceiver paper. However, it could only output classification logits. In a follow-up work, PerceiverIO, they generalized it to let the model also produce outputs of arbitrary size. How, you might ask? The idea is actually relatively simple: one defines outputs of an arbitrary size, and then applies cross-attention with the last hidden states of the latents, using the outputs as queries, and the latents as keys and values.

So let’s say one wants to perform masked language modeling (BERT-style) with the Perceiver. As the Perceiver’s input length will not have an impact on the computation time of the self-attention layers, one can provide raw bytes, providing `inputs` of length 2048 to the model. If one now masks out certain of these 2048 tokens, one can define the `outputs` as being of shape: `(batch_size, 2048, 768)`. Next, one performs cross-attention with the final hidden states of the latents to update the `outputs` tensor. After cross-attention, one still has a tensor of shape `(batch_size, 2048, 768)`. One can then place a regular language modeling head on top, to project the last dimension to the vocabulary size of the model, i.e. creating logits of shape `(batch_size, 2048, 262)` (as Perceiver uses a vocabulary size of 262 byte IDs).

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perceiver_architecture.jpg) Perceiver IO architecture. Taken from the [original paper](https://arxiv.org/abs/2105.15203)

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/deepmind/deepmind-research/tree/master/perceiver).

Tips:

-   The quickest way to get started with the Perceiver is by checking the [tutorial notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Perceiver).
-   Refer to the [blog post](https://huggingface.co/blog/perceiver) if you want to fully understand how the model works and is implemented in the library. Note that the models available in the library only showcase some examples of what you can do with the Perceiver. There are many more use cases, including question answering, named-entity recognition, object detection, audio classification, video classification, etc.

**Note**:

-   Perceiver does **not** work with `torch.nn.DataParallel` due to a bug in PyTorch, see [issue #36035](https://github.com/pytorch/pytorch/issues/36035)

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)
-   [Image classification task guide](../tasks/image_classification)

## Perceiver specific outputs

### class transformers.models.perceiver.modeling\_perceiver.PerceiverModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L61)

( logits: FloatTensor = None last\_hidden\_state: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None cross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Base class for Perceiver base model’s outputs, with potential hidden states, attentions and cross-attentions.

### class transformers.models.perceiver.modeling\_perceiver.PerceiverDecoderOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L92)

( logits: FloatTensor = None cross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_labels)`) — Output of the basic decoder.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Base class for Perceiver decoder outputs, with potential cross-attentions.

### class transformers.models.perceiver.modeling\_perceiver.PerceiverMaskedLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L110)

( loss: typing.Optional\[torch.FloatTensor\] = None logits: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None cross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Masked language modeling (MLM) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_latents, num_latents)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Base class for Perceiver’s masked language model outputs.

### class transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L141)

( loss: typing.Optional\[torch.FloatTensor\] = None logits: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None cross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

Base class for Perceiver’s outputs of sequence/image classification models, optical flow and multimodal autoencoding.

## PerceiverConfig

### class transformers.PerceiverConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/configuration_perceiver.py#L36)

( num\_latents = 256 d\_latents = 1280 d\_model = 768 num\_blocks = 1 num\_self\_attends\_per\_block = 26 num\_self\_attention\_heads = 8 num\_cross\_attention\_heads = 8 qk\_channels = None v\_channels = None cross\_attention\_shape\_for\_attention = 'kv' self\_attention\_widening\_factor = 1 cross\_attention\_widening\_factor = 1 hidden\_act = 'gelu' attention\_probs\_dropout\_prob = 0.1 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 use\_query\_residual = True vocab\_size = 262 max\_position\_embeddings = 2048 image\_size = 56 train\_size = \[368, 496\] num\_frames = 16 audio\_samples\_per\_frame = 1920 samples\_per\_patch = 16 output\_shape = \[1, 16, 224, 224\] output\_num\_channels = 512 \_label\_trainable\_num\_channels = 1024 \*\*kwargs )

Parameters

-   **num\_latents** (`int`, _optional_, defaults to 256) — The number of latents.
-   **d\_latents** (`int`, _optional_, defaults to 1280) — Dimension of the latent embeddings.
-   **d\_model** (`int`, _optional_, defaults to 768) — Dimension of the inputs. Should only be provided in case \[_PerceiverTextPreprocessor_\] is used or no preprocessor is provided.
-   **num\_blocks** (`int`, _optional_, defaults to 1) — Number of blocks in the Transformer encoder.
-   **num\_self\_attends\_per\_block** (`int`, _optional_, defaults to 26) — The number of self-attention layers per block.
-   **num\_self\_attention\_heads** (`int`, _optional_, defaults to 8) — Number of attention heads for each self-attention layer in the Transformer encoder.
-   **num\_cross\_attention\_heads** (`int`, _optional_, defaults to 8) — Number of attention heads for each cross-attention layer in the Transformer encoder.
-   **qk\_channels** (`int`, _optional_) — Dimension to project the queries + keys before applying attention in the cross-attention and self-attention layers of the encoder. Will default to preserving the dimension of the queries if not specified.
-   **v\_channels** (`int`, _optional_) — Dimension to project the values before applying attention in the cross-attention and self-attention layers of the encoder. Will default to preserving the dimension of the queries if not specified.
-   **cross\_attention\_shape\_for\_attention** (`str`, _optional_, defaults to `'kv'`) — Dimension to use when downsampling the queries and keys in the cross-attention layer of the encoder.
-   **self\_attention\_widening\_factor** (`int`, _optional_, defaults to 1) — Dimension of the feed-forward layer in the cross-attention layer of the Transformer encoder.
-   **cross\_attention\_widening\_factor** (`int`, _optional_, defaults to 1) — Dimension of the feed-forward layer in the self-attention layers of the Transformer encoder.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **use\_query\_residual** (`float`, _optional_, defaults to `True`) — Whether to add a query residual in the cross-attention layer of the encoder.
-   **vocab\_size** (`int`, _optional_, defaults to 262) — Vocabulary size for the masked language modeling model.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 2048) — The maximum sequence length that the masked language modeling model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **image\_size** (`int`, _optional_, defaults to 56) — Size of the images after preprocessing, for [PerceiverForImageClassificationLearned](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned).
-   **train\_size** (`List[int]`, _optional_, defaults to \[368, 496\]) — Training size of the images for the optical flow model.
-   **num\_frames** (`int`, _optional_, defaults to 16) — Number of video frames used for the multimodal autoencoding model.
-   **audio\_samples\_per\_frame** (`int`, _optional_, defaults to 1920) — Number of audio samples per frame for the multimodal autoencoding model.
-   **samples\_per\_patch** (`int`, _optional_, defaults to 16) — Number of audio samples per patch when preprocessing the audio for the multimodal autoencoding model.
-   **output\_num\_channels** (`int`, _optional_, defaults to 512) — Number of output channels for each modalitiy decoder.
-   **output\_shape** (`List[int]`, _optional_, defaults to `[1, 16, 224, 224]`) — Shape of the output (batch\_size, num\_frames, height, width) for the video decoder queries of the multimodal autoencoding model. This excludes the channel dimension.

This is the configuration class to store the configuration of a [PerceiverModel](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverModel). It is used to instantiate an Perceiver model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Perceiver [deepmind/language-perceiver](https://huggingface.co/deepmind/language-perceiver) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import PerceiverModel, PerceiverConfig

>>> 
>>> configuration = PerceiverConfig()

>>> 
>>> model = PerceiverModel(configuration)

>>> 
>>> configuration = model.config
```

## PerceiverTokenizer

### class transformers.PerceiverTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/tokenization_perceiver.py#L27)

( pad\_token = '\[PAD\]' bos\_token = '\[BOS\]' eos\_token = '\[EOS\]' mask\_token = '\[MASK\]' cls\_token = '\[CLS\]' sep\_token = '\[SEP\]' model\_max\_length = 2048 \*\*kwargs )

Parameters

-   **pad\_token** (`str`, _optional_, defaults to `"[PAD]"`) — The token used for padding, for example when batching sequences of different lengths.
-   **bos\_token** (`str`, _optional_, defaults to `"[BOS]"`) — The BOS token (reserved in the vocab, but not actually used).
-   **eos\_token** (`str`, _optional_, defaults to `"[EOS]"`) — The end of sequence token (reserved in the vocab, but not actually used).
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **mask\_token** (`str`, _optional_, defaults to `"[MASK]"`) — The MASK token, useful for masked language modeling.
-   **cls\_token** (`str`, _optional_, defaults to `"[CLS]"`) — The CLS token (reserved in the vocab, but not actually used).
-   **sep\_token** (`str`, _optional_, defaults to `"[SEP]"`) — The separator token, which is used when building a sequence from two sequences.

Construct a Perceiver tokenizer. The Perceiver simply uses raw bytes utf-8 encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L2732)

( text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = None text\_pair: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = None text\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = None text\_pair\_target: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = None add\_special\_tokens: bool = True padding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = False truncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = None max\_length: typing.Optional\[int\] = None stride: int = 0 is\_split\_into\_words: bool = False pad\_to\_multiple\_of: typing.Optional\[int\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None return\_token\_type\_ids: typing.Optional\[bool\] = None return\_attention\_mask: typing.Optional\[bool\] = None return\_overflowing\_tokens: bool = False return\_special\_tokens\_mask: bool = False return\_offsets\_mapping: bool = False return\_length: bool = False verbose: bool = True \*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Parameters

-   **text** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **text\_pair** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **text\_target** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **text\_pair\_target** (`str`, `List[str]`, `List[List[str]]`, _optional_) — The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
-   **add\_special\_tokens** (`bool`, _optional_, defaults to `True`) — Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is usefull if you want to add `bos` or `eos` tokens automatically.
-   **padding** (`bool`, `str` or [PaddingStrategy](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.utils.PaddingStrategy), _optional_, defaults to `False`) — Activates and controls padding. Accepts the following values:
    
    -   `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    -   `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    -   `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    
-   **truncation** (`bool`, `str` or [TruncationStrategy](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), _optional_, defaults to `False`) — Activates and controls truncation. Accepts the following values:
    
    -   `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
    -   `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).
    
-   **max\_length** (`int`, _optional_) — Controls the maximum length to use by one of the truncation/padding parameters.
    
    If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.
    
-   **stride** (`int`, _optional_, defaults to 0) — If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.
-   **is\_split\_into\_words** (`bool`, _optional_, defaults to `False`) — Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.
-   **pad\_to\_multiple\_of** (`int`, _optional_) — If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).
-   **return\_tensors** (`str` or [TensorType](/docs/transformers/v4.34.0/en/internal/file_utils#transformers.TensorType), _optional_) — If set, will return tensors instead of list of python integers. Acceptable values are:
    
    -   `'tf'`: Return TensorFlow `tf.constant` objects.
    -   `'pt'`: Return PyTorch `torch.Tensor` objects.
    -   `'np'`: Return Numpy `np.ndarray` objects.
    
-   **return\_token\_type\_ids** (`bool`, _optional_) — Whether to return token type IDs. If left to the default, will return the token type IDs according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **return\_attention\_mask** (`bool`, _optional_) — Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer’s default, defined by the `return_outputs` attribute.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **return\_overflowing\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead of returning overflowing tokens.
-   **return\_special\_tokens\_mask** (`bool`, _optional_, defaults to `False`) — Whether or not to return special tokens mask information.
-   **return\_offsets\_mapping** (`bool`, _optional_, defaults to `False`) — Whether or not to return `(char_start, char_end)` for each token.
    
    This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast), if using Python’s tokenizer, this method will raise `NotImplementedError`.
    
-   **return\_length** (`bool`, _optional_, defaults to `False`) — Whether or not to return the lengths of the encoded inputs.
-   **verbose** (`bool`, _optional_, defaults to `True`) — Whether or not to print more information and warnings. \*\*kwargs — passed to the `self.tokenize()` method

A [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

-   **input\_ids** — List of token ids to be fed to a model.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **token\_type\_ids** — List of token type ids to be fed to a model (when `return_token_type_ids=True` or if _“token\_type\_ids”_ is in `self.model_input_names`).
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **attention\_mask** — List of indices specifying which tokens should be attended to by the model (when `return_attention_mask=True` or if _“attention\_mask”_ is in `self.model_input_names`).
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **overflowing\_tokens** — List of overflowing tokens sequences (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **num\_truncated\_tokens** — Number of tokens truncated (when a `max_length` is specified and `return_overflowing_tokens=True`).
    
-   **special\_tokens\_mask** — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
    
-   **length** — The length of the inputs (when `return_length=True`)
    

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences.

## PerceiverFeatureExtractor

Preprocess an image or a batch of images.

## PerceiverImageProcessor

### class transformers.PerceiverImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/image_processing_perceiver.py#L46)

( do\_center\_crop: bool = True crop\_size: typing.Dict\[str, int\] = None do\_resize: bool = True size: typing.Dict\[str, int\] = None resample: Resampling = <Resampling.BICUBIC: 3> do\_rescale: bool = True rescale\_factor: typing.Union\[int, float\] = 0.00392156862745098 do\_normalize: bool = True image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None \*\*kwargs )

Parameters

-   **do\_center\_crop** (`bool`, `optional`, defaults to `True`) — Whether or not to center crop the image. If the input size if smaller than `crop_size` along any edge, the image will be padded with zeros and then center cropped. Can be overridden by the `do_center_crop` parameter in the `preprocess` method.
-   **crop\_size** (`Dict[str, int]`, _optional_, defaults to `{"height" -- 256, "width": 256}`): Desired output size when applying center-cropping. Can be overridden by the `crop_size` parameter in the `preprocess` method.
-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image to `(size["height"], size["width"])`. Can be overridden by the `do_resize` parameter in the `preprocess` method.
-   **size** (`Dict[str, int]` _optional_, defaults to `{"height" -- 224, "width": 224}`): Size of the image after resizing. Can be overridden by the `size` parameter in the `preprocess` method.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BICUBIC`) — Defines the resampling filter to use if resizing the image. Can be overridden by the `resample` parameter in the `preprocess` method.
-   **do\_rescale** (`bool`, _optional_, defaults to `True`) — Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the `do_rescale` parameter in the `preprocess` method.
-   **rescale\_factor** (`int` or `float`, _optional_, defaults to `1/255`) — Defines the scale factor to use if rescaling the image. Can be overridden by the `rescale_factor` parameter in the `preprocess` method. do\_normalize — Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_MEAN`) — Mean to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `IMAGENET_STANDARD_STD`) — Standard deviation to use if normalizing the image. This is a float or list of floats the length of the number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.

Constructs a Perceiver image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/image_processing_perceiver.py#L209)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\] do\_center\_crop: typing.Optional\[bool\] = None crop\_size: typing.Union\[typing.Dict\[str, int\], NoneType\] = None do\_resize: typing.Optional\[bool\] = None size: typing.Union\[typing.Dict\[str, int\], NoneType\] = None resample: Resampling = None do\_rescale: typing.Optional\[bool\] = None rescale\_factor: typing.Optional\[float\] = None do\_normalize: typing.Optional\[bool\] = None image\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = None image\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None return\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None data\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'> input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None \*\*kwargs )

Parameters

-   **images** (`ImageInput`) — Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.
-   **do\_center\_crop** (`bool`, _optional_, defaults to `self.do_center_crop`) — Whether to center crop the image to `crop_size`.
-   **crop\_size** (`Dict[str, int]`, _optional_, defaults to `self.crop_size`) — Desired output size after applying the center crop.
-   **do\_resize** (`bool`, _optional_, defaults to `self.do_resize`) — Whether to resize the image.
-   **size** (`Dict[str, int]`, _optional_, defaults to `self.size`) — Size of the image after resizing.
-   **resample** (`int`, _optional_, defaults to `self.resample`) — Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`, Only has an effect if `do_resize` is set to `True`.
-   **do\_rescale** (`bool`, _optional_, defaults to `self.do_rescale`) — Whether to rescale the image.
-   **rescale\_factor** (`float`, _optional_, defaults to `self.rescale_factor`) — Rescale factor to rescale the image by if `do_rescale` is set to `True`.
-   **do\_normalize** (`bool`, _optional_, defaults to `self.do_normalize`) — Whether to normalize the image.
-   **image\_mean** (`float` or `List[float]`, _optional_, defaults to `self.image_mean`) — Image mean.
-   **image\_std** (`float` or `List[float]`, _optional_, defaults to `self.image_std`) — Image standard deviation.
-   **return\_tensors** (`str` or `TensorType`, _optional_) — The type of tensors to return. Can be one of:
    
    -   Unset: Return a list of `np.ndarray`.
    -   `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
    -   `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
    -   `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
    -   `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
    
-   **data\_format** (`ChannelDimension` or `str`, _optional_, defaults to `ChannelDimension.FIRST`) — The channel dimension format for the output image. Can be one of:
    
    -   `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    
-   **input\_data\_format** (`ChannelDimension` or `str`, _optional_) — The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of:
    
    -   `"channels_first"` or `ChannelDimension.FIRST`: image in (num\_channels, height, width) format.
    -   `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num\_channels) format.
    -   `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
    

Preprocess an image or batch of images.

## PerceiverTextPreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverTextPreprocessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2846)

( config: PerceiverConfig )

Text preprocessing for Perceiver Encoder. Can be used to embed `inputs` and add positional encodings.

The dimensionality of the embeddings is determined by the `d_model` attribute of the configuration.

## PerceiverImagePreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverImagePreprocessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L3003)

( config prep\_type = 'conv' spatial\_downsample: int = 4 temporal\_downsample: int = 1 position\_encoding\_type: str = 'fourier' in\_channels: int = 3 out\_channels: int = 64 conv\_after\_patching: bool = False conv\_after\_patching\_in\_channels: int = 54 conv2d\_use\_batchnorm: bool = True concat\_or\_add\_pos: str = 'concat' project\_pos\_dim: int = -1 \*\*position\_encoding\_kwargs )

Parameters

-   **config** (\[_PerceiverConfig_\]) — Model configuration.
-   **prep\_type** (`str`, _optional_, defaults to `"conv"`) — Preprocessing type. Can be “conv1x1”, “conv”, “patches”, “pixels”.
-   **spatial\_downsample** (`int`, _optional_, defaults to 4) — Spatial downsampling factor.
-   **temporal\_downsample** (`int`, _optional_, defaults to 1) — Temporal downsampling factor (only relevant in case a time dimension is present).
-   **position\_encoding\_type** (`str`, _optional_, defaults to `"fourier"`) — Position encoding type. Can be “fourier” or “trainable”.
-   **in\_channels** (`int`, _optional_, defaults to 3) — Number of channels in the input.
-   **out\_channels** (`int`, _optional_, defaults to 64) — Number of channels in the output.
-   **conv\_after\_patching** (`bool`, _optional_, defaults to `False`) — Whether to apply a convolutional layer after patching.
-   **conv\_after\_patching\_in\_channels** (`int`, _optional_, defaults to 54) — Number of channels in the input of the convolutional layer after patching.
-   **conv2d\_use\_batchnorm** (`bool`, _optional_, defaults to `True`) — Whether to use batch normalization in the convolutional layer.
-   **concat\_or\_add\_pos** (`str`, _optional_, defaults to `"concat"`) — How to concatenate the position encoding to the input. Can be “concat” or “add”.
-   **project\_pos\_dim** (`int`, _optional_, defaults to -1) — Dimension of the position encoding to project to. If -1, no projection is applied.
-   \***\*position\_encoding\_kwargs** (`Dict`, _optional_) — Keyword arguments for the position encoding.

Image preprocessing for Perceiver Encoder.

Note: the _out\_channels_ argument refers to the output channels of a convolutional layer, if _prep\_type_ is set to “conv1x1” or “conv”. If one adds absolute position embeddings, one must make sure the _num\_channels_ of the position encoding kwargs are set equal to the _out\_channels_.

## PerceiverOneHotPreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverOneHotPreprocessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L3232)

( config: PerceiverConfig )

One-hot preprocessor for Perceiver Encoder. Can be used to add a dummy index dimension to the input.

## PerceiverAudioPreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverAudioPreprocessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L3258)

( config prep\_type: str = 'patches' samples\_per\_patch: int = 96 position\_encoding\_type: str = 'fourier' concat\_or\_add\_pos: str = 'concat' out\_channels = 64 project\_pos\_dim = -1 \*\*position\_encoding\_kwargs )

Parameters

-   **config** (\[_PerceiverConfig_\]) — Model configuration.
-   **prep\_type** (`str`, _optional_, defaults to `"patches"`) — Preprocessor type to use. Only “patches” is supported.
-   **samples\_per\_patch** (`int`, _optional_, defaults to 96) — Number of samples per patch.
-   **position\_encoding\_type** (`str`, _optional_, defaults to `"fourier"`) — Type of position encoding to use. Can be “trainable” or “fourier”.
-   **concat\_or\_add\_pos** (`str`, _optional_, defaults to `"concat"`) — How to concatenate the position encoding to the input. Can be “concat” or “add”.
-   **out\_channels** (`int`, _optional_, defaults to 64) — Number of channels in the output.
-   **project\_pos\_dim** (`int`, _optional_, defaults to -1) — Dimension of the position encoding to project to. If -1, no projection is applied.
-   \***\*position\_encoding\_kwargs** (`Dict`, _optional_) — Keyword arguments for the position encoding.

Audio preprocessing for Perceiver Encoder.

## PerceiverMultimodalPreprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverMultimodalPreprocessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L3355)

( modalities: typing.Mapping\[str, typing.Callable\[..., typing.Tuple\[torch.Tensor, typing.Optional\[torch.Tensor\], torch.Tensor\]\]\] mask\_probs: typing.Union\[typing.Mapping\[str, float\], NoneType\] = None min\_padding\_size: int = 2 )

Parameters

-   **modalities** (`Mapping[str, PreprocessorType]`) — Dict mapping modality name to preprocessor.
-   **mask\_probs** (`Dict[str, float]`) — Dict mapping modality name to masking probability of that modality.
-   **min\_padding\_size** (`int`, _optional_, defaults to 2) — The minimum padding size for all modalities. The final output will have num\_channels equal to the maximum channels across all modalities plus min\_padding\_size.

Multimodal preprocessing for Perceiver Encoder.

Inputs for each modality are preprocessed, then padded with trainable position embeddings to have the same number of channels.

## PerceiverProjectionDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverProjectionDecoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2051)

( config )

Baseline projection decoder (no cross-attention).

## PerceiverBasicDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverBasicDecoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2077)

( config: PerceiverConfig output\_num\_channels: int position\_encoding\_type: typing.Optional\[str\] = 'trainable' output\_index\_dims: typing.Optional\[int\] = None num\_channels: typing.Optional\[int\] = 128 subsampled\_index\_dims: typing.Optional\[int\] = None qk\_channels: typing.Optional\[int\] = None v\_channels: typing.Optional\[int\] = None num\_heads: typing.Optional\[int\] = 1 widening\_factor: typing.Optional\[int\] = 1 use\_query\_residual: typing.Optional\[bool\] = False concat\_preprocessed\_input: typing.Optional\[bool\] = False final\_project: typing.Optional\[bool\] = True position\_encoding\_only: typing.Optional\[bool\] = False \*\*position\_encoding\_kwargs )

Parameters

-   **config** (\[_PerceiverConfig_\]) — Model configuration.
-   **output\_num\_channels** (`int`, _optional_) — The number of channels in the output. Will only be used in case _final\_project_ is set to `True`.
-   **position\_encoding\_type** (`str`, _optional_, defaults to “trainable”) — The type of position encoding to use. Can be either “trainable”, “fourier”, or “none”.
-   **output\_index\_dims** (`int`, _optional_) — The number of dimensions of the output queries. Ignored if ‘position\_encoding\_type’ == ‘none’.
-   **num\_channels** (`int`, _optional_, defaults to 128) — The number of channels of the decoder queries. Ignored if ‘position\_encoding\_type’ == ‘none’.
-   **qk\_channels** (`int`, _optional_) — The number of channels of the queries and keys in the cross-attention layer.
-   **v\_channels** (`int`, _optional_) — The number of channels of the values in the cross-attention layer.
-   **num\_heads** (`int`, _optional_, defaults to 1) — The number of attention heads in the cross-attention layer.
-   **widening\_factor** (`int`, _optional_, defaults to 1) — The widening factor of the cross-attention layer.
-   **use\_query\_residual** (`bool`, _optional_, defaults to `False`) — Whether to use a residual connection between the query and the output of the cross-attention layer.
-   **concat\_preprocessed\_input** (`bool`, _optional_, defaults to `False`) — Whether to concatenate the preprocessed input to the query.
-   **final\_project** (`bool`, _optional_, defaults to `True`) — Whether to project the output of the cross-attention layer to a target dimension.
-   **position\_encoding\_only** (`bool`, _optional_, defaults to `False`) — Whether to only use this class to define output queries.

Cross-attention-based decoder. This class can be used to decode the final hidden states of the latents using a cross-attention operation, in which the latents produce keys and values.

The shape of the output of this class depends on how one defines the output queries (also called decoder queries).

## PerceiverClassificationDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverClassificationDecoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2263)

( config \*\*decoder\_kwargs )

Cross-attention based classification decoder. Light-weight wrapper of `PerceiverBasicDecoder` for logit output. Will turn the output of the Perceiver encoder which is of shape (batch\_size, num\_latents, d\_latents) to a tensor of shape (batch\_size, num\_labels). The queries are of shape (batch\_size, 1, num\_labels).

## PerceiverOpticalFlowDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverOpticalFlowDecoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2309)

( config output\_image\_shape output\_num\_channels = 2 rescale\_factor = 100.0 \*\*decoder\_kwargs )

Cross-attention based optical flow decoder.

## PerceiverBasicVideoAutoencodingDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverBasicVideoAutoencodingDecoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2344)

( config: PerceiverConfig output\_shape: typing.List\[int\] position\_encoding\_type: str \*\*decoder\_kwargs )

Parameters

-   **config** (\[_PerceiverConfig_\]) — Model configuration.
-   **output\_shape** (`List[int]`) — Shape of the output as (batch\_size, num\_frames, height, width), excluding the channel dimension.
-   **position\_encoding\_type** (`str`) — The type of position encoding to use. Can be either “trainable”, “fourier”, or “none”.

Cross-attention based video-autoencoding decoder. Light-weight wrapper of \[_PerceiverBasicDecoder_\] with video reshaping logic.

## PerceiverMultimodalDecoder

### class transformers.models.perceiver.modeling\_perceiver.PerceiverMultimodalDecoder

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2421)

( config: PerceiverConfig modalities: typing.Dict\[str, transformers.models.perceiver.modeling\_perceiver.PerceiverAbstractDecoder\] num\_outputs: int output\_num\_channels: int min\_padding\_size: typing.Optional\[int\] = 2 subsampled\_index\_dims: typing.Union\[typing.Dict\[str, transformers.models.perceiver.modeling\_perceiver.PerceiverAbstractDecoder\], NoneType\] = None \*\*decoder\_kwargs )

Parameters

-   **config** (\[_PerceiverConfig_\]) — Model configuration.
-   **modalities** (`Dict[str, PerceiverAbstractDecoder]`) — Dictionary mapping modality name to the decoder of that modality.
-   **num\_outputs** (`int`) — The number of outputs of the decoder.
-   **output\_num\_channels** (`int`) — The number of channels in the output.
-   **min\_padding\_size** (`int`, _optional_, defaults to 2) — The minimum padding size for all modalities. The final output will have num\_channels equal to the maximum channels across all modalities plus min\_padding\_size.
-   **subsampled\_index\_dims** (`Dict[str, PerceiverAbstractDecoder]`, _optional_) — Dictionary mapping modality name to the subsampled index dimensions to use for the decoder query of that modality.

Multimodal decoding by composing uni-modal decoders. The _modalities_ argument of the constructor is a dictionary mapping modality name to the decoder of that modality. That decoder will be used to construct queries for that modality. Modality-specific queries are padded with trainable modality-specific parameters, after which they are concatenated along the time dimension.

Next, there is a shared cross attention operation across all modalities.

## PerceiverProjectionPostprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverProjectionPostprocessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2982)

( in\_channels: int out\_channels: int )

Parameters

-   **in\_channels** (`int`) — Number of channels in the input.
-   **out\_channels** (`int`) — Number of channels in the output.

Projection postprocessing for Perceiver. Can be used to project the channels of the decoder output to a lower dimension.

## PerceiverAudioPostprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverAudioPostprocessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2955)

( config: PerceiverConfig in\_channels: int postproc\_type: str = 'patches' )

Parameters

-   **config** (\[_PerceiverConfig_\]) — Model configuration.
-   **in\_channels** (`int`) — Number of channels in the input.
-   **postproc\_type** (`str`, _optional_, defaults to `"patches"`) — Postprocessor type to use. Currently, only “patches” is supported.

Audio postprocessing for Perceiver. Can be used to convert the decoder output to audio features.

## PerceiverClassificationPostprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverClassificationPostprocessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2935)

( config: PerceiverConfig in\_channels: int )

Parameters

-   **config** (\[_PerceiverConfig_\]) — Model configuration.
-   **in\_channels** (`int`) — Number of channels in the input.

Classification postprocessing for Perceiver. Can be used to convert the decoder output to classification logits.

## PerceiverMultimodalPostprocessor

### class transformers.models.perceiver.modeling\_perceiver.PerceiverMultimodalPostprocessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L2901)

( modalities: typing.Mapping\[str, typing.Callable\[..., typing.Any\]\] input\_is\_dict: bool = False )

Parameters

-   **modalities** (`Mapping[str, PostprocessorType]`) — Dictionary mapping modality name to postprocessor class for that modality.
-   **input\_is\_dict** (`bool`, _optional_, defaults to `False`) — If True, input is assumed to be dictionary structured, and outputs keep the same dictionary shape. If False, input is a tensor which is sliced up during postprocessing by _modality\_sizes_.

Multimodal postprocessing for Perceiver. Can be used to combine modality-specific postprocessors into a single postprocessor.

## PerceiverModel

### class transformers.PerceiverModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L716)

( config decoder = None input\_preprocessor: typing.Callable\[..., typing.Tuple\[torch.Tensor, typing.Optional\[torch.Tensor\], torch.Tensor\]\] = None output\_postprocessor: typing.Callable\[..., typing.Any\] = None )

Parameters

-   **config** ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.
-   **decoder** (_DecoderType_, _optional_) — Optional decoder to use to decode the latent representation of the encoder. Examples include _transformers.models.perceiver.modeling\_perceiver.PerceiverBasicDecoder_, _transformers.models.perceiver.modeling\_perceiver.PerceiverClassificationDecoder_, _transformers.models.perceiver.modeling\_perceiver.PerceiverMultimodalDecoder_.
-   **input\_preprocessor** (_PreprocessorType_, _optional_) — Optional input preprocessor to use. Examples include _transformers.models.perceiver.modeling\_perceiver.PerceiverImagePreprocessor_, _transformers.models.perceiver.modeling\_perceiver.PerceiverAudioPreprocessor_, _transformers.models.perceiver.modeling\_perceiver.PerceiverTextPreprocessor_, _transformers.models.perceiver.modeling\_perceiver.PerceiverMultimodalPreprocessor_.
-   **output\_postprocessor** (_PostprocessorType_, _optional_) — Optional output postprocessor to use. Examples include _transformers.models.perceiver.modeling\_perceiver.PerceiverImagePostprocessor_, _transformers.models.perceiver.modeling\_perceiver.PerceiverAudioPostprocessor_, _transformers.models.perceiver.modeling\_perceiver.PerceiverClassificationPostprocessor_, _transformers.models.perceiver.modeling\_perceiver.PerceiverProjectionPostprocessor_, _transformers.models.perceiver.modeling\_perceiver.PerceiverMultimodalPostprocessor_.
-   **Note** that you can define your own decoders, preprocessors and/or postprocessors to fit your use-case. —

The Perceiver: a scalable, fully attentional architecture. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L752)

( inputs: FloatTensor attention\_mask: typing.Optional\[torch.FloatTensor\] = None subsampled\_output\_points: typing.Union\[typing.Dict\[str, torch.Tensor\], NoneType\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverModelOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **inputs** (`torch.FloatTensor`) — Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.models.perceiver.modeling\_perceiver.PerceiverModelOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

The [PerceiverModel](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import PerceiverConfig, PerceiverTokenizer, PerceiverImageProcessor, PerceiverModel
>>> from transformers.models.perceiver.modeling_perceiver import (
...     PerceiverTextPreprocessor,
...     PerceiverImagePreprocessor,
...     PerceiverClassificationDecoder,
... )
>>> import torch
>>> import requests
>>> from PIL import Image

>>> 
>>> 
>>> 
>>> 
>>> 
>>> config = PerceiverConfig()
>>> preprocessor = PerceiverTextPreprocessor(config)
>>> decoder = PerceiverClassificationDecoder(
...     config,
...     num_channels=config.d_latents,
...     trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
...     use_query_residual=True,
... )
>>> model = PerceiverModel(config, input_preprocessor=preprocessor, decoder=decoder)

>>> 
>>> tokenizer = PerceiverTokenizer()
>>> text = "hello world"
>>> inputs = tokenizer(text, return_tensors="pt").input_ids

>>> with torch.no_grad():
...     outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2]

>>> 
>>> criterion = torch.nn.CrossEntropyLoss()

>>> labels = torch.tensor([1])
>>> loss = criterion(logits, labels)

>>> 
>>> 
>>> config = PerceiverConfig(image_size=224)
>>> preprocessor = PerceiverImagePreprocessor(
...     config,
...     prep_type="conv1x1",
...     spatial_downsample=1,
...     out_channels=256,
...     position_encoding_type="trainable",
...     concat_or_add_pos="concat",
...     project_pos_dim=256,
...     trainable_position_encoding_kwargs=dict(
...         num_channels=256,
...         index_dims=config.image_size**2,
...     ),
... )

>>> model = PerceiverModel(
...     config,
...     input_preprocessor=preprocessor,
...     decoder=PerceiverClassificationDecoder(
...         config,
...         num_channels=config.d_latents,
...         trainable_position_encoding_kwargs=dict(num_channels=config.d_latents, index_dims=1),
...         use_query_residual=True,
...     ),
... )

>>> 
>>> image_processor = PerceiverImageProcessor()
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> inputs = image_processor(image, return_tensors="pt").pixel_values

>>> with torch.no_grad():
...     outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2]

>>> 
>>> criterion = torch.nn.CrossEntropyLoss()

>>> labels = torch.tensor([1])
>>> loss = criterion(logits, labels)
```

## PerceiverForMaskedLM

### class transformers.PerceiverForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L954)

( config: PerceiverConfig )

Parameters

-   **config** ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for masked language modeling. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L986)

( inputs: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None input\_ids: typing.Optional\[torch.Tensor\] = None ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverMaskedLMOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **inputs** (`torch.FloatTensor`) — Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
-   **attention\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

A [transformers.models.perceiver.modeling\_perceiver.PerceiverMaskedLMOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMaskedLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Masked language modeling (MLM) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_latents, num_latents)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

The [PerceiverForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```perceiver
>>> from transformers import AutoTokenizer, PerceiverForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("deepmind/language-perceiver")
>>> model = PerceiverForMaskedLM.from_pretrained("deepmind/language-perceiver")

>>> 
>>> text = "This is an incomplete sentence where some words are missing."
>>> inputs = tokenizer(text, padding="max_length", return_tensors="pt")
>>> 
>>> inputs["input_ids"][0, 52:61] = tokenizer.mask_token_id
>>> labels = tokenizer(text, padding="max_length", return_tensors="pt").input_ids

>>> outputs = model(**inputs, labels=labels)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
19.87

>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2048, 262]

>>> 
>>> text = "This is an incomplete sentence where some words are missing."
>>> encoding = tokenizer(text, padding="max_length", return_tensors="pt")

>>> 
>>> encoding["input_ids"][0, 52:61] = tokenizer.mask_token_id

>>> 
>>> with torch.no_grad():
...     outputs = model(**encoding)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2048, 262]

>>> masked_tokens_predictions = logits[0, 52:61].argmax(dim=-1).tolist()
>>> tokenizer.decode(masked_tokens_predictions)
' missing.'
```

## PerceiverForSequenceClassification

### class transformers.PerceiverForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1089)

( config )

Parameters

-   **config** ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for text classification. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1110)

( inputs: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None input\_ids: typing.Optional\[torch.Tensor\] = None ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **inputs** (`torch.FloatTensor`) — Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
-   **attention\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

The [PerceiverForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```perceiver
>>> from transformers import AutoTokenizer, PerceiverForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("deepmind/language-perceiver")
>>> model = PerceiverForSequenceClassification.from_pretrained("deepmind/language-perceiver")

>>> text = "hello world"
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
>>> outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 2]
```

## PerceiverForImageClassificationLearned

### class transformers.PerceiverForImageClassificationLearned

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1214)

( config )

Parameters

-   **config** ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses learned position embeddings. In other words, this model is not given any privileged information about the structure of images. As shown in the paper, this model can achieve a top-1 accuracy of 72.7 on ImageNet.

[PerceiverForImageClassificationLearned](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) uses [PerceiverImagePreprocessor](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor) (with `prep_type="conv1x1"`) to preprocess the input images, and [PerceiverClassificationDecoder](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder) to decode the latent representation of [PerceiverModel](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverModel) into classification logits.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1245)

( inputs: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None pixel\_values: typing.Optional\[torch.Tensor\] = None ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **inputs** (`torch.FloatTensor`) — Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
-   **attention\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

The [PerceiverForImageClassificationLearned](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, PerceiverForImageClassificationLearned
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-learned")
>>> model = PerceiverForImageClassificationLearned.from_pretrained("deepmind/vision-perceiver-learned")

>>> inputs = image_processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 1000]

>>> 
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: tabby, tabby cat
```

## PerceiverForImageClassificationFourier

### class transformers.PerceiverForImageClassificationFourier

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1357)

( config )

Parameters

-   **config** ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses fixed 2D Fourier position embeddings. As shown in the paper, this model can achieve a top-1 accuracy of 79.0 on ImageNet, and 84.5 when pre-trained on a large-scale dataset (i.e. JFT).

[PerceiverForImageClassificationLearned](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) uses [PerceiverImagePreprocessor](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor) (with `prep_type="pixels"`) to preprocess the input images, and [PerceiverClassificationDecoder](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder) to decode the latent representation of [PerceiverModel](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverModel) into classification logits.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1389)

( inputs: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None pixel\_values: typing.Optional\[torch.Tensor\] = None ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **inputs** (`torch.FloatTensor`) — Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
-   **attention\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

The [PerceiverForImageClassificationFourier](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForImageClassificationFourier) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, PerceiverForImageClassificationFourier
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-fourier")
>>> model = PerceiverForImageClassificationFourier.from_pretrained("deepmind/vision-perceiver-fourier")

>>> inputs = image_processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 1000]

>>> 
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: tabby, tabby cat
```

## PerceiverForImageClassificationConvProcessing

### class transformers.PerceiverForImageClassificationConvProcessing

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1500)

( config )

Parameters

-   **config** ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for image classification, for tasks such as ImageNet.

This model uses a 2D conv+maxpool preprocessing network. As shown in the paper, this model can achieve a top-1 accuracy of 82.1 on ImageNet.

[PerceiverForImageClassificationLearned](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForImageClassificationLearned) uses [PerceiverImagePreprocessor](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor) (with `prep_type="conv"`) to preprocess the input images, and [PerceiverClassificationDecoder](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassificationDecoder) to decode the latent representation of [PerceiverModel](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverModel) into classification logits.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1533)

( inputs: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None pixel\_values: typing.Optional\[torch.Tensor\] = None ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **inputs** (`torch.FloatTensor`) — Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
-   **attention\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

The [PerceiverForImageClassificationConvProcessing](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForImageClassificationConvProcessing) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, PerceiverForImageClassificationConvProcessing
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("deepmind/vision-perceiver-conv")
>>> model = PerceiverForImageClassificationConvProcessing.from_pretrained("deepmind/vision-perceiver-conv")

>>> inputs = image_processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(inputs=inputs)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 1000]

>>> 
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: tabby, tabby cat
```

## PerceiverForOpticalFlow

### class transformers.PerceiverForOpticalFlow

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1644)

( config )

Parameters

-   **config** ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for optical flow, for tasks such as Sintel and KITTI. [PerceiverForOpticalFlow](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForOpticalFlow) uses [PerceiverImagePreprocessor](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverImagePreprocessor) (with _prep\_type=“patches”_) to preprocess the input images, and [PerceiverOpticalFlowDecoder](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverOpticalFlowDecoder) to decode the latent representation of [PerceiverModel](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverModel).

As input, one concatenates 2 subsequent frames along the channel dimension and extract a 3 x 3 patch around each pixel (leading to 3 x 3 x 3 x 2 = 54 values for each pixel). Fixed Fourier position encodings are used to encode the position of each pixel in the patch. Next, one applies the Perceiver encoder. To decode, one queries the latent representation using the same encoding used for the input.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1694)

( inputs: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **inputs** (`torch.FloatTensor`) — Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
-   **attention\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the optical flow loss. Indices should be in `[0, ..., config.num_labels - 1]`.

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

The [PerceiverForOpticalFlow](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForOpticalFlow) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import PerceiverForOpticalFlow
>>> import torch

>>> model = PerceiverForOpticalFlow.from_pretrained("deepmind/optical-flow-perceiver")

>>> 
>>> 
>>> 
>>> 
>>> patches = torch.randn(1, 2, 27, 368, 496)
>>> outputs = model(inputs=patches)
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 368, 496, 2]
```

## PerceiverForMultimodalAutoencoding

### class transformers.PerceiverForMultimodalAutoencoding

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1789)

( config: PerceiverConfig )

Parameters

-   **config** ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Example use of Perceiver for multimodal (video) autoencoding, for tasks such as Kinetics-700.

[PerceiverForMultimodalAutoencoding](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForMultimodalAutoencoding) uses [PerceiverMultimodalPreprocessor](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalPreprocessor) to preprocess the 3 modalities: images, audio and class labels. This preprocessor uses modality-specific preprocessors to preprocess every modality separately, after which they are concatenated. Trainable position embeddings are used to pad each modality to the same number of channels to make concatenation along the time dimension possible. Next, one applies the Perceiver encoder.

[PerceiverMultimodalDecoder](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder) is used to decode the latent representation of [PerceiverModel](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverModel). This decoder uses each modality-specific decoder to construct queries. The decoder queries are created based on the inputs after preprocessing. However, autoencoding an entire video in a single forward pass is computationally infeasible, hence one only uses parts of the decoder queries to do cross-attention with the latent representation. This is determined by the subsampled indices for each modality, which can be provided as additional input to the forward pass of [PerceiverForMultimodalAutoencoding](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForMultimodalAutoencoding).

[PerceiverMultimodalDecoder](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverMultimodalDecoder) also pads the decoder queries of the different modalities to the same number of channels, in order to concatenate them along the time dimension. Next, cross-attention is performed with the latent representation of [PerceiverModel](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverModel).

Finally, `~models.perceiver.modeling_perceiver.PerceiverMultiModalPostprocessor` is used to turn this tensor into an actual video. It first splits up the output into the different modalities, and then applies the respective postprocessor for each modality.

Note that, by masking the classification label during evaluation (i.e. simply providing a tensor of zeros for the “label” modality), this auto-encoding model becomes a Kinetics 700 video classifier.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/perceiver/modeling_perceiver.py#L1905)

( inputs: typing.Optional\[torch.Tensor\] = None attention\_mask: typing.Optional\[torch.Tensor\] = None subsampled\_output\_points: typing.Union\[typing.Dict\[str, torch.Tensor\], NoneType\] = None head\_mask: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None labels: typing.Optional\[torch.Tensor\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **inputs** (`torch.FloatTensor`) — Inputs to the perceiver. Can be anything: images, text, audio, video, etc.
-   **attention\_mask** (`torch.FloatTensor` of shape `batch_size, sequence_length`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.models.perceiver.modeling\_perceiver.PerceiverClassifierOutput](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.models.perceiver.modeling_perceiver.PerceiverClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([PerceiverConfig](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer plus the initial embedding outputs.
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.

The [PerceiverForMultimodalAutoencoding](/docs/transformers/v4.34.0/en/model_doc/perceiver#transformers.PerceiverForMultimodalAutoencoding) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import PerceiverForMultimodalAutoencoding
>>> import torch
>>> import numpy as np

>>> 
>>> images = torch.randn((1, 16, 3, 224, 224))
>>> audio = torch.randn((1, 30720, 1))
>>> inputs = dict(image=images, audio=audio, label=torch.zeros((images.shape[0], 700)))

>>> model = PerceiverForMultimodalAutoencoding.from_pretrained("deepmind/multimodal-perceiver")

>>> 
>>> 
>>> nchunks = 128
>>> image_chunk_size = np.prod((16, 224, 224)) // nchunks
>>> audio_chunk_size = audio.shape[1] // model.config.samples_per_patch // nchunks
>>> 
>>> chunk_idx = 0
>>> subsampling = {
...     "image": torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
...     "audio": torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
...     "label": None,
... }

>>> outputs = model(inputs=inputs, subsampled_output_points=subsampling)
>>> logits = outputs.logits
>>> list(logits["audio"].shape)
[1, 240]

>>> list(logits["image"].shape)
[1, 6272, 3]

>>> list(logits["label"].shape)
[1, 700]
```