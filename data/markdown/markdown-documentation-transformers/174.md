# OpenAI GPT2

[![Models](https://img.shields.io/badge/All_model_pages-gpt2-blueviolet)](https://huggingface.co/models?filter=gpt2) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/gpt2)

## Overview

OpenAI GPT-2 model was proposed in [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever from [OpenAI](https://huggingface.co/openai). It’s a causal (unidirectional) transformer pretrained using language modeling on a very large corpus of ~40 GB of text data.

The abstract from the paper is the following:

_GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset\[1\] of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data._

Tips:

-   GPT-2 is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
-   GPT-2 was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text as it can be observed in the _run\_generation.py_ example script.
-   The model can take the _past\_key\_values_ (for PyTorch) or _past_ (for TF) as input, which is the previously computed key/value attention pairs. Using this (_past\_key\_values_ or _past_) value prevents the model from re-computing pre-computed values in the context of text generation. For PyTorch, see _past\_key\_values_ argument of the [GPT2Model.forward()](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Model.forward) method, or for TF the _past_ argument of the [TFGPT2Model.call()](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2Model.call) method for more information on its usage.
-   Enabling the _scale\_attn\_by\_inverse\_layer\_idx_ and _reorder\_and\_upcast\_attn_ flags will apply the training stability improvements from [Mistral](https://github.com/stanford-crfm/mistral/) (for PyTorch only).

[Write With Transformer](https://transformer.huggingface.co/doc/gpt2-large) is a webapp created and hosted by Hugging Face showcasing the generative capabilities of several models. GPT-2 is one of them and is available in five different sizes: small, medium, large, xl and a distilled version of the small checkpoint: _distilgpt-2_.

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://openai.com/blog/better-language-models/).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with GPT2. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

-   A blog on how to [Finetune a non-English GPT-2 Model with Hugging Face](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface).
-   A blog on [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) with GPT-2.
-   A blog on [Training CodeParrot 🦜 from Scratch](https://huggingface.co/blog/codeparrot), a large GPT-2 model.
-   A blog on [Faster Text Generation with TensorFlow and XLA](https://huggingface.co/blog/tf-xla-generate) with GPT-2.
-   A blog on [How to train a Language Model with Megatron-LM](https://huggingface.co/blog/megatron-training) with a GPT-2 model.
-   A notebook on how to [finetune GPT2 to generate lyrics in the style of your favorite artist](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb). 🌎
-   A notebook on how to [finetune GPT2 to generate tweets in the style of your favorite Twitter user](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb). 🌎
-   [Causal language modeling](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) chapter of the 🤗 Hugging Face Course.
-   [GPT2LMHeadModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2LMHeadModel) is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling), [text generation example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-generation), and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
-   [TFGPT2LMHeadModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2LMHeadModel) is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
-   [FlaxGPT2LMHeadModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.FlaxGPT2LMHeadModel) is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#causal-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/causal_language_modeling_flax.ipynb).
-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Causal language modeling task guide](../tasks/language_modeling)

## GPT2Config

### class transformers.GPT2Config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/configuration_gpt2.py#L37)

( vocab\_size = 50257 n\_positions = 1024 n\_embd = 768 n\_layer = 12 n\_head = 12 n\_inner = None activation\_function = 'gelu\_new' resid\_pdrop = 0.1 embd\_pdrop = 0.1 attn\_pdrop = 0.1 layer\_norm\_epsilon = 1e-05 initializer\_range = 0.02 summary\_type = 'cls\_index' summary\_use\_proj = True summary\_activation = None summary\_proj\_to\_labels = True summary\_first\_dropout = 0.1 scale\_attn\_weights = True use\_cache = True bos\_token\_id = 50256 eos\_token\_id = 50256 scale\_attn\_by\_inverse\_layer\_idx = False reorder\_and\_upcast\_attn = False \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 50257) — Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [GPT2Model](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Model) or [TFGPT2Model](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2Model).
-   **n\_positions** (`int`, _optional_, defaults to 1024) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **n\_embd** (`int`, _optional_, defaults to 768) — Dimensionality of the embeddings and hidden states.
-   **n\_layer** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **n\_head** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **n\_inner** (`int`, _optional_, defaults to None) — Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n\_embd
-   **activation\_function** (`str`, _optional_, defaults to `"gelu_new"`) — Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
-   **resid\_pdrop** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **embd\_pdrop** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the embeddings.
-   **attn\_pdrop** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention.
-   **layer\_norm\_epsilon** (`float`, _optional_, defaults to 1e-5) — The epsilon to use in the layer normalization layers.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **summary\_type** (`string`, _optional_, defaults to `"cls_index"`) — Argument used when doing sequence summary, used in the models [GPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel) and [TFGPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2DoubleHeadsModel).
    
    Has to be one of the following options:
    
    -   `"last"`: Take the last token hidden state (like XLNet).
    -   `"first"`: Take the first token hidden state (like BERT).
    -   `"mean"`: Take the mean of all tokens hidden states.
    -   `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
    -   `"attn"`: Not implemented now, use multi-head attention.
    
-   **summary\_use\_proj** (`bool`, _optional_, defaults to `True`) — Argument used when doing sequence summary, used in the models [GPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel) and [TFGPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2DoubleHeadsModel).
    
    Whether or not to add a projection after the vector extraction.
    
-   **summary\_activation** (`str`, _optional_) — Argument used when doing sequence summary. Used in for the multiple choice head in [GPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel).
    
    Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    
-   **summary\_proj\_to\_labels** (`bool`, _optional_, defaults to `True`) — Argument used when doing sequence summary, used in the models [GPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel) and [TFGPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2DoubleHeadsModel).
    
    Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    
-   **summary\_first\_dropout** (`float`, _optional_, defaults to 0.1) — Argument used when doing sequence summary, used in the models [GPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel) and [TFGPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2DoubleHeadsModel).
    
    The dropout ratio to be used after the projection and activation.
    
-   **scale\_attn\_weights** (`bool`, _optional_, defaults to `True`) — Scale attention weights by dividing by sqrt(hidden\_size)..
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models).
-   **scale\_attn\_by\_inverse\_layer\_idx** (`bool`, _optional_, defaults to `False`) — Whether to additionally scale attention weights by `1 / layer_idx + 1`.
-   **reorder\_and\_upcast\_attn** (`bool`, _optional_, defaults to `False`) — Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention dot-product/softmax to float() when training with mixed precision.

This is the configuration class to store the configuration of a [GPT2Model](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Model) or a [TFGPT2Model](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2Model). It is used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the GPT-2 [gpt2](https://huggingface.co/gpt2) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import GPT2Config, GPT2Model

>>> 
>>> configuration = GPT2Config()

>>> 
>>> model = GPT2Model(configuration)

>>> 
>>> configuration = model.config
```

## GPT2Tokenizer

### class transformers.GPT2Tokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/tokenization_gpt2.py#L101)

( vocab\_file merges\_file errors = 'replace' unk\_token = '<|endoftext|>' bos\_token = '<|endoftext|>' eos\_token = '<|endoftext|>' pad\_token = None add\_prefix\_space = False add\_bos\_token = False \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **errors** (`str`, _optional_, defaults to `"replace"`) — Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
-   **unk\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **bos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The end of sequence token.
-   **add\_prefix\_space** (`bool`, _optional_, defaults to `False`) — Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (GPT2 tokenizer detect beginning of words by the preceding space).

Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import GPT2Tokenizer

>>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>>> tokenizer("Hello world")["input_ids"]
[15496, 995]

>>> tokenizer(" Hello world")["input_ids"]
[18435, 995]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/tokenization_gpt2.py#L321)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None )

## GPT2TokenizerFast

### class transformers.GPT2TokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/tokenization_gpt2_fast.py#L66)

( vocab\_file = None merges\_file = None tokenizer\_file = None unk\_token = '<|endoftext|>' bos\_token = '<|endoftext|>' eos\_token = '<|endoftext|>' add\_prefix\_space = False \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **errors** (`str`, _optional_, defaults to `"replace"`) — Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
-   **unk\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **bos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The end of sequence token.
-   **add\_prefix\_space** (`bool`, _optional_, defaults to `False`) — Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (GPT2 tokenizer detect beginning of words by the preceding space).
-   **trim\_offsets** (`bool`, _optional_, defaults to `True`) — Whether or not the post-processing step should trim offsets to avoid including whitespaces.

Construct a “fast” GPT-2 tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import GPT2TokenizerFast

>>> tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
>>> tokenizer("Hello world")["input_ids"]
[15496, 995]

>>> tokenizer(" Hello world")["input_ids"]
[18435, 995]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

## GPT2 specific outputs

### class transformers.models.gpt2.modeling\_gpt2.GPT2DoubleHeadsModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L489)

( loss: typing.Optional\[torch.FloatTensor\] = None mc\_loss: typing.Optional\[torch.FloatTensor\] = None logits: FloatTensor = None mc\_logits: FloatTensor = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss.
-   **mc\_loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `mc_labels` is provided) — Multiple choice classification loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **mc\_logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    GPT2Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for outputs of models predicting if two sentences are consecutive or not.

### class transformers.models.gpt2.modeling\_tf\_gpt2.TFGPT2DoubleHeadsModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_tf_gpt2.py#L526)

( logits: tf.Tensor = None mc\_logits: tf.Tensor = None past\_key\_values: List\[tf.Tensor\] | None = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **logits** (`tf.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **mc\_logits** (`tf.Tensor` of shape `(batch_size, num_choices)`) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
-   **past\_key\_values** (`List[tf.Tensor]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for outputs of models predicting if two sentences are consecutive or not.

## GPT2Model

### class transformers.GPT2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L669)

( config )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L747)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None encoder\_hidden\_states: typing.Optional\[torch.Tensor\] = None encoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
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

A [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

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
    

The [GPT2Model](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, GPT2Model
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = GPT2Model.from_pretrained("gpt2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## GPT2LMHeadModel

### class transformers.GPT2LMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L956)

( config )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1045)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None encoder\_hidden\_states: typing.Optional\[torch.Tensor\] = None encoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
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

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

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
    

The [GPT2LMHeadModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2LMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, GPT2LMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = GPT2LMHeadModel.from_pretrained("gpt2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## GPT2DoubleHeadsModel

### class transformers.GPT2DoubleHeadsModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1148)

( config )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the input embeddings, the classification head takes as input the input of a specified classification token index in the input sequence).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1233)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None mc\_token\_ids: typing.Optional\[torch.LongTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None mc\_labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None \*\*kwargs ) → [transformers.models.gpt2.modeling\_gpt2.GPT2DoubleHeadsModelOutput](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
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
-   **mc\_token\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices)`, _optional_, default to index of the last token of the input) — Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) - 1]`.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set `labels = input_ids`. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size - 1]`
-   **mc\_labels** (`torch.LongTensor` of shape `(batch_size)`, _optional_) — Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]` where _num\_choices_ is the size of the second dimension of the input tensors. (see _input\_ids_ above)

A [transformers.models.gpt2.modeling\_gpt2.GPT2DoubleHeadsModelOutput](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss.
    
-   **mc\_loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `mc_labels` is provided) — Multiple choice classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **mc\_logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    GPT2Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [GPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2DoubleHeadsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, GPT2DoubleHeadsModel

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = GPT2DoubleHeadsModel.from_pretrained("gpt2")

>>> 
>>> num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})
>>> 
>>> embedding_layer = model.resize_token_embeddings(len(tokenizer))

>>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
>>> encoded_choices = [tokenizer.encode(s) for s in choices]
>>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

>>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  
>>> mc_token_ids = torch.tensor([cls_token_location])  

>>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
>>> lm_logits = outputs.logits
>>> mc_logits = outputs.mc_logits
```

## GPT2ForQuestionAnswering

### class transformers.GPT2ForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1599)

( config )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GPT-2 Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1614)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None start\_positions: typing.Optional\[torch.LongTensor\] = None end\_positions: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
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

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    
-   **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
    
-   **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [GPT2ForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2ForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

This example uses a random model as the real ones are all very big. To get proper results, you should use gpt2 instead of gpt2. If you get out-of-memory when loading that checkpoint, you can try adding `device_map="auto"` in the `from_pretrained` call.

Example:

```
>>> from transformers import AutoTokenizer, GPT2ForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = GPT2ForQuestionAnswering.from_pretrained("gpt2")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]

>>> 
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
```

## GPT2ForSequenceClassification

### class transformers.GPT2ForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1376)

( config )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GPT2 Model transformer with a sequence classification head on top (linear layer).

[GPT2ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-1) do.

Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1390)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
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

A `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [GPT2ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, GPT2ForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/DialogRPT-updown")
>>> model = GPT2ForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = GPT2ForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, GPT2ForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/DialogRPT-updown")
>>> model = GPT2ForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = GPT2ForSequenceClassification.from_pretrained(
...     "microsoft/DialogRPT-updown", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## GPT2ForTokenClassification

### class transformers.GPT2ForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1501)

( config )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

GPT2 Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_gpt2.py#L1523)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`) — Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have their past given to this model should not be passed as `input_ids` as they have already been computed.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
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
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [GPT2ForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2ForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, GPT2ForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("brad1141/gpt2-finetuned-comp2")
>>> model = GPT2ForTokenClassification.from_pretrained("brad1141/gpt2-finetuned-comp2")

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
>>> predicted_tokens_classes
['Lead', 'Lead', 'Lead', 'Position', 'Lead', 'Lead', 'Lead', 'Lead', 'Lead', 'Lead', 'Lead', 'Lead']

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
0.25
```

## TFGPT2Model

### class transformers.TFGPT2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_tf_gpt2.py#L675)

( \*args \*\*kwargs )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_tf_gpt2.py#L680)

( input\_ids: TFModelInputType | None = None past\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None encoder\_hidden\_states: np.ndarray | tf.Tensor | None = None encoder\_attention\_mask: np.ndarray | tf.Tensor | None = None use\_cache: Optional\[bool\] = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPastAndCrossAttentions) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
-   **attention\_mask** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **encoder\_hidden\_states** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.
-   **encoder\_attention\_mask** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
-   **past\_key\_values** (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`) — contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past`). Set to `False` during training, `True` during generation

A [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPastAndCrossAttentions) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`List[tf.Tensor]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(tf.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    

The [TFGPT2Model](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFGPT2Model
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = TFGPT2Model.from_pretrained("gpt2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFGPT2LMHeadModel

### class transformers.TFGPT2LMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_tf_gpt2.py#L752)

( \*args \*\*kwargs )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_tf_gpt2.py#L788)

( input\_ids: TFModelInputType | None = None past\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None encoder\_hidden\_states: np.ndarray | tf.Tensor | None = None encoder\_attention\_mask: np.ndarray | tf.Tensor | None = None use\_cache: Optional\[bool\] = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
-   **attention\_mask** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **encoder\_hidden\_states** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.
-   **encoder\_attention\_mask** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
-   **past\_key\_values** (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`) — contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past`). Set to `False` during training, `True` during generation
-   **labels** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the cross entropy classification loss. Indices should be in `[0, ..., config.vocab_size - 1]`.

A [transformers.modeling\_tf\_outputs.TFCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **loss** (`tf.Tensor` of shape `(n,)`, _optional_, where n is the number of non-masked labels, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **past\_key\_values** (`List[tf.Tensor]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    

The [TFGPT2LMHeadModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2LMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFGPT2LMHeadModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = TFGPT2LMHeadModel.from_pretrained("gpt2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

## TFGPT2DoubleHeadsModel

### class transformers.TFGPT2DoubleHeadsModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_tf_gpt2.py#L886)

( \*args \*\*kwargs )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GPT2 Model transformer with a language modeling and a multiple-choice classification head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the input embeddings, the classification head takes as input the input of a specified classification token index in the input sequence).

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_tf_gpt2.py#L895)

( input\_ids: TFModelInputType | None = None past\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None mc\_token\_ids: np.ndarray | tf.Tensor | None = None use\_cache: Optional\[bool\] = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → [transformers.models.gpt2.modeling\_tf\_gpt2.TFGPT2DoubleHeadsModelOutput](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.models.gpt2.modeling_tf_gpt2.TFGPT2DoubleHeadsModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
-   **attention\_mask** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **mc\_token\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, num_choices)`, _optional_, default to index of the last token of the input) — Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) - 1]`.

A [transformers.models.gpt2.modeling\_tf\_gpt2.TFGPT2DoubleHeadsModelOutput](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.models.gpt2.modeling_tf_gpt2.TFGPT2DoubleHeadsModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **logits** (`tf.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **mc\_logits** (`tf.Tensor` of shape `(batch_size, num_choices)`) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
    
-   **past\_key\_values** (`List[tf.Tensor]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFGPT2DoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2DoubleHeadsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import tensorflow as tf
>>> from transformers import AutoTokenizer, TFGPT2DoubleHeadsModel

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = TFGPT2DoubleHeadsModel.from_pretrained("gpt2")

>>> 
>>> num_added_tokens = tokenizer.add_special_tokens({"cls_token": "[CLS]"})

>>> embedding_layer = model.resize_token_embeddings(
...     len(tokenizer)
... )  

>>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
>>> encoded_choices = [tokenizer.encode(s) for s in choices]
>>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

>>> input_ids = tf.constant(encoded_choices)[None, :]  
>>> mc_token_ids = tf.constant([cls_token_location])  

>>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
>>> lm_prediction_scores, mc_prediction_scores = outputs[:2]
```

## TFGPT2ForSequenceClassification

### class transformers.TFGPT2ForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_tf_gpt2.py#L1021)

( \*args \*\*kwargs )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The GPT2 Model transformer with a sequence classification head on top (linear layer).

[TFGPT2ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2ForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-1) do.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_tf_gpt2.py#L1033)

( input\_ids: TFModelInputType | None = None past\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None use\_cache: Optional\[bool\] = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutputWithPast](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.modeling_tf_outputs.TFSequenceClassifierOutputWithPast) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length` if `past_key_values` is `None` else `past_key_values[0].shape[-2]` (`sequence_length` of input past key value states). Indices of input sequence tokens in the vocabulary.
    
    If `past_key_values` is used, only input IDs that do not have their past calculated should be passed as `input_ids`.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **past\_key\_values** (`List[tf.Tensor]` of length `config.n_layers`) — Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see `past_key_values` output below). Can be used to speed up sequential decoding. The token ids which have their past given to this model should not be passed as input ids as they have already been computed.
-   **attention\_mask** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for `past_key_values`. In other words, the `attention_mask` always has to have the length: `len(past_key_values) + len(input_ids)`
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the cross entropy classification loss. Indices should be in `[0, ..., config.vocab_size - 1]`.

A [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutputWithPast](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.modeling_tf_outputs.TFSequenceClassifierOutputWithPast) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **loss** (`tf.Tensor` of shape `(batch_size, )`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **past\_key\_values** (`List[tf.Tensor]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFGPT2ForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.TFGPT2ForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFGPT2ForSequenceClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/DialogRPT-updown")
>>> model = TFGPT2ForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

>>> logits = model(**inputs).logits

>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TFGPT2ForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown", num_labels=num_labels)

>>> labels = tf.constant(1)
>>> loss = model(**inputs, labels=labels).loss
```

## TFSequenceClassifierOutputWithPast

### class transformers.modeling\_tf\_outputs.TFSequenceClassifierOutputWithPast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/modeling_tf_outputs.py#L901)

( loss: tf.Tensor | None = None logits: tf.Tensor = None past\_key\_values: List\[tf.Tensor\] | None = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **loss** (`tf.Tensor` of shape `(batch_size, )`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
-   **past\_key\_values** (`List[tf.Tensor]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for outputs of sentence classification models.

## TFGPT2Tokenizer

### class transformers.TFGPT2Tokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/tokenization_gpt2_tf.py#L11)

( \*args \*\*kwargs )

Parameters

-   **vocab** (Dict\[str, int\]) — Vocabulary dict for Byte Pair Tokenizer
-   **merges** (List\[str\]) — Merges list for Byte Pair Tokenizer

This is an in-graph tokenizer for GPT2. It should be initialized similarly to other tokenizers, using the `from_pretrained()` method. It can also be initialized with the `from_tokenizer()` method, which imports settings from an existing standard tokenizer object.

In-graph tokenizers, unlike other Hugging Face tokenizers, are actually Keras layers and are designed to be run when the model is called, rather than during preprocessing. As a result, they have somewhat more limited options than standard tokenizer classes. They are most useful when you want to create an end-to-end model that goes straight from `tf.string` inputs to outputs.

#### from\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/tokenization_gpt2_tf.py#L73)

( config )

Parameters

-   **config** (Dict) — Dictionary with keys such as stated in `get_config`.

Creates TFGPT2Tokenizer from configurations

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/tokenization_gpt2_tf.py#L55)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\] \*init\_inputs \*\*kwargs )

Parameters

-   **pretrained\_model\_name\_or\_path** (Union\[str, os.PathLike\]) — Path to pretrained model

Creates TFGPT2Tokenizer from pretrained GPT2Tokenizer

Examples:

```
from transformers import TFGPT2Tokenizer

tf_tokenizer = TFGPT2Tokenizer.from_pretrained("gpt2")
```

#### from\_tokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/tokenization_gpt2_tf.py#L35)

( tokenizer: GPT2Tokenizer \*args \*\*kwargs )

Parameters

-   **tokenizer** (GPT2Tokenizer) —

Creates TFGPT2Tokenizer from GPT2Tokenizer

Examples:

```
from transformers import AutoTokenizer, TFGPT2Tokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tf_tokenizer = TFGPT2Tokenizer.from_tokenizer(tokenizer)
```

## FlaxGPT2Model

### class transformers.FlaxGPT2Model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_flax_gpt2.py#L665)

( config: GPT2Config input\_shape: typing.Tuple = (1, 1) seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True \*\*kwargs )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_flax_gpt2.py#L456)

( input\_ids attention\_mask = None position\_ids = None encoder\_hidden\_states: typing.Optional\[jax.Array\] = None encoder\_attention\_mask: typing.Optional\[jax.Array\] = None params: dict = None past\_key\_values: dict = None dropout\_rng: PRNGKey = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **past\_key\_values** (`Dict[str, np.ndarray]`, _optional_, returned by `init_cache` or when passing previous `past_key_values`) — Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast auto-regressive decoding. Pre-computed key and value hidden-states are of shape _\[batch\_size, max\_length\]_.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **last\_hidden\_state** (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`tuple(tuple(jnp.ndarray))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    

The `FlaxGPT2PreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxGPT2Model

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = FlaxGPT2Model.from_pretrained("gpt2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## FlaxGPT2LMHeadModel

### class transformers.FlaxGPT2LMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_flax_gpt2.py#L742)

( config: GPT2Config input\_shape: typing.Tuple = (1, 1) seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True \*\*kwargs )

Parameters

-   **config** ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt2/modeling_flax_gpt2.py#L456)

( input\_ids attention\_mask = None position\_ids = None encoder\_hidden\_states: typing.Optional\[jax.Array\] = None encoder\_attention\_mask: typing.Optional\[jax.Array\] = None params: dict = None past\_key\_values: dict = None dropout\_rng: PRNGKey = None train: bool = False output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`numpy.ndarray` of shape `(batch_size, input_ids_length)`) — `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **past\_key\_values** (`Dict[str, np.ndarray]`, _optional_, returned by `init_cache` or when passing previous `past_key_values`) — Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast auto-regressive decoding. Pre-computed key and value hidden-states are of shape _\[batch\_size, max\_length\]_.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([GPT2Config](/docs/transformers/v4.34.0/en/model_doc/gpt2#transformers.GPT2Config)) and inputs.

-   **logits** (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Cross attentions weights after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **past\_key\_values** (`tuple(tuple(jnp.ndarray))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `jnp.ndarray` tuples of length `config.n_layers`, with each tuple containing the cached key, value states of the self-attention and the cross-attention layers if model is used in encoder-decoder setting. Only relevant if `config.is_decoder = True`.
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    

The `FlaxGPT2PreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxGPT2LMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
>>> model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
>>> outputs = model(**inputs)

>>> 
>>> next_token_logits = outputs.logits[:, -1]
```