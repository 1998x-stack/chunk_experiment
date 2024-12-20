# OpenAI GPT

[![Models](https://img.shields.io/badge/All_model_pages-openai--gpt-blueviolet)](https://huggingface.co/models?filter=openai-gpt) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/openai-gpt)

## Overview

OpenAI GPT model was proposed in [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. It’s a causal (unidirectional) transformer pre-trained using language modeling on a large corpus will long range dependencies, the Toronto Book Corpus.

The abstract from the paper is the following:

_Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering, semantic similarity assessment, and document classification. Although large unlabeled text corpora are abundant, labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to perform adequately. We demonstrate that large gains on these tasks can be realized by generative pretraining of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. In contrast to previous approaches, we make use of task-aware input transformations during fine-tuning to achieve effective transfer while requiring minimal changes to the model architecture. We demonstrate the effectiveness of our approach on a wide range of benchmarks for natural language understanding. Our general task-agnostic model outperforms discriminatively trained models that use architectures specifically crafted for each task, significantly improving upon the state of the art in 9 out of the 12 tasks studied._

Tips:

-   GPT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
-   GPT was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text as it can be observed in the _run\_generation.py_ example script.

[Write With Transformer](https://transformer.huggingface.co/doc/gpt) is a webapp created and hosted by Hugging Face showcasing the generative capabilities of several models. GPT is one of them.

This model was contributed by [thomwolf](https://huggingface.co/thomwolf). The original code can be found [here](https://github.com/openai/finetune-transformer-lm).

Note:

If you want to reproduce the original tokenization process of the _OpenAI GPT_ paper, you will need to install `ftfy` and `SpaCy`:

```
pip install spacy ftfy==4.4.3
python -m spacy download en
```

If you don’t install `ftfy` and `SpaCy`, the [OpenAIGPTTokenizer](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTTokenizer) will default to tokenize using BERT’s `BasicTokenizer` followed by Byte-Pair Encoding (which should be fine for most usage, don’t worry).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with OpenAI GPT. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

-   A blog post on [outperforming OpenAI GPT-3 with SetFit for text-classification](https://www.philschmid.de/getting-started-setfit).
-   See also: [Text classification task guide](../tasks/sequence_classification)

-   A blog on how to [Finetune a non-English GPT-2 Model with Hugging Face](https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface).
-   A blog on [How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate) with GPT-2.
-   A blog on [Training CodeParrot 🦜 from Scratch](https://huggingface.co/blog/codeparrot), a large GPT-2 model.
-   A blog on [Faster Text Generation with TensorFlow and XLA](https://huggingface.co/blog/tf-xla-generate) with GPT-2.
-   A blog on [How to train a Language Model with Megatron-LM](https://huggingface.co/blog/megatron-training) with a GPT-2 model.
-   A notebook on how to [finetune GPT2 to generate lyrics in the style of your favorite artist](https://colab.research.google.com/github/AlekseyKorshuk/huggingartists/blob/master/huggingartists-demo.ipynb). 🌎
-   A notebook on how to [finetune GPT2 to generate tweets in the style of your favorite Twitter user](https://colab.research.google.com/github/borisdayma/huggingtweets/blob/master/huggingtweets-demo.ipynb). 🌎
-   [Causal language modeling](https://huggingface.co/course/en/chapter7/6?fw=pt#training-a-causal-language-model-from-scratch) chapter of the 🤗 Hugging Face Course.
-   [OpenAIGPTLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTLMHeadModel) is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling), [text generation example script](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-generation/run_generation.py) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
-   [TFOpenAIGPTLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.TFOpenAIGPTLMHeadModel) is supported by this [causal language modeling example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_clmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
-   See also: [Causal language modeling task guide](../tasks/language_modeling)

-   A course material on [Byte-Pair Encoding tokenization](https://huggingface.co/course/en/chapter6/5).

## OpenAIGPTConfig

### class transformers.OpenAIGPTConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/configuration_openai.py#L27)

( vocab\_size = 40478 n\_positions = 512 n\_embd = 768 n\_layer = 12 n\_head = 12 afn = 'gelu' resid\_pdrop = 0.1 embd\_pdrop = 0.1 attn\_pdrop = 0.1 layer\_norm\_epsilon = 1e-05 initializer\_range = 0.02 summary\_type = 'cls\_index' summary\_use\_proj = True summary\_activation = None summary\_proj\_to\_labels = True summary\_first\_dropout = 0.1 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 40478) — Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [OpenAIGPTModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTModel) or [TFOpenAIGPTModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.TFOpenAIGPTModel).
-   **n\_positions** (`int`, _optional_, defaults to 512) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **n\_embd** (`int`, _optional_, defaults to 768) — Dimensionality of the embeddings and hidden states.
-   **n\_layer** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **n\_head** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **afn** (`str` or `Callable`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **resid\_pdrop** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **embd\_pdrop** (`int`, _optional_, defaults to 0.1) — The dropout ratio for the embeddings.
-   **attn\_pdrop** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention.
-   **layer\_norm\_epsilon** (`float`, _optional_, defaults to 1e-5) — The epsilon to use in the layer normalization layers
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **summary\_type** (`str`, _optional_, defaults to `"cls_index"`) — Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).
    
    Has to be one of the following options:
    
    -   `"last"`: Take the last token hidden state (like XLNet).
    -   `"first"`: Take the first token hidden state (like BERT).
    -   `"mean"`: Take the mean of all tokens hidden states.
    -   `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
    -   `"attn"`: Not implemented now, use multi-head attention.
    
-   **summary\_use\_proj** (`bool`, _optional_, defaults to `True`) — Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).
    
    Whether or not to add a projection after the vector extraction.
    
-   **summary\_activation** (`str`, _optional_) — Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).
    
    Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    
-   **summary\_proj\_to\_labels** (`bool`, _optional_, defaults to `True`) — Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).
    
    Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    
-   **summary\_first\_dropout** (`float`, _optional_, defaults to 0.1) — Argument used when doing sequence summary, used in the models [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) and [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel).
    
    The dropout ratio to be used after the projection and activation.
    
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [OpenAIGPTModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTModel) or a [TFOpenAIGPTModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.TFOpenAIGPTModel). It is used to instantiate a GPT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the GPT [openai-gpt](https://huggingface.co/openai-gpt) architecture from OpenAI.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import OpenAIGPTConfig, OpenAIGPTModel

>>> 
>>> configuration = OpenAIGPTConfig()

>>> 
>>> model = OpenAIGPTModel(configuration)

>>> 
>>> configuration = model.config
```

## OpenAIGPTTokenizer

### class transformers.OpenAIGPTTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/tokenization_openai.py#L245)

( vocab\_file merges\_file unk\_token = '<unk>' \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

Construct a GPT Tokenizer. Based on Byte-Pair-Encoding with the following peculiarities:

-   lowercases all inputs,
-   uses `SpaCy` tokenizer and `ftfy` for pre-BPE tokenization if they are installed, fallback to BERT’s `BasicTokenizer` if not.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/tokenization_openai.py#L378)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None )

## OpenAIGPTTokenizerFast

### class transformers.OpenAIGPTTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/tokenization_openai_fast.py#L40)

( vocab\_file = None merges\_file = None tokenizer\_file = None unk\_token = '<unk>' \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

Construct a “fast” GPT Tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on Byte-Pair-Encoding with the following peculiarities:

-   lower case all inputs
-   uses BERT’s BasicTokenizer for pre-BPE tokenization

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

## OpenAI specific outputs

### class transformers.models.openai.modeling\_openai.OpenAIGPTDoubleHeadsModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_openai.py#L298)

( loss: typing.Optional\[torch.FloatTensor\] = None mc\_loss: typing.Optional\[torch.FloatTensor\] = None logits: FloatTensor = None mc\_logits: FloatTensor = None hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss.
-   **mc\_loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `mc_labels` is provided) — Multiple choice classification loss.
-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **mc\_logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for outputs of models predicting if two sentences are consecutive or not.

### class transformers.models.openai.modeling\_tf\_openai.TFOpenAIGPTDoubleHeadsModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_tf_openai.py#L362)

( logits: tf.Tensor = None mc\_logits: tf.Tensor = None hidden\_states: Tuple\[tf.Tensor\] | None = None attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **logits** (`tf.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **mc\_logits** (`tf.Tensor` of shape `(batch_size, num_choices)`) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

Base class for outputs of models predicting if two sentences are consecutive or not.

## OpenAIGPTModel

### class transformers.OpenAIGPTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_openai.py#L402)

( config )

Parameters

-   **config** ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_openai.py#L428)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [OpenAIGPTModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, OpenAIGPTModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
>>> model = OpenAIGPTModel.from_pretrained("openai-gpt")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## OpenAIGPTLMHeadModel

### class transformers.OpenAIGPTLMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_openai.py#L533)

( config )

Parameters

-   **config** ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_openai.py#L550)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

A [transformers.modeling\_outputs.CausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [OpenAIGPTLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, OpenAIGPTLMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
>>> model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## OpenAIGPTDoubleHeadsModel

### class transformers.OpenAIGPTDoubleHeadsModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_openai.py#L624)

( config )

Parameters

-   **config** ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the input embeddings, the classification head takes as input the input of a specified classification token index in the input sequence).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_openai.py#L644)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None mc\_token\_ids: typing.Optional\[torch.LongTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None mc\_labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.openai.modeling\_openai.OpenAIGPTDoubleHeadsModelOutput](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **mc\_token\_ids** (`torch.LongTensor` of shape `(batch_size, num_choices)`, _optional_, default to index of the last token of the input) — Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) - 1]`.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set `labels = input_ids` Indices are selected in `[-1, 0, ..., config.vocab_size]` All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
-   **mc\_labels** (`torch.LongTensor` of shape `(batch_size)`, _optional_) — Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]` where _num\_choices_ is the size of the second dimension of the input tensors. (see _input\_ids_ above)

A [transformers.models.openai.modeling\_openai.OpenAIGPTDoubleHeadsModelOutput](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.models.openai.modeling_openai.OpenAIGPTDoubleHeadsModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss.
    
-   **mc\_loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `mc_labels` is provided) — Multiple choice classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **mc\_logits** (`torch.FloatTensor` of shape `(batch_size, num_choices)`) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [OpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTDoubleHeadsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, OpenAIGPTDoubleHeadsModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
>>> model = OpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")
>>> tokenizer.add_special_tokens(
...     {"cls_token": "[CLS]"}
... )  
>>> model.resize_token_embeddings(len(tokenizer))

>>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
>>> input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  
>>> mc_token_ids = torch.tensor([input_ids.size(-1) - 1, input_ids.size(-1) - 1]).unsqueeze(0)  

>>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
>>> lm_logits = outputs.logits
>>> mc_logits = outputs.mc_logits
```

## OpenAIGPTForSequenceClassification

### class transformers.OpenAIGPTForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_openai.py#L752)

( config )

Parameters

-   **config** ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Original OpenAI GPT Model transformer with a sequence classification head on top (linear layer). [OpenAIGPTForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the last token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_openai.py#L762)

( input\_ids: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [OpenAIGPTForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, OpenAIGPTForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
>>> model = OpenAIGPTForSequenceClassification.from_pretrained("openai-gpt")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = OpenAIGPTForSequenceClassification.from_pretrained("openai-gpt", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, OpenAIGPTForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
>>> model = OpenAIGPTForSequenceClassification.from_pretrained("openai-gpt", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = OpenAIGPTForSequenceClassification.from_pretrained(
...     "openai-gpt", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## TFOpenAIGPTModel

### class transformers.TFOpenAIGPTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_tf_openai.py#L492)

( \*args \*\*kwargs )

Parameters

-   **config** ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare OpenAI GPT transformer model outputting raw hidden-states without any specific head on top.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_tf_openai.py#L497)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

A [transformers.modeling\_tf\_outputs.TFBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **hidden\_states** (`tuple(tf.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFOpenAIGPTModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.TFOpenAIGPTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFOpenAIGPTModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
>>> model = TFOpenAIGPTModel.from_pretrained("openai-gpt")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFOpenAIGPTLMHeadModel

### class transformers.TFOpenAIGPTLMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_tf_openai.py#L539)

( \*args \*\*kwargs )

Parameters

-   **config** ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OpenAI GPT Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_tf_openai.py#L552)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFCausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the cross entropy classification loss. Indices should be in `[0, ..., config.vocab_size - 1]`.

A [transformers.modeling\_tf\_outputs.TFCausalLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFCausalLMOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(n,)`, _optional_, where n is the number of non-masked labels, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFOpenAIGPTLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.TFOpenAIGPTLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFOpenAIGPTLMHeadModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
>>> model = TFOpenAIGPTLMHeadModel.from_pretrained("openai-gpt")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)
>>> logits = outputs.logits
```

## TFOpenAIGPTDoubleHeadsModel

### class transformers.TFOpenAIGPTDoubleHeadsModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_tf_openai.py#L626)

( \*args \*\*kwargs )

Parameters

-   **config** ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OpenAI GPT Model transformer with a language modeling and a multiple-choice classification head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers. The language modeling head has its weights tied to the input embeddings, the classification head takes as input the input of a specified classification token index in the input sequence).

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_tf_openai.py#L635)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None mc\_token\_ids: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → [transformers.models.openai.modeling\_tf\_openai.TFOpenAIGPTDoubleHeadsModelOutput](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.models.openai.modeling_tf_openai.TFOpenAIGPTDoubleHeadsModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **mc\_token\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, num_choices)`, _optional_, default to index of the last token of the input) — Index of the classification token in each input sequence. Selected in the range `[0, input_ids.size(-1) - 1]`.

A [transformers.models.openai.modeling\_tf\_openai.TFOpenAIGPTDoubleHeadsModelOutput](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.models.openai.modeling_tf_openai.TFOpenAIGPTDoubleHeadsModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

-   **logits** (`tf.Tensor` of shape `(batch_size, num_choices, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **mc\_logits** (`tf.Tensor` of shape `(batch_size, num_choices)`) — Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFOpenAIGPTDoubleHeadsModel](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.TFOpenAIGPTDoubleHeadsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import tensorflow as tf
>>> from transformers import AutoTokenizer, TFOpenAIGPTDoubleHeadsModel

>>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
>>> model = TFOpenAIGPTDoubleHeadsModel.from_pretrained("openai-gpt")

>>> 
>>> tokenizer.add_special_tokens({"cls_token": "[CLS]"})
>>> model.resize_token_embeddings(len(tokenizer))  
>>> print(tokenizer.cls_token_id, len(tokenizer))  

>>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
>>> encoding = tokenizer(choices, return_tensors="tf")
>>> inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
>>> inputs["mc_token_ids"] = tf.constant(
...     [inputs["input_ids"].shape[-1] - 1, inputs["input_ids"].shape[-1] - 1]
... )[
...     None, :
... ]  
>>> outputs = model(inputs)
>>> lm_prediction_scores, mc_prediction_scores = outputs[:2]
```

## TFOpenAIGPTForSequenceClassification

### class transformers.TFOpenAIGPTForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_tf_openai.py#L753)

( \*args \*\*kwargs )

Parameters

-   **config** ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The OpenAI GPT Model transformer with a sequence classification head on top (linear layer).

[TFOpenAIGPTForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.TFOpenAIGPTForSequenceClassification) uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/openai/modeling_tf_openai.py#L765)

( input\_ids: TFModelInputType | None = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`tf.Tensor` or `Numpy array` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` or `Numpy array` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the cross entropy classification loss. Indices should be in `[0, ..., config.vocab_size - 1]`.

A [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([OpenAIGPTConfig](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.OpenAIGPTConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(batch_size, )`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFOpenAIGPTForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/openai-gpt#transformers.TFOpenAIGPTForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFOpenAIGPTForSequenceClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
>>> model = TFOpenAIGPTForSequenceClassification.from_pretrained("openai-gpt")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

>>> logits = model(**inputs).logits

>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TFOpenAIGPTForSequenceClassification.from_pretrained("openai-gpt", num_labels=num_labels)

>>> labels = tf.constant(1)
>>> loss = model(**inputs, labels=labels).loss
```