# CodeGen

## Overview

The CodeGen model was proposed in [A Conversational Paradigm for Program Synthesis](https://arxiv.org/abs/2203.13474) by Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong.

CodeGen is an autoregressive language model for program synthesis trained sequentially on [The Pile](https://pile.eleuther.ai/), BigQuery, and BigPython.

The abstract from the paper is the following:

_Program synthesis strives to generate a computer program as a solution to a given problem specification. We propose a conversational program synthesis approach via large language models, which addresses the challenges of searching over a vast program space and user intent specification faced in prior approaches. Our new approach casts the process of writing a specification and program as a multi-turn conversation between a user and a system. It treats program synthesis as a sequence prediction problem, in which the specification is expressed in natural language and the desired program is conditionally sampled. We train a family of large language models, called CodeGen, on natural language and programming language data. With weak supervision in the data and the scaling up of data size and model size, conversational capacities emerge from the simple autoregressive language modeling. To study the model behavior on conversational program synthesis, we develop a multi-turn programming benchmark (MTPB), where solving each problem requires multi-step synthesis via multi-turn conversation between the user and the model. Our findings show the emergence of conversational capabilities and the effectiveness of the proposed conversational program synthesis paradigm. In addition, our model CodeGen (with up to 16B parameters trained on TPU-v4) outperforms OpenAI’s Codex on the HumanEval benchmark. We make the training library JaxFormer including checkpoints available as open source contribution: [this https URL](https://github.com/salesforce/codegen)._

This model was contributed by [Hiroaki Hayashi](https://huggingface.co/rooa). The original code can be found [here](https://github.com/salesforce/codegen).

## Checkpoint Naming

-   CodeGen model [checkpoints](https://huggingface.co/models?other=codegen) are available on different pre-training data with variable sizes.
-   The format is: `Salesforce/codegen-{size}-{data}`, where
    -   `size`: `350M`, `2B`, `6B`, `16B`
    -   `data`:
        -   `nl`: Pre-trained on the Pile
        -   `multi`: Initialized with `nl`, then further pre-trained on multiple programming languages data
        -   `mono`: Initialized with `multi`, then further pre-trained on Python data
-   For example, `Salesforce/codegen-350M-mono` offers a 350 million-parameter checkpoint pre-trained sequentially on the Pile, multiple programming languages, and Python.

## How to use

```
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> checkpoint = "Salesforce/codegen-350M-mono"
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)

>>> text = "def hello_world():"

>>> completion = model.generate(**tokenizer(text, return_tensors="pt"))

>>> print(tokenizer.decode(completion[0]))
def hello_world():
    print("Hello World")

hello_world()
```

## Documentation resources

-   [Causal language modeling task guide](../tasks/language_modeling)

## CodeGenConfig

### class transformers.CodeGenConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/codegen/configuration_codegen.py#L44)

( vocab\_size = 50400 n\_positions = 2048 n\_ctx = 2048 n\_embd = 4096 n\_layer = 28 n\_head = 16 rotary\_dim = 64 n\_inner = None activation\_function = 'gelu\_new' resid\_pdrop = 0.0 embd\_pdrop = 0.0 attn\_pdrop = 0.0 layer\_norm\_epsilon = 1e-05 initializer\_range = 0.02 use\_cache = True bos\_token\_id = 50256 eos\_token\_id = 50256 tie\_word\_embeddings = False \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 50400) — Vocabulary size of the CodeGen model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [CodeGenModel](/docs/transformers/v4.34.0/en/model_doc/codegen#transformers.CodeGenModel).
-   **n\_positions** (`int`, _optional_, defaults to 2048) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **n\_embd** (`int`, _optional_, defaults to 4096) — Dimensionality of the embeddings and hidden states.
-   **n\_layer** (`int`, _optional_, defaults to 28) — Number of hidden layers in the Transformer encoder.
-   **n\_head** (`int`, _optional_, defaults to 16) — Number of attention heads for each attention layer in the Transformer encoder.
-   **rotary\_dim** (`int`, _optional_, defaults to 64) — Number of dimensions in the embedding that Rotary Position Embedding is applied to.
-   **n\_inner** (`int`, _optional_, defaults to None) — Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n\_embd
-   **activation\_function** (`str`, _optional_, defaults to `"gelu_new"`) — Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
-   **resid\_pdrop** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **embd\_pdrop** (`int`, _optional_, defaults to 0.1) — The dropout ratio for the embeddings.
-   **attn\_pdrop** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention.
-   **layer\_norm\_epsilon** (`float`, _optional_, defaults to 1e-5) — The epsilon to use in the layer normalization layers.
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models).

This is the configuration class to store the configuration of a [CodeGenModel](/docs/transformers/v4.34.0/en/model_doc/codegen#transformers.CodeGenModel). It is used to instantiate a CodeGen model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the CodeGen [Salesforce/codegen-2B-mono](https://huggingface.co/Salesforce/codegen-2B-mono) architecture. Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import CodeGenConfig, CodeGenModel

>>> 
>>> configuration = CodeGenConfig()

>>> 
>>> model = CodeGenModel(configuration)

>>> 
>>> configuration = model.config
```

## CodeGenTokenizer

### class transformers.CodeGenTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/codegen/tokenization_codegen.py#L98)

( vocab\_file merges\_file errors = 'replace' unk\_token = '<|endoftext|>' bos\_token = '<|endoftext|>' eos\_token = '<|endoftext|>' pad\_token = None add\_prefix\_space = False add\_bos\_token = False \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **errors** (`str`, _optional_, defaults to `"replace"`) — Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
-   **unk\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **bos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The end of sequence token.
-   **add\_prefix\_space** (`bool`, _optional_, defaults to `False`) — Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (CodeGen tokenizer detect beginning of words by the preceding space).

Construct a CodeGen tokenizer. Based on byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import CodeGenTokenizer

>>> tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
>>> tokenizer("Hello world")["input_ids"]
[15496, 995]

>>> tokenizer(" Hello world")["input_ids"]
[18435, 995]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/codegen/tokenization_codegen.py#L284)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None )

## CodeGenTokenizerFast

### class transformers.CodeGenTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/codegen/tokenization_codegen_fast.py#L63)

( vocab\_file = None merges\_file = None tokenizer\_file = None unk\_token = '<|endoftext|>' bos\_token = '<|endoftext|>' eos\_token = '<|endoftext|>' add\_prefix\_space = False \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **merges\_file** (`str`) — Path to the merges file.
-   **errors** (`str`, _optional_, defaults to `"replace"`) — Paradigm to follow when decoding bytes to UTF-8. See [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
-   **unk\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **bos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `<|endoftext|>`) — The end of sequence token.
-   **add\_prefix\_space** (`bool`, _optional_, defaults to `False`) — Whether or not to add an initial space to the input. This allows to treat the leading word just as any other word. (CodeGen tokenizer detect beginning of words by the preceding space).
-   **trim\_offsets** (`bool`, _optional_, defaults to `True`) — Whether or not the post-processing step should trim offsets to avoid including whitespaces.

Construct a “fast” CodeGen tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import CodeGenTokenizerFast

>>> tokenizer = CodeGenTokenizerFast.from_pretrained("Salesforce/codegen-350M-mono")
>>> tokenizer("Hello world")["input_ids"]
[15496, 995]

>>> tokenizer(" Hello world")["input_ids"]
[18435, 995]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/codegen/tokenization_codegen_fast.py#L186)

( token\_ids: typing.Union\[int, typing.List\[int\], ForwardRef('np.ndarray'), ForwardRef('torch.Tensor'), ForwardRef('tf.Tensor')\] skip\_special\_tokens: bool = False clean\_up\_tokenization\_spaces: bool = None truncate\_before\_pattern: typing.Optional\[typing.List\[str\]\] = None \*\*kwargs ) → `str`

Parameters

-   **token\_ids** (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`) — List of tokenized input ids. Can be obtained using the `__call__` method.
-   **skip\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not to remove special tokens in the decoding.
-   **clean\_up\_tokenization\_spaces** (`bool`, _optional_) — Whether or not to clean up the tokenization spaces. If `None`, will default to `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
-   **truncate\_before\_pattern** (`List[str]`, _optional_, defaults to `None`) — A list of regular expression strings that will be used to truncate the returned string. This can be used to remove extra pieces of code (e.g. truncate if observing a comment symbol ”#” at the beginning of a new line). An example pattern could be \`\[”^#”, re.escape(”<|endoftext|>”), ”^'''”, ”

The decoded sentence.

Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special tokens and clean up tokenization spaces.

Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

”\]\`. kwargs (additional keyword arguments, _optional_): Will be passed to the underlying model specific decode method.

## CodeGenModel

### class transformers.CodeGenModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/codegen/modeling_codegen.py#L410)

( config )

Parameters

-   **config** ([CodeGenConfig](/docs/transformers/v4.34.0/en/model_doc/codegen#transformers.CodeGenConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare CodeGen Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/codegen/modeling_codegen.py#L433)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using `AutoProcenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_attention_heads,)` or `(n_layer, num_attention_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_dim)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CodeGenConfig](/docs/transformers/v4.34.0/en/model_doc/codegen#transformers.CodeGenConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CodeGenModel](/docs/transformers/v4.34.0/en/model_doc/codegen#transformers.CodeGenModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, CodeGenModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
>>> model = CodeGenModel.from_pretrained("Salesforce/codegen-2B-mono")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## CodeGenForCausalLM

### class transformers.CodeGenForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/codegen/modeling_codegen.py#L604)

( config )

Parameters

-   **config** ([CodeGenConfig](/docs/transformers/v4.34.0/en/model_doc/codegen#transformers.CodeGenConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CodeGen Model transformer with a language modeling head on top.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/codegen/modeling_codegen.py#L648)

( input\_ids: typing.Optional\[torch.LongTensor\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using `AutoProcenizer`. See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_attention_heads,)` or `(n_layer, num_attention_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_dim)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

A [transformers.modeling\_outputs.CausalLMOutputWithPast](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([CodeGenConfig](/docs/transformers/v4.34.0/en/model_doc/codegen#transformers.CodeGenConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss (for next-token prediction).
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`)
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [CodeGenForCausalLM](/docs/transformers/v4.34.0/en/model_doc/codegen#transformers.CodeGenForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> import torch
>>> from transformers import AutoTokenizer, CodeGenForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
>>> model = CodeGenForCausalLM.from_pretrained("Salesforce/codegen-2B-mono")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> loss = outputs.loss
>>> logits = outputs.logits
```