# LED

## Overview

The LED model was proposed in [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, Arman Cohan.

The abstract from the paper is the following:

_Transformer-based models are unable to process long sequences due to their self-attention operation, which scales quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or longer. Longformer’s attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on WikiHop and TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization dataset._

Tips:

-   [LEDForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDForConditionalGeneration) is an extension of [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) exchanging the traditional _self-attention_ layer with _Longformer_’s _chunked self-attention_ layer. [LEDTokenizer](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDTokenizer) is an alias of [BartTokenizer](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartTokenizer).
-   LED works very well on long-range _sequence-to-sequence_ tasks where the `input_ids` largely exceed a length of 1024 tokens.
-   LED pads the `input_ids` to be a multiple of `config.attention_window` if required. Therefore a small speed-up is gained, when [LEDTokenizer](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDTokenizer) is used with the `pad_to_multiple_of` argument.
-   LED makes use of _global attention_ by means of the `global_attention_mask` (see [LongformerModel](/docs/transformers/v4.34.0/en/model_doc/longformer#transformers.LongformerModel)). For summarization, it is advised to put _global attention_ only on the first `<s>` token. For question answering, it is advised to put _global attention_ on all tokens of the question.
-   To fine-tune LED on all 16384, _gradient checkpointing_ can be enabled in case training leads to out-of-memory (OOM) errors. This can be done by executing `model.gradient_checkpointing_enable()`. Moreover, the `use_cache=False` flag can be used to disable the caching mechanism to save memory.
-   A notebook showing how to evaluate LED, can be accessed [here](https://colab.research.google.com/drive/12INTTR6n64TzS4RrXZxMSXfrOd9Xzamo?usp=sharing).
-   A notebook showing how to fine-tune LED, can be accessed [here](https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing).
-   LED is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.

This model was contributed by [patrickvonplaten](https://huggingface.co/patrickvonplaten).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## LEDConfig

### class transformers.LEDConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/configuration_led.py#L31)

( vocab\_size = 50265max\_encoder\_position\_embeddings = 16384max\_decoder\_position\_embeddings = 1024encoder\_layers = 12encoder\_ffn\_dim = 4096encoder\_attention\_heads = 16decoder\_layers = 12decoder\_ffn\_dim = 4096decoder\_attention\_heads = 16encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0use\_cache = Trueis\_encoder\_decoder = Trueactivation\_function = 'gelu'd\_model = 1024dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02decoder\_start\_token\_id = 2classifier\_dropout = 0.0pad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2attention\_window: typing.Union\[typing.List\[int\], int\] = 512\*\*kwargs )

This is the configuration class to store the configuration of a [LEDModel](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDModel). It is used to instantiate an LED model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the LED [allenai/led-base-16384](https://huggingface.co/allenai/led-base-16384) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import LEDModel, LEDConfig

>>> 
>>> configuration = LEDConfig()

>>> 
>>> model = LEDModel(configuration)

>>> 
>>> configuration = model.config
```

## LEDTokenizer

### class transformers.LEDTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/tokenization_led.py#L93)

( vocab\_filemerges\_fileerrors = 'replace'bos\_token = '<s>'eos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'add\_prefix\_space = False\*\*kwargs )

Constructs a LED tokenizer, which is smilar to the ROBERTa tokenizer, using byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import LEDTokenizer

>>> tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/tokenization_led.py#L344)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A LED sequence has the following format:

-   single sequence: `<s> X </s>`
-   pair of sequences: `<s> A </s></s> B </s>`

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/tokenization_led.py#L370)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/tokenization_led.py#L398)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. LED does not make use of token type ids, therefore a list of zeros is returned.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/tokenization_led.py#L314)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## LEDTokenizerFast

### class transformers.LEDTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/tokenization_led_fast.py#L50)

( vocab\_file = Nonemerges\_file = Nonetokenizer\_file = Noneerrors = 'replace'bos\_token = '<s>'eos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'add\_prefix\_space = Falsetrim\_offsets = True\*\*kwargs )

Construct a “fast” LED tokenizer (backed by HuggingFace’s _tokenizers_ library), derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import LEDTokenizerFast

>>> tokenizer = LEDTokenizerFast.from_pretrained("allenai/led-base-16384")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/tokenization_led_fast.py#L274)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. LED does not make use of token type ids, therefore a list of zeros is returned.

## LED specific outputs

### class transformers.models.led.modeling\_led.LEDEncoderBaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L1175)

( last\_hidden\_state: FloatTensorhidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneglobal\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for LEDEncoder’s outputs, with potential hidden states, local and global attentions.

### class transformers.models.led.modeling\_led.LEDSeq2SeqModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L1218)

( last\_hidden\_state: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for model encoder’s outputs that also contains : pre-computed hidden states that can speed up sequential decoding.

### class transformers.models.led.modeling\_led.LEDSeq2SeqLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L1286)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for sequence-to-sequence language models outputs.

### class transformers.models.led.modeling\_led.LEDSeq2SeqSequenceClassifierOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L1353)

( loss: typing.Optional\[torch.FloatTensor\] = Nonelogits: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of sequence-to-sequence sentence classification models.

### class transformers.models.led.modeling\_led.LEDSeq2SeqQuestionAnsweringModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L1420)

( loss: typing.Optional\[torch.FloatTensor\] = Nonestart\_logits: FloatTensor = Noneend\_logits: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_global\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Base class for outputs of sequence-to-sequence question answering models.

### class transformers.models.led.modeling\_tf\_led.TFLEDEncoderBaseModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_tf_led.py#L1340)

( last\_hidden\_state: tf.Tensor = Nonehidden\_states: Tuple\[tf.Tensor\] | None = Noneattentions: Tuple\[tf.Tensor\] | None = Noneglobal\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for Longformer’s outputs, with potential hidden states, local and global attentions.

### class transformers.models.led.modeling\_tf\_led.TFLEDSeq2SeqModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_tf_led.py#L1383)

( last\_hidden\_state: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonedecoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Nonedecoder\_attentions: Tuple\[tf.Tensor\] | None = Nonecross\_attentions: Tuple\[tf.Tensor\] | None = Noneencoder\_last\_hidden\_state: tf.Tensor | None = Noneencoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Noneencoder\_attentions: Tuple\[tf.Tensor\] | None = Noneencoder\_global\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for model encoder’s outputs that also contains : pre-computed hidden states that can speed up sequential decoding.

### class transformers.models.led.modeling\_tf\_led.TFLEDSeq2SeqLMOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_tf_led.py#L1451)

( loss: tf.Tensor | None = Nonelogits: tf.Tensor = Nonepast\_key\_values: List\[tf.Tensor\] | None = Nonedecoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Nonedecoder\_attentions: Tuple\[tf.Tensor\] | None = Nonecross\_attentions: Tuple\[tf.Tensor\] | None = Noneencoder\_last\_hidden\_state: tf.Tensor | None = Noneencoder\_hidden\_states: Tuple\[tf.Tensor\] | None = Noneencoder\_attentions: Tuple\[tf.Tensor\] | None = Noneencoder\_global\_attentions: Tuple\[tf.Tensor\] | None = None )

Base class for sequence-to-sequence language models outputs.

## LEDModel

### class transformers.LEDModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L2211)

( config: LEDConfig )

Parameters

-   **config** ([LEDConfig](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare LED Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). See the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L2240)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneglobal\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [LEDModel](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LEDModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> model = LEDModel.from_pretrained("allenai/led-base-16384")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## LEDForConditionalGeneration

### class transformers.LEDForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L2335)

( config: LEDConfig )

Parameters

-   **config** ([LEDConfig](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LED Model with a language modeling head. Can be used for summarization. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). See the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L2375)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneglobal\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [LEDForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Conditional generation example:

```
>>> from transformers import AutoTokenizer, LEDForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> TXT = "My friends are <mask> but they eat too many carbs."

>>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]

>>> prediction = model.generate(input_ids)[0]
>>> print(tokenizer.decode(prediction, skip_special_tokens=True))
```

Summarization example:

```
>>> import torch
>>> from transformers import AutoTokenizer, LEDForConditionalGeneration

>>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

>>> ARTICLE_TO_SUMMARIZE = '''Transformers (Vaswani et al., 2017) have achieved state-of-the-art
...     results in a wide range of natural language tasks including generative language modeling
...     (Dai et al., 2019; Radford et al., 2019) and discriminative ... language understanding (Devlin et al., 2019).
...     This success is partly due to the self-attention component which enables the network to capture contextual
...     information from the entire sequence. While powerful, the memory and computational requirements of
...     self-attention grow quadratically with sequence length, making it infeasible (or very expensive) to
...     process long sequences. To address this limitation, we present Longformer, a modified Transformer
...     architecture with a self-attention operation that scales linearly with the sequence length, making it
...     versatile for processing long documents (Fig 1). This is an advantage for natural language tasks such as
...     long document classification, question answering (QA), and coreference resolution, where existing approaches
...     partition or shorten the long context into smaller sequences that fall within the typical 512 token limit
...     of BERT-style pretrained models. Such partitioning could potentially result in loss of important
...     cross-partition information, and to mitigate this problem, existing methods often rely on complex
...     architectures to address such interactions. On the other hand, our proposed Longformer is able to build
...     contextual representations of the entire context using multiple layers of attention, reducing the need for
...     task-specific architectures.'''
>>> inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")

>>> 
>>> global_attention_mask = torch.zeros_like(inputs)
>>> global_attention_mask[:, 0] = 1

>>> 
>>> summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=32)
>>> print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
```

## LEDForSequenceClassification

### class transformers.LEDForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L2525)

( config: LEDConfig\*\*kwargs )

Parameters

-   **config** ([LEDConfig](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LED model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). See the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L2547)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneglobal\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [LEDForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, LEDForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> model = LEDForSequenceClassification.from_pretrained("allenai/led-base-16384")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = LEDForSequenceClassification.from_pretrained("allenai/led-base-16384", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, LEDForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> model = LEDForSequenceClassification.from_pretrained("allenai/led-base-16384", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = LEDForSequenceClassification.from_pretrained(
...     "allenai/led-base-16384", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## LEDForQuestionAnswering

### class transformers.LEDForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L2661)

( config )

Parameters

-   **config** ([LEDConfig](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LED Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). See the superclass documentation for the generic methods the library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_led.py#L2676)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneglobal\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The [LEDForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LEDForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> model = LEDForQuestionAnswering.from_pretrained("allenai/led-base-16384")

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

## TFLEDModel

### class transformers.TFLEDModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_tf_led.py#L2218)

( \*args\*\*kwargs )

Parameters

-   **config** ([LEDConfig](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare LED Model outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_tf_led.py#L2230)

( input\_ids: TFModelInputType | None = Noneattention\_mask: tf.Tensor | None = Nonedecoder\_input\_ids: tf.Tensor | None = Nonedecoder\_attention\_mask: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Nonedecoder\_head\_mask: tf.Tensor | None = Noneencoder\_outputs: tf.Tensor | None = Noneglobal\_attention\_mask: tf.Tensor | None = Nonepast\_key\_values: Tuple\[Tuple\[tf.Tensor\]\] | None = Noneinputs\_embeds: tf.Tensor | None = Nonedecoder\_inputs\_embeds: tf.Tensor | None = Noneuse\_cache: bool | None = Noneoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Nonereturn\_dict: bool | None = Nonetraining: bool = False\*\*kwargs ) → [transformers.models.led.modeling\_tf\_led.TFLEDSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/model_doc/led#transformers.models.led.modeling_tf_led.TFLEDSeq2SeqModelOutput) or `tuple(tf.Tensor)`

The [TFLEDModel](/docs/transformers/v4.34.0/en/model_doc/led#transformers.TFLEDModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFLEDModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
>>> model = TFLEDModel.from_pretrained("allenai/led-base-16384")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFLEDForConditionalGeneration

### class transformers.TFLEDForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_tf_led.py#L2322)

( \*args\*\*kwargs )

Parameters

-   **config** ([LEDConfig](/docs/transformers/v4.34.0/en/model_doc/led#transformers.LEDConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The LED Model with a language modeling head. Can be used for summarization. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/led/modeling_tf_led.py#L2363)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: TFLEDEncoderBaseModelOutput | None = Noneglobal\_attention\_mask: np.ndarray | tf.Tensor | None = Nonepast\_key\_values: Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\] | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: bool | None = Noneoutput\_attentions: bool | None = Noneoutput\_hidden\_states: bool | None = Nonereturn\_dict: bool | None = Nonelabels: tf.Tensor | None = Nonetraining: bool = False ) → [transformers.models.led.modeling\_tf\_led.TFLEDSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/model_doc/led#transformers.models.led.modeling_tf_led.TFLEDSeq2SeqLMOutput) or `tuple(tf.Tensor)`

The [TFLEDForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/led#transformers.TFLEDForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TFLEDForConditionalGeneration
>>> import tensorflow as tf

>>> mname = "allenai/led-base-16384"
>>> tokenizer = AutoTokenizer.from_pretrained(mname)
>>> TXT = "My friends are <mask> but they eat too many carbs."
>>> model = TFLEDForConditionalGeneration.from_pretrained(mname)
>>> batch = tokenizer([TXT], return_tensors="tf")
>>> logits = model(inputs=batch.input_ids).logits
>>> probs = tf.nn.softmax(logits[0])
>>> 
```