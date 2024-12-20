# BART

[![Models](https://img.shields.io/badge/All_model_pages-bart-blueviolet)](https://huggingface.co/models?filter=bart) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/bart-large-mnli)

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign @patrickvonplaten

## Overview

The Bart model was proposed in [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.

According to the abstract,

-   Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT).
-   The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme, where spans of text are replaced with a single mask token.
-   BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.

Tips:

-   BART is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
    
-   Sequence-to-sequence model with an encoder and a decoder. Encoder is fed a corrupted version of the tokens, decoder is fed the original tokens (but has a mask to hide the future words like a regular transformers decoder). A composition of the following transformations are applied on the pretraining tasks for the encoder:
    
    -   mask random tokens (like in BERT)
    -   delete random tokens
    -   mask a span of k tokens with a single mask token (a span of 0 tokens is an insertion of a mask token)
    -   permute sentences
    -   rotate the document to make it start at a specific token

This model was contributed by [sshleifer](https://huggingface.co/sshleifer). The Authors’ code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/bart).

### Examples

-   Examples and scripts for fine-tuning BART and other models for sequence to sequence tasks can be found in [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md).
-   An example of how to train [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) with a Hugging Face `datasets` object can be found in this [forum discussion](https://discuss.huggingface.co/t/train-bart-for-conditional-generation-e-g-summarization/1904).
-   [Distilled checkpoints](https://huggingface.co/models?search=distilbart) are described in this [paper](https://arxiv.org/abs/2010.13002).

## Implementation Notes

-   Bart doesn’t use `token_type_ids` for sequence classification. Use [BartTokenizer](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartTokenizer) or [encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) to get the proper splitting.
-   The forward pass of [BartModel](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartModel) will create the `decoder_input_ids` if they are not passed. This is different than some other modeling APIs. A typical use case of this feature is mask filling.
-   Model predictions are intended to be identical to the original implementation when `forced_bos_token_id=0`. This only works, however, if the string you pass to `fairseq.encode` starts with a space.
-   [generate()](/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate) should be used for conditional generation tasks like summarization, see the example in that docstrings.
-   Models that load the _facebook/bart-large-cnn_ weights will not have a `mask_token_id`, or be able to perform mask-filling tasks.

## Mask Filling

The `facebook/bart-base` and `facebook/bart-large` checkpoints can be used to fill multi-token masks.

```
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
tok = BartTokenizer.from_pretrained("facebook/bart-large")
example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
batch = tok(example_english_phrase, return_tensors="pt")
generated_ids = model.generate(batch["input_ids"])
assert tok.batch_decode(generated_ids, skip_special_tokens=True) == [
    "UN Chief Says There Is No Plan to Stop Chemical Weapons in Syria"
]
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BART. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

-   A blog post on [Distributed Training: Train BART/T5 for Summarization using 🤗 Transformers and Amazon SageMaker](https://huggingface.co/blog/sagemaker-distributed-training-seq2seq).
-   A notebook on how to [finetune BART for summarization with fastai using blurr](https://colab.research.google.com/github/ohmeow/ohmeow_website/blob/master/posts/2021-05-25-mbart-sequence-classification-with-blurr.ipynb). 🌎
-   A notebook on how to [finetune BART for summarization in two languages with Trainer class](https://colab.research.google.com/github/elsanns/xai-nlp-notebooks/blob/master/fine_tune_bart_summarization_two_langs.ipynb). 🌎
-   [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization.ipynb).
-   [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb).
-   [FlaxBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.FlaxBartForConditionalGeneration) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/summarization).
-   [Summarization](https://huggingface.co/course/chapter7/5?fw=pt#summarization) chapter of the 🤗 Hugging Face course.
-   [Summarization task guide](../tasks/summarization)

-   [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling#robertabertdistilbert-and-masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb).
-   [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/language-modeling#run_mlmpy) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).
-   [FlaxBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.FlaxBartForConditionalGeneration) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#masked-language-modeling) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/masked_language_modeling_flax.ipynb).
-   [Masked language modeling](https://huggingface.co/course/chapter7/3?fw=pt) chapter of the 🤗 Hugging Face Course.
-   [Masked language modeling task guide](../tasks/masked_language_modeling)

-   A notebook on how to [finetune mBART using Seq2SeqTrainer for Hindi to English translation](https://colab.research.google.com/github/vasudevgupta7/huggingface-tutorials/blob/main/translation_training.ipynb). 🌎
-   [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb).
-   [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/translation) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb).
-   [Translation task guide](../tasks/translation)

See also:

-   [Text classification task guide](../tasks/sequence_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Causal language modeling task guide](../tasks/language_modeling)

## BartConfig

### class transformers.BartConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/configuration_bart.py#L35)

( vocab\_size = 50265max\_position\_embeddings = 1024encoder\_layers = 12encoder\_ffn\_dim = 4096encoder\_attention\_heads = 16decoder\_layers = 12decoder\_ffn\_dim = 4096decoder\_attention\_heads = 16encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0activation\_function = 'gelu'd\_model = 1024dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02classifier\_dropout = 0.0scale\_embedding = Falseuse\_cache = Truenum\_labels = 3pad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2is\_encoder\_decoder = Truedecoder\_start\_token\_id = 2forced\_eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [BartModel](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartModel). It is used to instantiate a BART model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BART [facebook/bart-large](https://huggingface.co/facebook/bart-large) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BartConfig, BartModel

>>> 
>>> configuration = BartConfig()

>>> 
>>> model = BartModel(configuration)

>>> 
>>> configuration = model.config
```

## BartTokenizer

### class transformers.BartTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/tokenization_bart.py#L101)

( vocab\_filemerges\_fileerrors = 'replace'bos\_token = '<s>'eos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'add\_prefix\_space = False\*\*kwargs )

Constructs a BART tokenizer, which is smilar to the ROBERTa tokenizer, using byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import BartTokenizer

>>> tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/tokenization_bart.py#L342)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BART sequence has the following format:

-   single sequence: `<s> X </s>`
-   pair of sequences: `<s> A </s></s> B </s>`

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/tokenization_bart.py#L394)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. BART does not make use of token type ids, therefore a list of zeros is returned.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/tokenization_bart.py#L367)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## BartTokenizerFast

### class transformers.BartTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/tokenization_bart_fast.py#L70)

( vocab\_file = Nonemerges\_file = Nonetokenizer\_file = Noneerrors = 'replace'bos\_token = '<s>'eos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'add\_prefix\_space = Falsetrim\_offsets = True\*\*kwargs )

Construct a “fast” BART tokenizer (backed by HuggingFace’s _tokenizers_ library), derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import BartTokenizerFast

>>> tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer needs to be instantiated with `add_prefix_space=True`.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/tokenization_bart_fast.py#L286)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. BART does not make use of token type ids, therefore a list of zeros is returned.

## BartModel

### class transformers.BartModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1180)

( config: BartConfig )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare BART Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1209)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [BartModel](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BartModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
>>> model = BartModel.from_pretrained("facebook/bart-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## BartForConditionalGeneration

### class transformers.BartForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1307)

( config: BartConfig )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The BART Model with a language modeling head. Can be used for summarization. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1347)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```
>>> from transformers import AutoTokenizer, BartForConditionalGeneration

>>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> ARTICLE_TO_SUMMARIZE = (
...     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
...     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
...     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
... )
>>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

>>> 
>>> summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
>>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'PG&E scheduled the blackouts in response to forecasts for high winds amid dry conditions'
```

Mask filling example:

```
>>> from transformers import AutoTokenizer, BartForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
>>> model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

>>> TXT = "My friends are <mask> but they eat too many carbs."
>>> input_ids = tokenizer([TXT], return_tensors="pt")["input_ids"]
>>> logits = model(input_ids).logits

>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)

>>> tokenizer.decode(predictions).split()
['not', 'good', 'healthy', 'great', 'very']
```

## BartForSequenceClassification

### class transformers.BartForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1483)

( config: BartConfig\*\*kwargs )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1499)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [BartForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, BartForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-sst2")
>>> model = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'POSITIVE'

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
0.0
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, BartForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-sst2")
>>> model = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = BartForSequenceClassification.from_pretrained(
...     "valhalla/bart-large-sst2", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## BartForQuestionAnswering

### class transformers.BartForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1613)

( config )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1628)

( input\_ids: Tensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The [BartForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BartForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-finetuned-squadv1")
>>> model = BartForQuestionAnswering.from_pretrained("valhalla/bart-large-finetuned-squadv1")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

>>> inputs = tokenizer(question, text, return_tensors="pt")
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()

>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
' nice puppet'

>>> 
>>> target_start_index = torch.tensor([14])
>>> target_end_index = torch.tensor([15])

>>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
>>> loss = outputs.loss
>>> round(loss.item(), 2)
0.59
```

## BartForCausalLM

### class transformers.BartForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1751)

( config )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BART decoder with with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_bart.py#L1784)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, BartForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
>>> model = BartForCausalLM.from_pretrained("facebook/bart-base", add_cross_attention=False)
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
>>> list(logits.shape) == expected_shape
True
```

## TFBartModel

### class transformers.TFBartModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_tf_bart.py#L1151)

( \*args\*\*kwargs )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare BART Model outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_tf_bart.py#L1165)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_position\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Nonecross\_attn\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: Optional\[Union\[Tuple, TFBaseModelOutput\]\] = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False\*\*kwargs ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput) or `tuple(tf.Tensor)`

The [TFBartModel](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFBartModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
>>> model = TFBartModel.from_pretrained("facebook/bart-large")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFBartForConditionalGeneration

### class transformers.TFBartForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_tf_bart.py#L1256)

( \*args\*\*kwargs )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The BART Model with a language modeling head. Can be used for summarization. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_tf_bart.py#L1292)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_position\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Nonecross\_attn\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: Optional\[TFBaseModelOutput\] = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or `tuple(tf.Tensor)`

The [TFBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```
>>> from transformers import AutoTokenizer, TFBartForConditionalGeneration

>>> model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

>>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="tf")

>>> 
>>> summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=5)
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
```

Mask filling example:

```
>>> from transformers import AutoTokenizer, TFBartForConditionalGeneration

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
>>> TXT = "My friends are <mask> but they eat too many carbs."

>>> model = TFBartForConditionalGeneration.from_pretrained("facebook/bart-large")
>>> input_ids = tokenizer([TXT], return_tensors="tf")["input_ids"]
>>> logits = model(input_ids).logits
>>> probs = tf.nn.softmax(logits[0])
>>> 
```

## TFBartForSequenceClassification

### class transformers.TFBartForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_tf_bart.py#L1445)

( \*args\*\*kwargs )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_tf_bart.py#L1453)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_input\_ids: np.ndarray | tf.Tensor | None = Nonedecoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_position\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Nonedecoder\_head\_mask: np.ndarray | tf.Tensor | None = Nonecross\_attn\_head\_mask: np.ndarray | tf.Tensor | None = Noneencoder\_outputs: Optional\[TFBaseModelOutput\] = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonedecoder\_inputs\_embeds: np.ndarray | tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqSequenceClassifierOutput) or `tuple(tf.Tensor)`

The [TFBartForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.TFBartForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## FlaxBartModel

### class transformers.FlaxBartModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1241)

( config: BartConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare Bart Model transformer outputting raw hidden-states without any specific head on top. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1176)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The `FlaxBartPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBartModel

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
>>> model = FlaxBartModel.from_pretrained("facebook/bart-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L999)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1062)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> last_decoder_hidden_states = outputs.last_hidden_state
```

## FlaxBartForConditionalGeneration

### class transformers.FlaxBartForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1325)

( config: BartConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The BART Model with a language modeling head. Can be used for summarization. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1176)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The `FlaxBartPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="np")

>>> 
>>> summary_ids = model.generate(inputs["input_ids"]).sequences
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
```

Mask filling example:

```
>>> import jax
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

>>> TXT = "My friends are <mask> but they eat too many carbs."
>>> input_ids = tokenizer([TXT], return_tensors="jax")["input_ids"]

>>> logits = model(input_ids).logits
>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero()[0].item()
>>> probs = jax.nn.softmax(logits[0, masked_index], axis=0)
>>> values, predictions = jax.lax.top_k(probs, k=1)

>>> tokenizer.decode(predictions).split()
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L999)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1329)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> logits = outputs.logits
```

## FlaxBartForSequenceClassification

### class transformers.FlaxBartForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1638)

( config: BartConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1176)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The `FlaxBartPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBartForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
>>> model = FlaxBartForSequenceClassification.from_pretrained("facebook/bart-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L999)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1062)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> last_decoder_hidden_states = outputs.last_hidden_state
```

## FlaxBartForQuestionAnswering

### class transformers.FlaxBartForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1724)

( config: BartConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

BART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1176)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The `FlaxBartPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBartForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
>>> model = FlaxBartForQuestionAnswering.from_pretrained("facebook/bart-base")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
>>> inputs = tokenizer(question, text, return_tensors="jax")

>>> outputs = model(**inputs)
>>> start_scores = outputs.start_logits
>>> end_scores = outputs.end_logits
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L999)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1062)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> import jax.numpy as jnp
>>> from transformers import AutoTokenizer, FlaxBartForConditionalGeneration

>>> model = FlaxBartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> last_decoder_hidden_states = outputs.last_hidden_state
```

## FlaxBartForCausalLM

### class transformers.FlaxBartForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1960)

( config: BartConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BartConfig](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Bart Decoder Model with a language modeling head on top (linear layer with weights tied to the input embeddings) e.g for autoregressive tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bart/modeling_flax_bart.py#L1798)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneencoder\_hidden\_states: typing.Optional\[jax.Array\] = Noneencoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonepast\_key\_values: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

The `FlaxBartDecoderPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxBartForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
>>> model = FlaxBartForCausalLM.from_pretrained("facebook/bart-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
>>> outputs = model(**inputs)

>>> 
>>> next_token_logits = outputs.logits[:, -1]
```