# MBart and MBart-50

[![Models](https://img.shields.io/badge/All_model_pages-mbart-blueviolet)](https://huggingface.co/models?filter=mbart) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/mbart-large-50-one-to-many-mmt)

**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign @patrickvonplaten

## Overview of MBart

The MBart model was presented in [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.

According to the abstract, MBART is a sequence-to-sequence denoising auto-encoder pretrained on large-scale monolingual corpora in many languages using the BART objective. mBART is one of the first methods for pretraining a complete sequence-to-sequence model by denoising full texts in multiple languages, while previous approaches have focused only on the encoder, decoder, or reconstructing parts of the text.

This model was contributed by [valhalla](https://huggingface.co/valhalla). The Authors’ code can be found [here](https://github.com/pytorch/fairseq/tree/master/examples/mbart)

### Training of MBart

MBart is a multilingual encoder-decoder (sequence-to-sequence) model primarily intended for translation task. As the model is multilingual it expects the sequences in a different format. A special language id token is added in both the source and target text. The source text format is `X [eos, src_lang_code]` where `X` is the source text. The target text format is `[tgt_lang_code] X [eos]`. `bos` is never used.

The regular [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) will encode source text format passed as first argument or with the `text` keyword, and target text format passed with the `text_label` keyword argument.

-   Supervised training

```
>>> from transformers import MBartForConditionalGeneration, MBartTokenizer

>>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
>>> example_english_phrase = "UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"

>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")

>>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
>>> 
>>> model(**inputs)
```

-   Generation
    
    While generating the target text set the `decoder_start_token_id` to the target language id. The following example shows how to translate English to Romanian using the _facebook/mbart-large-en-ro_ model.
    

```
>>> from transformers import MBartForConditionalGeneration, MBartTokenizer

>>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX")
>>> article = "UN Chief Says There Is No Military Solution in Syria"
>>> inputs = tokenizer(article, return_tensors="pt")
>>> translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["ro_RO"])
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Şeful ONU declară că nu există o soluţie militară în Siria"
```

## Overview of MBart-50

MBart-50 was introduced in the [Multilingual Translation with Extensible Multilingual Pretraining and Finetuning](https://arxiv.org/abs/2008.00401) paper by Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, Angela Fan. MBart-50 is created using the original _mbart-large-cc25_ checkpoint by extendeding its embedding layers with randomly initialized vectors for an extra set of 25 language tokens and then pretrained on 50 languages.

According to the abstract

_Multilingual translation models can be created through multilingual finetuning. Instead of finetuning on one direction, a pretrained model is finetuned on many directions at the same time. It demonstrates that pretrained models can be extended to incorporate additional languages without loss of performance. Multilingual finetuning improves on average 1 BLEU over the strongest baselines (being either multilingual from scratch or bilingual finetuning) while improving 9.3 BLEU on average over bilingual baselines from scratch._

### Training of MBart-50

The text format for MBart-50 is slightly different from mBART. For MBart-50 the language id token is used as a prefix for both source and target text i.e the text format is `[lang_code] X [eos]`, where `lang_code` is source language id for source text and target language id for target text, with `X` being the source or target text respectively.

MBart-50 has its own tokenizer [MBart50Tokenizer](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBart50Tokenizer).

-   Supervised training

```
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")

src_text = " UN Chief Says There Is No Military Solution in Syria"
tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"

model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")

model(**model_inputs)  
```

-   Generation
    
    To generate using the mBART-50 multilingual translation models, `eos_token_id` is used as the `decoder_start_token_id` and the target language id is forced as the first generated token. To force the target language id as the first generated token, pass the _forced\_bos\_token\_id_ parameter to the _generate_ method. The following example shows how to translate between Hindi to French and Arabic to English using the _facebook/mbart-50-large-many-to-many_ checkpoint.
    

```
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


tokenizer.src_lang = "hi_IN"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)



tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar, return_tensors="pt")
generated_tokens = model.generate(**encoded_ar, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

```

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)
-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## MBartConfig

### class transformers.MBartConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/configuration_mbart.py#L34)

( vocab\_size = 50265max\_position\_embeddings = 1024encoder\_layers = 12encoder\_ffn\_dim = 4096encoder\_attention\_heads = 16decoder\_layers = 12decoder\_ffn\_dim = 4096decoder\_attention\_heads = 16encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0use\_cache = Trueis\_encoder\_decoder = Trueactivation\_function = 'gelu'd\_model = 1024dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02classifier\_dropout = 0.0scale\_embedding = Falsepad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2forced\_eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [MBartModel](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartModel). It is used to instantiate an MBART model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the MBART [facebook/mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import MBartConfig, MBartModel

>>> 
>>> configuration = MBartConfig()

>>> 
>>> model = MBartModel(configuration)

>>> 
>>> configuration = model.config
```

## MBartTokenizer

### class transformers.MBartTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/tokenization_mbart.py#L53)

( vocab\_filebos\_token = '<s>'eos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'tokenizer\_file = Nonesrc\_lang = Nonetgt\_lang = Nonesp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = Noneadditional\_special\_tokens = None\*\*kwargs )

Construct an MBART tokenizer.

Adapted from [RobertaTokenizer](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaTokenizer) and [XLNetTokenizer](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetTokenizer). Based on [SentencePiece](https://github.com/google/sentencepiece).

The tokenization method is `<tokens> <eos> <language code>` for source language documents, and \`<language code>

<tokens> <eos>\` for target language documents.

Examples:

```
>>> from transformers import MBartTokenizer

>>> tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO")
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/tokenization_mbart.py#L217)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. An MBART sequence has the following format, where `X` represents the sequence:

-   `input_ids` (for encoder) `X [eos, src_lang_code]`
-   `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a separator.

## MBartTokenizerFast

### class transformers.MBartTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/tokenization_mbart_fast.py#L63)

( vocab\_file = Nonetokenizer\_file = Nonebos\_token = '<s>'eos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'src\_lang = Nonetgt\_lang = Noneadditional\_special\_tokens = None\*\*kwargs )

Construct a “fast” MBART tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

The tokenization method is `<tokens> <eos> <language code>` for source language documents, and \`<language code>

<tokens> <eos>\` for target language documents.

Examples:

```
>>> from transformers import MBartTokenizerFast

>>> tokenizer = MBartTokenizerFast.from_pretrained(
...     "facebook/mbart-large-en-ro", src_lang="en_XX", tgt_lang="ro_RO"
... )
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_romanian = "Şeful ONU declară că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_romanian, return_tensors="pt")
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/tokenization_mbart_fast.py#L162)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. The special tokens depend on calling set\_lang.

An MBART sequence has the following format, where `X` represents the sequence:

-   `input_ids` (for encoder) `X [eos, src_lang_code]`
-   `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a separator.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/tokenization_mbart_fast.py#L191)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. mBART does not make use of token type ids, therefore a list of zeros is returned.

#### set\_src\_lang\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/tokenization_mbart_fast.py#L246)

( src\_lang )

Reset the special tokens to the source lang setting. No prefix and suffix=\[eos, src\_lang\_code\].

#### set\_tgt\_lang\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/tokenization_mbart_fast.py#L261)

( lang: str )

Reset the special tokens to the target language setting. No prefix and suffix=\[eos, tgt\_lang\_code\].

## MBart50Tokenizer

### class transformers.MBart50Tokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart50/tokenization_mbart50.py#L49)

( vocab\_filesrc\_lang = Nonetgt\_lang = Noneeos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None\*\*kwargs )

Construct a MBart50 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

Examples:

```
>>> from transformers import MBart50Tokenizer

>>> tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
>>> src_text = " UN Chief Says There Is No Military Solution in Syria"
>>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
>>> model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
>>> 
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart50/tokenization_mbart50.py#L301)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. An MBART-50 sequence has the following format, where `X` represents the sequence:

-   `input_ids` (for encoder) `[src_lang_code] X [eos]`
-   `labels`: (for decoder) `[tgt_lang_code] X [eos]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a separator.

Converts a sequence of tokens (string) in a single string.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart50/tokenization_mbart50.py#L271)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = Nonealready\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

#### set\_src\_lang\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart50/tokenization_mbart50.py#L358)

( src\_lang: str )

Reset the special tokens to the source lang setting. prefix=\[src\_lang\_code\] and suffix=\[eos\].

#### set\_tgt\_lang\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart50/tokenization_mbart50.py#L364)

( tgt\_lang: str )

Reset the special tokens to the target language setting. prefix=\[tgt\_lang\_code\] and suffix=\[eos\].

## MBart50TokenizerFast

### class transformers.MBart50TokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart50/tokenization_mbart50_fast.py#L59)

( vocab\_file = Nonesrc\_lang = Nonetgt\_lang = Nonetokenizer\_file = Noneeos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'\*\*kwargs )

Construct a “fast” MBART tokenizer for mBART-50 (backed by HuggingFace’s _tokenizers_ library). Based on [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

Examples:

```
>>> from transformers import MBart50TokenizerFast

>>> tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
>>> src_text = " UN Chief Says There Is No Military Solution in Syria"
>>> tgt_text = "Şeful ONU declară că nu există o soluţie militară în Siria"
>>> model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
>>> 
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart50/tokenization_mbart50_fast.py#L173)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. The special tokens depend on calling set\_lang.

An MBART-50 sequence has the following format, where `X` represents the sequence:

-   `input_ids` (for encoder) `[src_lang_code] X [eos]`
-   `labels`: (for decoder) `[tgt_lang_code] X [eos]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a separator.

#### set\_src\_lang\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart50/tokenization_mbart50_fast.py#L220)

( src\_lang: str )

Reset the special tokens to the source lang setting. prefix=\[src\_lang\_code\] and suffix=\[eos\].

#### set\_tgt\_lang\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart50/tokenization_mbart50_fast.py#L235)

( tgt\_lang: str )

Reset the special tokens to the target language setting. prefix=\[src\_lang\_code\] and suffix=\[eos\].

## MBartModel

### class transformers.MBartModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1158)

( config: MBartConfig )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare MBART Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1192)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [MBartModel](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, MBartModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
>>> model = MBartModel.from_pretrained("facebook/mbart-large-cc25")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## MBartForConditionalGeneration

### class transformers.MBartForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1282)

( config: MBartConfig )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MBART Model with a language modeling head. Can be used for summarization, after fine-tuning the pretrained models. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1322)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [MBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Translation example:

```
>>> from transformers import AutoTokenizer, MBartForConditionalGeneration

>>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-en-ro")

>>> example_english_phrase = "42 is the answer"
>>> inputs = tokenizer(example_english_phrase, return_tensors="pt")

>>> 
>>> generated_ids = model.generate(**inputs, num_beams=4, max_length=5)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'42 este răspuns'
```

Mask filling example:

```
>>> from transformers import AutoTokenizer, MBartForConditionalGeneration

>>> model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> 
>>> TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"

>>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="pt")["input_ids"]
>>> logits = model(input_ids).logits

>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)

>>> tokenizer.decode(predictions).split()
['nett', 'sehr', 'ganz', 'nicht', 'so']
```

## MBartForQuestionAnswering

### class transformers.MBartForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1581)

( config )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MBART Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1596)

( input\_ids: Tensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The [MBartForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, MBartForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
>>> model = MBartForQuestionAnswering.from_pretrained("facebook/mbart-large-cc25")

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

## MBartForSequenceClassification

### class transformers.MBartForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1452)

( config: MBartConfig\*\*kwargs )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1468)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [MBartForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, MBartForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
>>> model = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-cc25")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-cc25", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, MBartForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
>>> model = MBartForSequenceClassification.from_pretrained("facebook/mbart-large-cc25", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = MBartForSequenceClassification.from_pretrained(
...     "facebook/mbart-large-cc25", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## MBartForCausalLM

### class transformers.MBartForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1714)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_mbart.py#L1747)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, MBartForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
>>> model = MBartForCausalLM.from_pretrained("facebook/mbart-large-cc25", add_cross_attention=False)
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
>>> list(logits.shape) == expected_shape
True
```

## TFMBartModel

### class transformers.TFMBartModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_tf_mbart.py#L1162)

( \*args\*\*kwargs )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare MBART Model outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_tf_mbart.py#L1174)

( input\_ids: TFModelInputType = Noneattention\_mask: tf.Tensor | None = Nonedecoder\_input\_ids: tf.Tensor | None = Nonedecoder\_attention\_mask: tf.Tensor | None = Nonedecoder\_position\_ids: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Nonedecoder\_head\_mask: tf.Tensor | None = Nonecross\_attn\_head\_mask: tf.Tensor | None = Noneencoder\_outputs: Optional\[Union\[Tuple, TFBaseModelOutput\]\] = Nonepast\_key\_values: Tuple\[Tuple\[tf.Tensor\]\] | None = Noneinputs\_embeds: tf.Tensor | None = Nonedecoder\_inputs\_embeds: tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False\*\*kwargs ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput) or `tuple(tf.Tensor)`

The [TFMBartModel](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.TFMBartModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFMBartModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
>>> model = TFMBartModel.from_pretrained("facebook/mbart-large-cc25")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFMBartForConditionalGeneration

### class transformers.TFMBartForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_tf_mbart.py#L1267)

( \*args\*\*kwargs )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The MBART Model with a language modeling head. Can be used for summarization, after fine-tuning the pretrained models. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_tf_mbart.py#L1305)

( input\_ids: TFModelInputType = Noneattention\_mask: tf.Tensor | None = Nonedecoder\_input\_ids: tf.Tensor | None = Nonedecoder\_attention\_mask: tf.Tensor | None = Nonedecoder\_position\_ids: tf.Tensor | None = Nonehead\_mask: tf.Tensor | None = Nonedecoder\_head\_mask: tf.Tensor | None = Nonecross\_attn\_head\_mask: tf.Tensor | None = Noneencoder\_outputs: Optional\[TFBaseModelOutput\] = Nonepast\_key\_values: Tuple\[Tuple\[tf.Tensor\]\] = Noneinputs\_embeds: tf.Tensor | None = Nonedecoder\_inputs\_embeds: tf.Tensor | None = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or `tuple(tf.Tensor)`

The [TFMBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.TFMBartForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Translation example:

```
>>> from transformers import AutoTokenizer, TFMBartForConditionalGeneration

>>> model = TFMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-en-ro")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-en-ro")

>>> example_english_phrase = "42 is the answer"
>>> inputs = tokenizer(example_english_phrase, return_tensors="tf")

>>> 
>>> generated_ids = model.generate(**inputs, num_beams=4, max_length=5)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'42 este răspuns'
```

Mask filling example:

```
>>> from transformers import AutoTokenizer, TFMBartForConditionalGeneration
>>> import tensorflow as tf

>>> model = TFMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> 
>>> TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"

>>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="tf")["input_ids"]
>>> logits = model(input_ids).logits

>>> masked_index = tf.where(input_ids[0] == tokenizer.mask_token_id)[0, 0]
>>> probs = tf.nn.softmax(logits[0, masked_index], axis=0)
>>> values, predictions = tf.math.top_k(probs, 5)

>>> tokenizer.decode(predictions).split()
['nett', 'sehr', 'ganz', 'nicht', 'so']
```

## FlaxMBartModel

### class transformers.FlaxMBartModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1276)

( config: MBartConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare MBart Model transformer outputting raw hidden-states without any specific head on top. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1213)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The `FlaxMBartPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartModel

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
>>> model = FlaxMBartModel.from_pretrained("facebook/mbart-large-cc25")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1037)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1100)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> last_decoder_hidden_states = outputs.last_hidden_state
```

## FlaxMBartForConditionalGeneration

### class transformers.FlaxMBartForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1361)

( config: MBartConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The MMBart Model with a language modeling head. Can be used for summarization. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1213)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The `FlaxMBartPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration, MBartConfig

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> ARTICLE_TO_SUMMARIZE = "Meine Freunde sind cool, aber sie essen zu viel Kuchen."
>>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="np")

>>> 
>>> summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=5).sequences
>>> print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
```

Mask filling example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> 
>>> TXT = "</s> Meine Freunde sind <mask> nett aber sie essen zu viel Kuchen. </s> de_DE"
>>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="np")["input_ids"]

>>> logits = model(input_ids).logits
>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero()[0].item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)

>>> tokenizer.decode(predictions).split()
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1037)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1365)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxCausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> logits = outputs.logits
```

## FlaxMBartForSequenceClassification

### class transformers.FlaxMBartForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1674)

( config: MBartConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

MBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1213)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The `FlaxMBartPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
>>> model = FlaxMBartForSequenceClassification.from_pretrained("facebook/mbart-large-cc25")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1037)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1100)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> last_decoder_hidden_states = outputs.last_hidden_state
```

## FlaxMBartForQuestionAnswering

### class transformers.FlaxMBartForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1761)

( config: MBartConfiginput\_shape: typing.Tuple\[int\] = (1, 1)seed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([MBartConfig](/docs/transformers/v4.34.0/en/model_doc/mbart#transformers.MBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

MBart Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1213)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_input\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The `FlaxMBartPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
>>> model = FlaxMBartForQuestionAnswering.from_pretrained("facebook/mbart-large-cc25")

>>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
>>> inputs = tokenizer(question, text, return_tensors="jax")

>>> outputs = model(**inputs)
>>> start_scores = outputs.start_logits
>>> end_scores = outputs.end_logits
```

#### encode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1037)

( input\_ids: Arrayattention\_mask: typing.Optional\[jax.Array\] = Noneposition\_ids: typing.Optional\[jax.Array\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutput) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)
```

#### decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mbart/modeling_flax_mbart.py#L1100)

( decoder\_input\_idsencoder\_outputsencoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_attention\_mask: typing.Optional\[jax.Array\] = Nonedecoder\_position\_ids: typing.Optional\[jax.Array\] = Nonepast\_key\_values: dict = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonetrain: bool = Falseparams: dict = Nonedropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, FlaxMBartForConditionalGeneration

>>> model = FlaxMBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

>>> text = "My friends are cool but they eat too many carbs."
>>> inputs = tokenizer(text, max_length=1024, return_tensors="jax")
>>> encoder_outputs = model.encode(**inputs)

>>> decoder_start_token_id = model.config.decoder_start_token_id
>>> decoder_input_ids = jnp.ones((inputs.input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id

>>> outputs = model.decode(decoder_input_ids, encoder_outputs)
>>> last_decoder_hidden_states = outputs.last_hidden_state
```