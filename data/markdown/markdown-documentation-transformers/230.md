**DISCLAIMER:** The default behaviour for the tokenizer has recently been fixed (and thus changed)!

The previous version adds `[self.eos_token_id, self.cur_lang_code]` at the end of the token sequence for both target and source tokenization. This is wrong as the NLLB paper mentions (page 48, 6.1.1. Model Architecture) :

_Note that we prefix the source sequence with the source language, as opposed to the target language as previously done in several works (Arivazhagan et al., 2019; Johnson et al., 2017). This is primarily because we prioritize optimizing zero-shot performance of our model on any pair of 200 languages at a minor cost to supervised performance._

Previous behaviour:

```
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
>>> tokenizer("How was your day?").input_ids
[13374, 1398, 4260, 4039, 248130, 2, 256047]

>>> 
>>> 
```

New behaviour

```
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
>>> tokenizer("How was your day?").input_ids
[256047, 13374, 1398, 4260, 4039, 248130, 2]
```

Enabling the old behaviour can be done as follows:

```
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", legacy_behaviour=True)
```

For more details, feel free to check the linked [PR](https://github.com/huggingface/transformers/pull/22313) and [Issue](https://github.com/huggingface/transformers/issues/19943).

# Overview of NLLB

The NLLB model was presented in [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) by Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang.

The abstract of the paper is the following:

_Driven by the goal of eradicating language barriers on a global scale, machine translation has solidified itself as a key focus of artificial intelligence research today. However, such efforts have coalesced around a small subset of languages, leaving behind the vast majority of mostly low-resource languages. What does it take to break the 200 language barrier while ensuring safe, high quality results, all while keeping ethical considerations in mind? In No Language Left Behind, we took on this challenge by first contextualizing the need for low-resource language translation support through exploratory interviews with native speakers. Then, we created datasets and models aimed at narrowing the performance gap between low and high-resource languages. More specifically, we developed a conditional compute model based on Sparsely Gated Mixture of Experts that is trained on data obtained with novel and effective data mining techniques tailored for low-resource languages. We propose multiple architectural and training improvements to counteract overfitting while training on thousands of tasks. Critically, we evaluated the performance of over 40,000 different translation directions using a human-translated benchmark, Flores-200, and combined human evaluation with a novel toxicity benchmark covering all languages in Flores-200 to assess translation safety. Our model achieves an improvement of 44% BLEU relative to the previous state-of-the-art, laying important groundwork towards realizing a universal translation system._

This implementation contains the dense models available on release.

**The sparse model NLLB-MoE (Mixture of Expert) is now available! More details [here](nllb-moe)**

This model was contributed by [Lysandre](https://huggingface.co/lysandre). The authors’ code can be found [here](https://github.com/facebookresearch/fairseq/tree/nllb).

## Generating with NLLB

While generating the target text set the `forced_bos_token_id` to the target language id. The following example shows how to translate English to French using the _facebook/nllb-200-distilled-600M_ model.

Note that we’re using the BCP-47 code for French `fra_Latn`. See [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) for the list of all BCP-47 in the Flores 200 dataset.

```
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

>>> article = "UN Chief says there is no military solution in Syria"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
Le chef de l'ONU dit qu'il n'y a pas de solution militaire en Syrie
```

### Generating from any other language than English

English (`eng_Latn`) is set as the default language from which to translate. In order to specify that you’d like to translate from a different language, you should specify the BCP-47 code in the `src_lang` keyword argument of the tokenizer initialization.

See example below for a translation from romanian to german:

```
>>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained(
...     "facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="ron_Latn"
... )
>>> model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True)

>>> article = "Şeful ONU spune că nu există o soluţie militară în Siria"
>>> inputs = tokenizer(article, return_tensors="pt")

>>> translated_tokens = model.generate(
...     **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["deu_Latn"], max_length=30
... )
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
UN-Chef sagt, es gibt keine militärische Lösung in Syrien
```

## Documentation resources

-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## NllbTokenizer

### class transformers.NllbTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nllb/tokenization_nllb.py#L49)

( vocab\_file bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' tokenizer\_file = None src\_lang = None tgt\_lang = None sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None additional\_special\_tokens = None legacy\_behaviour = False \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.
    
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **sep\_token** (`str`, _optional_, defaults to `"</s>"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **cls\_token** (`str`, _optional_, defaults to `"<s>"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **mask\_token** (`str`, _optional_, defaults to `"<mask>"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
-   **tokenizer\_file** (`str`, _optional_) — The path to a tokenizer file to use instead of the vocab file.
-   **src\_lang** (`str`, _optional_) — The language to use as source language for translation.
-   **tgt\_lang** (`str`, _optional_) — The language to use as target language for translation.
-   **sp\_model\_kwargs** (`Dict[str, str]`) — Additional keyword arguments to pass to the model initialization.

Construct an NLLB tokenizer.

Adapted from [RobertaTokenizer](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaTokenizer) and [XLNetTokenizer](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetTokenizer). Based on [SentencePiece](https://github.com/google/sentencepiece).

The tokenization method is `<tokens> <eos> <language code>` for source language documents, and \`<language code>

<tokens> <eos>\` for target language documents.

Examples:

```
>>> from transformers import NllbTokenizer

>>> tokenizer = NllbTokenizer.from_pretrained(
...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
... )
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nllb/tokenization_nllb.py#L267)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. An NLLB sequence has the following format, where `X` represents the sequence:

-   `input_ids` (for encoder) `X [eos, src_lang_code]`
-   `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a separator.

## NllbTokenizerFast

### class transformers.NllbTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nllb/tokenization_nllb_fast.py#L61)

( vocab\_file = None tokenizer\_file = None bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' src\_lang = None tgt\_lang = None additional\_special\_tokens = None legacy\_behaviour = False \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.
    
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **sep\_token** (`str`, _optional_, defaults to `"</s>"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **cls\_token** (`str`, _optional_, defaults to `"<s>"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **mask\_token** (`str`, _optional_, defaults to `"<mask>"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
-   **tokenizer\_file** (`str`, _optional_) — The path to a tokenizer file to use instead of the vocab file.
-   **src\_lang** (`str`, _optional_) — The language to use as source language for translation.
-   **tgt\_lang** (`str`, _optional_) — The language to use as target language for translation.

Construct a “fast” NLLB tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on [BPE](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=BPE#models).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

The tokenization method is `<tokens> <eos> <language code>` for source language documents, and \`<language code>

<tokens> <eos>\` for target language documents.

Examples:

```
>>> from transformers import NllbTokenizerFast

>>> tokenizer = NllbTokenizerFast.from_pretrained(
...     "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="fra_Latn"
... )
>>> example_english_phrase = " UN Chief Says There Is No Military Solution in Syria"
>>> expected_translation_french = "Le chef de l'ONU affirme qu'il n'y a pas de solution militaire en Syrie."
>>> inputs = tokenizer(example_english_phrase, text_target=expected_translation_french, return_tensors="pt")
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nllb/tokenization_nllb_fast.py#L210)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

list of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. The special tokens depend on calling set\_lang.

An NLLB sequence has the following format, where `X` represents the sequence:

-   `input_ids` (for encoder) `X [eos, src_lang_code]`
-   `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a separator.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nllb/tokenization_nllb_fast.py#L239)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task. nllb does not make use of token type ids, therefore a list of zeros is returned.

#### set\_src\_lang\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nllb/tokenization_nllb_fast.py#L294)

( src\_lang )

Reset the special tokens to the source lang setting.

-   In legacy mode: No prefix and suffix=\[eos, src\_lang\_code\].
-   In default mode: Prefix=\[src\_lang\_code\], suffix = \[eos\]

#### set\_tgt\_lang\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/nllb/tokenization_nllb_fast.py#L317)

( lang: str )

Reset the special tokens to the target lang setting.

-   In legacy mode: No prefix and suffix=\[eos, tgt\_lang\_code\].
-   In default mode: Prefix=\[tgt\_lang\_code\], suffix = \[eos\]