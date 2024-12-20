# BARThez

## Overview

The BARThez model was proposed in [BARThez: a Skilled Pretrained French Sequence-to-Sequence Model](https://arxiv.org/abs/2010.12321) by Moussa Kamal Eddine, Antoine J.-P. Tixier, Michalis Vazirgiannis on 23 Oct, 2020.

The abstract of the paper:

_Inductive transfer learning, enabled by self-supervised learning, have taken the entire Natural Language Processing (NLP) field by storm, with models such as BERT and BART setting new state of the art on countless natural language understanding tasks. While there are some notable exceptions, most of the available models and research have been conducted for the English language. In this work, we introduce BARThez, the first BART model for the French language (to the best of our knowledge). BARThez was pretrained on a very large monolingual French corpus from past research that we adapted to suit BART’s perturbation schemes. Unlike already existing BERT-based French language models such as CamemBERT and FlauBERT, BARThez is particularly well-suited for generative tasks, since not only its encoder but also its decoder is pretrained. In addition to discriminative tasks from the FLUE benchmark, we evaluate BARThez on a novel summarization dataset, OrangeSum, that we release with this paper. We also continue the pretraining of an already pretrained multilingual BART on BARThez’s corpus, and we show that the resulting model, which we call mBARTHez, provides a significant boost over vanilla BARThez, and is on par with or outperforms CamemBERT and FlauBERT._

This model was contributed by [moussakam](https://huggingface.co/moussakam). The Authors’ code can be found [here](https://github.com/moussaKam/BARThez).

### Examples

-   BARThez can be fine-tuned on sequence-to-sequence tasks in a similar way as BART, check: [examples/pytorch/summarization/](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization/README.md).

## BarthezTokenizer

### class transformers.BarthezTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/barthez/tokenization_barthez.py#L53)

( vocab\_file bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a _.spm_ extension) that contains the vocabulary necessary to instantiate a tokenizer.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.
    
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **sep\_token** (`str`, _optional_, defaults to `"</s>"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **cls\_token** (`str`, _optional_, defaults to `"<s>"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **mask\_token** (`str`, _optional_, defaults to `"<mask>"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
-   **additional\_special\_tokens** (`List[str]`, _optional_, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`) — Additional special tokens used by the tokenizer.
-   **sp\_model\_kwargs** (`dict`, _optional_) — Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things, to set:
    
    -   `enable_sampling`: Enable subword regularization.
        
    -   `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.
        
        -   `nbest_size = {0,1}`: No sampling is performed.
        -   `nbest_size > 1`: samples from the nbest\_size results.
        -   `nbest_size < 0`: assuming that nbest\_size is infinite and samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling algorithm.
    -   `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for BPE-dropout.
        
    
-   **sp\_model** (`SentencePieceProcessor`) — The _SentencePiece_ processor that is used for every conversion (string, tokens and IDs).

Adapted from [CamembertTokenizer](/docs/transformers/v4.34.0/en/model_doc/camembert#transformers.CamembertTokenizer) and [BartTokenizer](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartTokenizer). Construct a BARThez tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/barthez/tokenization_barthez.py#L161)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BARThez sequence has the following format:

-   single sequence: `<s> X </s>`
-   pair of sequences: `<s> A </s></s> B </s>`

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/barthez/tokenization_barthez.py#L214)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task.

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/barthez/tokenization_barthez.py#L187)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## BarthezTokenizerFast

### class transformers.BarthezTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/barthez/tokenization_barthez_fast.py#L62)

( vocab\_file = None tokenizer\_file = None bos\_token = '<s>' eos\_token = '</s>' sep\_token = '</s>' cls\_token = '<s>' unk\_token = '<unk>' pad\_token = '<pad>' mask\_token = '<mask>' \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a _.spm_ extension) that contains the vocabulary necessary to instantiate a tokenizer.
-   **bos\_token** (`str`, _optional_, defaults to `"<s>"`) — The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
    
    When building a sequence using special tokens, this is not the token that is used for the beginning of sequence. The token used is the `cls_token`.
    
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
    
    When building a sequence using special tokens, this is not the token that is used for the end of sequence. The token used is the `sep_token`.
    
-   **sep\_token** (`str`, _optional_, defaults to `"</s>"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **cls\_token** (`str`, _optional_, defaults to `"<s>"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **mask\_token** (`str`, _optional_, defaults to `"<mask>"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
-   **additional\_special\_tokens** (`List[str]`, _optional_, defaults to `["<s>NOTUSED", "</s>NOTUSED"]`) — Additional special tokens used by the tokenizer.

Adapted from [CamembertTokenizer](/docs/transformers/v4.34.0/en/model_doc/camembert#transformers.CamembertTokenizer) and [BartTokenizer](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartTokenizer). Construct a “fast” BARThez tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/barthez/tokenization_barthez_fast.py#L154)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BARThez sequence has the following format:

-   single sequence: `<s> X </s>`
-   pair of sequences: `<s> A </s></s> B </s>`

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/barthez/tokenization_barthez_fast.py#L180)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of zeros.

Create a mask from the two sequences passed to be used in a sequence-pair classification task.