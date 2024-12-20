# GPT-Sw3

## Overview

The GPT-Sw3 model was first proposed in [Lessons Learned from GPT-SW3: Building the First Large-Scale Generative Language Model for Swedish](http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.376.pdf) by Ariel Ekgren, Amaru Cuba Gyllensten, Evangelia Gogoulou, Alice Heiman, Severine Verlinden, Joey Öhman, Fredrik Carlsson, Magnus Sahlgren.

Since that first paper the authors have extended their work and trained new models on their new 1.2TB corpora named The Nordic Pile.

GPT-Sw3 is a collection of large decoder-only pretrained transformer language models that were developed by AI Sweden in collaboration with RISE and the WASP WARA for Media and Language. GPT-Sw3 has been trained on a dataset containing 320B tokens in Swedish, Norwegian, Danish, Icelandic, English, and programming code. The model was pretrained using a causal language modeling (CLM) objective utilizing the NeMo Megatron GPT implementation.

This model was contributed by [AI Sweden](https://huggingface.co/AI-Sweden).

The implementation uses the [GPT2Model](https://huggingface.co/docs/transformers/model_doc/gpt2) coupled with our `GPTSw3Tokenizer`. This means that `AutoTokenizer` and `AutoModelForCausalLM` map to our tokenizer implementation and the corresponding GPT2 model implementation respectively. _Note that sentencepiece is required to use our tokenizer and can be installed with:_ `pip install transformers[sentencepiece]` or `pip install sentencepiece`

Example usage:

```
>>> from transformers import AutoTokenizer, AutoModelForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("AI-Sweden/gpt-sw3-356m")
>>> model = AutoModelForCausalLM.from_pretrained("AI-Sweden/gpt-sw3-356m")

>>> input_ids = tokenizer("Träd är fina för att", return_tensors="pt")["input_ids"]

>>> generated_token_ids = model.generate(inputs=input_ids, max_new_tokens=10, do_sample=True)[0]

>>> print(tokenizer.decode(generated_token_ids))
Träd är fina för att de är färgstarka. Men ibland är det fint
```

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Causal language modeling task guide](../tasks/language_modeling)

## GPTSw3Tokenizer

### class transformers.GPTSw3Tokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt_sw3/tokenization_gpt_sw3.py#L41)

( vocab\_file do\_lower\_case = False remove\_space = False keep\_accents = False pad\_token = None unk\_token = None eos\_token = None bos\_token = None sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a _.spm_ extension) that contains the vocabulary necessary to instantiate a tokenizer.
-   **do\_lower\_case** (`bool`, _optional_, defaults to `False`) — Whether or not to lowercase the input when tokenizing.
-   **remove\_space** (`bool`, _optional_, defaults to `False`) — Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
-   **keep\_accents** (`bool`, _optional_, defaults to `False`) — Whether or not to keep accents when tokenizing.
-   **bos\_token** (`str`, _optional_) — The beginning of sequence token that can be used for downstream task, was not seen during pretraining. If not provided, will default to ’~’ or ’<|endoftext|>’, depending on model size.~
-   **eos\_token** (`str`, _optional_) — The end of sequence token seen during pretraining. If not provided, will default to ’<|endoftext|>’
-   **unk\_token** (`str`, _optional_) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead. If not provided, will default to ’‘.
-   **pad\_token** (`str`, _optional_) — The token used for padding, for example when batching sequences of different lengths. If not provided, will default to ’’ or ’’ depending on model size.
-   **sp\_model\_kwargs** (`dict`, _optional_) — Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things, to set:
    
    -   `enable_sampling`: Enable subword regularization.
        
    -   `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.
        
        -   `nbest_size = {0,1}`: No sampling is performed.
        -   `nbest_size > 1`: samples from the nbest\_size results.
        -   `nbest_size < 0`: assuming that nbest\_size is infinite and samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling algorithm.
    -   `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for BPE-dropout.
        
    
-   **sp\_model** (`SentencePieceProcessor`) — The _SentencePiece_ processor that is used for every conversion (string, tokens and IDs).
-   **whitespaces** (`set`) — The whitespaces that are replaced in the whitespace normalization in preprocessing.
-   **non\_printing\_characters\_re** (`Pattern`) — The compiled regular expression to remove non-printing characters in preprocessing.

Construct an GPTSw3 tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

Example usage:

```
>>> from transformers import GPTSw3Tokenizer

>>> tokenizer = GPTSw3Tokenizer.from_pretrained("AI-Sweden/gpt-sw3-126m")
>>> tokenizer("Svenska är kul!")["input_ids"]
[1814, 377, 3617, 63504]
```

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/gpt_sw3/tokenization_gpt_sw3.py#L254)

( save\_directory: str filename\_prefix: typing.Optional\[str\] = None )