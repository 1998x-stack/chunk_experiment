# MarianMT

[![Models](https://img.shields.io/badge/All_model_pages-marian-blueviolet)](https://huggingface.co/models?filter=marian) [![Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/docs-demos/opus-mt-zh-en)

**Bugs:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=sshleifer&labels=&template=bug-report.md&title) and assign @patrickvonplaten.

Translations should be similar, but not identical to output in the test set linked to in each model card.

Tips:

-   A framework for translation models, using the same models as BART.

## Implementation Notes

-   Each model is about 298 MB on disk, there are more than 1,000 models.
    
-   The list of supported language pairs can be found [here](https://huggingface.co/Helsinki-NLP).
    
-   Models were originally trained by [Jörg Tiedemann](https://researchportal.helsinki.fi/en/persons/j%C3%B6rg-tiedemann) using the [Marian](https://marian-nmt.github.io/) C++ library, which supports fast training and translation.
    
-   All models are transformer encoder-decoders with 6 layers in each component. Each model’s performance is documented in a model card.
    
-   The 80 opus models that require BPE preprocessing are not supported.
    
-   The modeling code is the same as [BartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bart#transformers.BartForConditionalGeneration) with a few minor modifications:
    
    -   static (sinusoid) positional embeddings (`MarianConfig.static_position_embeddings=True`)
    -   no layernorm\_embedding (`MarianConfig.normalize_embedding=False`)
    -   the model starts generating with `pad_token_id` (which has 0 as a token\_embedding) as the prefix (Bart uses `<s/>`),
-   Code to bulk convert models can be found in `convert_marian_to_pytorch.py`.
    
-   This model was contributed by [sshleifer](https://huggingface.co/sshleifer).
    

## Naming

-   All model names use the following format: `Helsinki-NLP/opus-mt-{src}-{tgt}`
-   The language codes used to name models are inconsistent. Two digit codes can usually be found [here](https://developers.google.com/admin-sdk/directory/v1/languages), three digit codes require googling “language code {code}“.
-   Codes formatted like `es_AR` are usually `code_{region}`. That one is Spanish from Argentina.
-   The models were converted in two stages. The first 1000 models use ISO-639-2 codes to identify languages, the second group use a combination of ISO-639-5 codes and ISO-639-2 codes.

## Examples

-   Since Marian models are smaller than many other translation models available in the library, they can be useful for fine-tuning experiments and integration tests.
-   [Fine-tune on GPU](https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/train_distil_marian_enro.sh)

## Multilingual Models

-   All model names use the following format: `Helsinki-NLP/opus-mt-{src}-{tgt}`:
-   If a model can output multiple languages, and you should specify a language code by prepending the desired output language to the `src_text`.
-   You can see a models’s supported language codes in its model card, under target constituents, like in [opus-mt-en-roa](https://huggingface.co/Helsinki-NLP/opus-mt-en-roa).
-   Note that if a model is only multilingual on the source side, like `Helsinki-NLP/opus-mt-roa-en`, no language codes are required.

New multi-lingual models from the [Tatoeba-Challenge repo](https://github.com/Helsinki-NLP/Tatoeba-Challenge) require 3 character language codes:

```
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fra<< this is a sentence in english that we want to translate to french",
...     ">>por<< This should go to portuguese",
...     ">>esp<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-roa"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)
>>> print(tokenizer.supported_language_codes)
['>>zlm_Latn<<', '>>mfe<<', '>>hat<<', '>>pap<<', '>>ast<<', '>>cat<<', '>>ind<<', '>>glg<<', '>>wln<<', '>>spa<<', '>>fra<<', '>>ron<<', '>>por<<', '>>ita<<', '>>oci<<', '>>arg<<', '>>min<<']

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français",
 'Isto deve ir para o português.',
 'Y esto al español']
```

Here is the code to see all available pretrained models on the hub:

```
from huggingface_hub import list_models

model_list = list_models()
org = "Helsinki-NLP"
model_ids = [x.modelId for x in model_list if x.modelId.startswith(org)]
suffix = [x.split("/")[1] for x in model_ids]
old_style_multi_models = [f"{org}/{s}" for s in suffix if s != s.lower()]
```

## Old Style Multi-Lingual Models

These are the old style multi-lingual models ported from the OPUS-MT-Train repo: and the members of each language group:

```
['Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU',
 'Helsinki-NLP/opus-mt-ROMANCE-en',
 'Helsinki-NLP/opus-mt-SCANDINAVIA-SCANDINAVIA',
 'Helsinki-NLP/opus-mt-de-ZH',
 'Helsinki-NLP/opus-mt-en-CELTIC',
 'Helsinki-NLP/opus-mt-en-ROMANCE',
 'Helsinki-NLP/opus-mt-es-NORWAY',
 'Helsinki-NLP/opus-mt-fi-NORWAY',
 'Helsinki-NLP/opus-mt-fi-ZH',
 'Helsinki-NLP/opus-mt-fi_nb_no_nn_ru_sv_en-SAMI',
 'Helsinki-NLP/opus-mt-sv-NORWAY',
 'Helsinki-NLP/opus-mt-sv-ZH']
GROUP_MEMBERS = {
 'ZH': ['cmn', 'cn', 'yue', 'ze_zh', 'zh_cn', 'zh_CN', 'zh_HK', 'zh_tw', 'zh_TW', 'zh_yue', 'zhs', 'zht', 'zh'],
 'ROMANCE': ['fr', 'fr_BE', 'fr_CA', 'fr_FR', 'wa', 'frp', 'oc', 'ca', 'rm', 'lld', 'fur', 'lij', 'lmo', 'es', 'es_AR', 'es_CL', 'es_CO', 'es_CR', 'es_DO', 'es_EC', 'es_ES', 'es_GT', 'es_HN', 'es_MX', 'es_NI', 'es_PA', 'es_PE', 'es_PR', 'es_SV', 'es_UY', 'es_VE', 'pt', 'pt_br', 'pt_BR', 'pt_PT', 'gl', 'lad', 'an', 'mwl', 'it', 'it_IT', 'co', 'nap', 'scn', 'vec', 'sc', 'ro', 'la'],
 'NORTH_EU': ['de', 'nl', 'fy', 'af', 'da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
 'SCANDINAVIA': ['da', 'fo', 'is', 'no', 'nb', 'nn', 'sv'],
 'SAMI': ['se', 'sma', 'smj', 'smn', 'sms'],
 'NORWAY': ['nb_NO', 'nb', 'nn_NO', 'nn', 'nog', 'no_nb', 'no'],
 'CELTIC': ['ga', 'cy', 'br', 'gd', 'kw', 'gv']
}
```

Example of translating english to many romance languages, using old-style 2 character language codes

```
>>> from transformers import MarianMTModel, MarianTokenizer

>>> src_text = [
...     ">>fr<< this is a sentence in english that we want to translate to french",
...     ">>pt<< This should go to portuguese",
...     ">>es<< And this to Spanish",
... ]

>>> model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
>>> tokenizer = MarianTokenizer.from_pretrained(model_name)

>>> model = MarianMTModel.from_pretrained(model_name)
>>> translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
>>> tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
["c'est une phrase en anglais que nous voulons traduire en français", 
 'Isto deve ir para o português.',
 'Y esto al español']
```

## Documentation resources

-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)
-   [Causal language modeling task guide](../tasks/language_modeling)

## MarianConfig

### class transformers.MarianConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/configuration_marian.py#L34)

( vocab\_size = 58101 decoder\_vocab\_size = None max\_position\_embeddings = 1024 encoder\_layers = 12 encoder\_ffn\_dim = 4096 encoder\_attention\_heads = 16 decoder\_layers = 12 decoder\_ffn\_dim = 4096 decoder\_attention\_heads = 16 encoder\_layerdrop = 0.0 decoder\_layerdrop = 0.0 use\_cache = True is\_encoder\_decoder = True activation\_function = 'gelu' d\_model = 1024 dropout = 0.1 attention\_dropout = 0.0 activation\_dropout = 0.0 init\_std = 0.02 decoder\_start\_token\_id = 58100 scale\_embedding = False pad\_token\_id = 58100 eos\_token\_id = 0 forced\_eos\_token\_id = 0 share\_encoder\_decoder\_embeddings = True \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 58101) — Vocabulary size of the Marian model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [MarianModel](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianModel) or [TFMarianModel](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.TFMarianModel).
-   **d\_model** (`int`, _optional_, defaults to 1024) — Dimensionality of the layers and the pooler layer.
-   **encoder\_layers** (`int`, _optional_, defaults to 12) — Number of encoder layers.
-   **decoder\_layers** (`int`, _optional_, defaults to 12) — Number of decoder layers.
-   **encoder\_attention\_heads** (`int`, _optional_, defaults to 16) — Number of attention heads for each attention layer in the Transformer encoder.
-   **decoder\_attention\_heads** (`int`, _optional_, defaults to 16) — Number of attention heads for each attention layer in the Transformer decoder.
-   **decoder\_ffn\_dim** (`int`, _optional_, defaults to 4096) — Dimensionality of the “intermediate” (often named feed-forward) layer in decoder.
-   **encoder\_ffn\_dim** (`int`, _optional_, defaults to 4096) — Dimensionality of the “intermediate” (often named feed-forward) layer in decoder.
-   **activation\_function** (`str` or `function`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **dropout** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for the attention probabilities.
-   **activation\_dropout** (`float`, _optional_, defaults to 0.0) — The dropout ratio for activations inside the fully connected layer.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 1024) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **init\_std** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **encoder\_layerdrop** (`float`, _optional_, defaults to 0.0) — The LayerDrop probability for the encoder. See the \[LayerDrop paper\](see [https://arxiv.org/abs/1909.11556](https://arxiv.org/abs/1909.11556)) for more details.
-   **decoder\_layerdrop** (`float`, _optional_, defaults to 0.0) — The LayerDrop probability for the decoder. See the \[LayerDrop paper\](see [https://arxiv.org/abs/1909.11556](https://arxiv.org/abs/1909.11556)) for more details.
-   **scale\_embedding** (`bool`, _optional_, defaults to `False`) — Scale embeddings by diving by sqrt(d\_model).
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models)
-   **forced\_eos\_token\_id** (`int`, _optional_, defaults to 0) — The id of the token to force as the last generated token when `max_length` is reached. Usually set to `eos_token_id`.

This is the configuration class to store the configuration of a [MarianModel](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianModel). It is used to instantiate an Marian model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Marian [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import MarianModel, MarianConfig

>>> 
>>> configuration = MarianConfig()

>>> 
>>> model = MarianModel(configuration)

>>> 
>>> configuration = model.config
```

## MarianTokenizer

### class transformers.MarianTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/tokenization_marian.py#L63)

( source\_spm target\_spm vocab target\_vocab\_file = None source\_lang = None target\_lang = None unk\_token = '<unk>' eos\_token = '</s>' pad\_token = '<pad>' model\_max\_length = 512 sp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = None separate\_vocabs = False \*\*kwargs )

Parameters

-   **source\_spm** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that contains the vocabulary for the source language.
-   **target\_spm** (`str`) — [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .spm extension) that contains the vocabulary for the target language.
-   **source\_lang** (`str`, _optional_) — A string representing the source language.
-   **target\_lang** (`str`, _optional_) — A string representing the target language.
-   **unk\_token** (`str`, _optional_, defaults to `"<unk>"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **eos\_token** (`str`, _optional_, defaults to `"</s>"`) — The end of sequence token.
-   **pad\_token** (`str`, _optional_, defaults to `"<pad>"`) — The token used for padding, for example when batching sequences of different lengths.
-   **model\_max\_length** (`int`, _optional_, defaults to 512) — The maximum sentence length the model accepts.
-   **additional\_special\_tokens** (`List[str]`, _optional_, defaults to `["<eop>", "<eod>"]`) — Additional special tokens used by the tokenizer.
-   **sp\_model\_kwargs** (`dict`, _optional_) — Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things, to set:
    
    -   `enable_sampling`: Enable subword regularization.
        
    -   `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.
        
        -   `nbest_size = {0,1}`: No sampling is performed.
        -   `nbest_size > 1`: samples from the nbest\_size results.
        -   `nbest_size < 0`: assuming that nbest\_size is infinite and samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling algorithm.
    -   `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for BPE-dropout.
        
    

Construct a Marian tokenizer. Based on [SentencePiece](https://github.com/google/sentencepiece).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

Examples:

```
>>> from transformers import MarianForCausalLM, MarianTokenizer

>>> model = MarianForCausalLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
>>> tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
>>> src_texts = ["I am a small frog.", "Tom asked his teacher for advice."]
>>> tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  
>>> inputs = tokenizer(src_texts, text_target=tgt_texts, return_tensors="pt", padding=True)

>>> outputs = model(**inputs)  
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/tokenization_marian.py#L287)

( token\_ids\_0 token\_ids\_1 = None )

Build model inputs from a sequence by appending eos\_token\_id.

## MarianModel

### class transformers.MarianModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_marian.py#L1106)

( config: MarianConfig )

Parameters

-   **config** ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Marian Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_marian.py#L1190)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None decoder\_input\_ids: typing.Optional\[torch.LongTensor\] = None decoder\_attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None decoder\_head\_mask: typing.Optional\[torch.Tensor\] = None cross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = None encoder\_outputs: typing.Union\[typing.Tuple\[torch.Tensor\], transformers.modeling\_outputs.BaseModelOutput, NoneType\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None decoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    Marian uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
-   **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
-   **head\_mask** (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, _optional_) — Tuple consists of (`last_hidden_state`, _optional_: `hidden_states`, _optional_: `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the optional initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the optional initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [MarianModel](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, MarianModel

>>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
>>> model = MarianModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

>>> inputs = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt")
>>> decoder_inputs = tokenizer(
...     "<pad> Studien haben gezeigt dass es hilfreich ist einen Hund zu besitzen",
...     return_tensors="pt",
...     add_special_tokens=False,
... )
>>> outputs = model(input_ids=inputs.input_ids, decoder_input_ids=decoder_inputs.input_ids)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 26, 512]
```

## MarianMTModel

### class transformers.MarianMTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_marian.py#L1292)

( config: MarianConfig )

Parameters

-   **config** ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Marian Model with a language modeling head. Can be used for summarization. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_marian.py#L1416)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None decoder\_input\_ids: typing.Optional\[torch.LongTensor\] = None decoder\_attention\_mask: typing.Optional\[torch.Tensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None decoder\_head\_mask: typing.Optional\[torch.Tensor\] = None cross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = None encoder\_outputs: typing.Union\[typing.Tuple\[torch.Tensor\], transformers.modeling\_outputs.BaseModelOutput, NoneType\] = None past\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None decoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    Marian uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
-   **decoder\_attention\_mask** (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
-   **head\_mask** (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **encoder\_outputs** (`tuple(tuple(torch.FloatTensor)`, _optional_) — Tuple consists of (`last_hidden_state`, _optional_: `hidden_states`, _optional_: `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **decoder\_inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
    
    If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.
    
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

A [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Language modeling loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [MarianMTModel](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianMTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Pytorch version of marian-nmt’s transformer.h (c++). Designed for the OPUS-NMT translation checkpoints. Available models are listed [here](https://huggingface.co/models?search=Helsinki-NLP).

Examples:

```
>>> from transformers import AutoTokenizer, MarianMTModel

>>> src = "fr"  
>>> trg = "en"  

>>> model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"
>>> model = MarianMTModel.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)

>>> sample_text = "où est l'arrêt de bus ?"
>>> batch = tokenizer([sample_text], return_tensors="pt")

>>> generated_ids = model.generate(**batch)
>>> tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
"Where's the bus stop?"
```

## MarianForCausalLM

### class transformers.MarianForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_marian.py#L1557)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_marian.py#L1590)

( input\_ids: LongTensor = None attention\_mask: typing.Optional\[torch.Tensor\] = None encoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = None encoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None head\_mask: typing.Optional\[torch.Tensor\] = None cross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = None past\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None use\_cache: typing.Optional\[bool\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **encoder\_hidden\_states** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.
-   **encoder\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
-   **head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are only required when the model is used as a decoder in a Sequence to Sequence model.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
    If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
-   **use\_cache** (`bool`, _optional_) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) and inputs.

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
    

Example:

```
>>> from transformers import AutoTokenizer, MarianForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
>>> model = MarianForCausalLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en", add_cross_attention=False)
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
>>> list(logits.shape) == expected_shape
True
```

## TFMarianModel

### class transformers.TFMarianModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_tf_marian.py#L1157)

( \*args \*\*kwargs )

Parameters

-   **config** ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare MARIAN Model outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_tf_marian.py#L1169)

( input\_ids: tf.Tensor | None = None attention\_mask: tf.Tensor | None = None decoder\_input\_ids: tf.Tensor | None = None decoder\_attention\_mask: tf.Tensor | None = None decoder\_position\_ids: tf.Tensor | None = None head\_mask: tf.Tensor | None = None decoder\_head\_mask: tf.Tensor | None = None cross\_attn\_head\_mask: tf.Tensor | None = None encoder\_outputs: tf.Tensor | None = None past\_key\_values: Tuple\[Tuple\[tf.Tensor\]\] | None = None inputs\_embeds: tf.Tensor | None = None decoder\_inputs\_embeds: tf.Tensor | None = None use\_cache: bool | None = None output\_attentions: bool | None = None output\_hidden\_states: bool | None = None return\_dict: bool | None = None training: bool = False \*\*kwargs ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    Marian uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
-   **decoder\_attention\_mask** (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, _optional_) — will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.
-   **decoder\_position\_ids** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **encoder\_outputs** (`tf.FloatTensor`, _optional_) — hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder. of shape `(batch_size, sequence_length, hidden_size)` is a sequence of
-   **past\_key\_values** (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`) — contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`). Set to `False` during training, `True` during generation
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

A [transformers.modeling\_tf\_outputs.TFSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`List[tf.Tensor]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFMarianModel](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.TFMarianModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFMarianModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
>>> model = TFMarianModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFMarianMTModel

### class transformers.TFMarianMTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_tf_marian.py#L1262)

( \*args \*\*kwargs )

Parameters

-   **config** ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The MARIAN Model with a language modeling head. Can be used for summarization. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_tf_marian.py#L1300)

( input\_ids: tf.Tensor | None = None attention\_mask: tf.Tensor | None = None decoder\_input\_ids: tf.Tensor | None = None decoder\_attention\_mask: tf.Tensor | None = None decoder\_position\_ids: tf.Tensor | None = None head\_mask: tf.Tensor | None = None decoder\_head\_mask: tf.Tensor | None = None cross\_attn\_head\_mask: tf.Tensor | None = None encoder\_outputs: TFBaseModelOutput | None = None past\_key\_values: Tuple\[Tuple\[tf.Tensor\]\] | None = None inputs\_embeds: tf.Tensor | None = None decoder\_inputs\_embeds: tf.Tensor | None = None use\_cache: bool | None = None output\_attentions: bool | None = None output\_hidden\_states: bool | None = None return\_dict: bool | None = None labels: tf.Tensor | None = None training: bool = False ) → [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`tf.Tensor` of shape `({0})`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`tf.Tensor` of shape `({0})`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    Marian uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
    
-   **decoder\_attention\_mask** (`tf.Tensor` of shape `(batch_size, target_sequence_length)`, _optional_) — will be made by default and ignore pad tokens. It is not recommended to set this for most use cases.
-   **decoder\_position\_ids** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **head\_mask** (`tf.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **decoder\_head\_mask** (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **cross\_attn\_head\_mask** (`tf.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, _optional_) — Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **encoder\_outputs** (`tf.FloatTensor`, _optional_) — hidden states at the output of the last layer of the encoder. Used in the cross-attention of the decoder. of shape `(batch_size, sequence_length, hidden_size)` is a sequence of
-   **past\_key\_values** (`Tuple[Tuple[tf.Tensor]]` of length `config.n_layers`) — contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that don’t have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`). Set to `False` during training, `True` during generation
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.tensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

A [transformers.modeling\_tf\_outputs.TFSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSeq2SeqLMOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(n,)`, _optional_, where n is the number of non-masked labels, returned when `labels` is provided) — Language modeling loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`List[tf.Tensor]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) of the decoder that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFMarianMTModel](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.TFMarianMTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

TF version of marian-nmt’s transformer.h (c++). Designed for the OPUS-NMT translation checkpoints. Available models are listed [here](https://huggingface.co/models?search=Helsinki-NLP).

Examples:

```
>>> from transformers import AutoTokenizer, TFMarianMTModel
>>> from typing import List

>>> src = "fr"  
>>> trg = "en"  
>>> sample_text = "où est l'arrêt de bus ?"
>>> model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"

>>> model = TFMarianMTModel.from_pretrained(model_name)
>>> tokenizer = AutoTokenizer.from_pretrained(model_name)
>>> batch = tokenizer([sample_text], return_tensors="tf")
>>> gen = model.generate(**batch)
>>> tokenizer.batch_decode(gen, skip_special_tokens=True)
"Where is the bus stop ?"
```

## FlaxMarianModel

### class transformers.FlaxMarianModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_flax_marian.py#L1205)

( config: MarianConfig input\_shape: typing.Tuple\[int\] = (1, 1) seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True \*\*kwargs )

Parameters

-   **config** ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare Marian Model transformer outputting raw hidden-states without any specific head on top. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_flax_marian.py#L1140)

( input\_ids: Array attention\_mask: typing.Optional\[jax.Array\] = None decoder\_input\_ids: typing.Optional\[jax.Array\] = None decoder\_attention\_mask: typing.Optional\[jax.Array\] = None position\_ids: typing.Optional\[jax.Array\] = None decoder\_position\_ids: typing.Optional\[jax.Array\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None train: bool = False params: dict = None dropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`jnp.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`jnp.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    For translation and summarization training, `decoder_input_ids` should be provided. If no `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right for denoising pre-training following the paper.
    
-   **decoder\_attention\_mask** (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
    
    If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **decoder\_position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxSeq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) and inputs.

-   **last\_hidden\_state** (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the decoder of the model.
    
    If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1, hidden_size)` is output.
    
-   **past\_key\_values** (`tuple(tuple(jnp.ndarray))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxMarianPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxMarianModel

>>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
>>> model = FlaxMarianModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## FlaxMarianMTModel

### class transformers.FlaxMarianMTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_flax_marian.py#L1289)

( config: MarianConfig input\_shape: typing.Tuple\[int\] = (1, 1) seed: int = 0 dtype: dtype = <class 'jax.numpy.float32'> \_do\_init: bool = True \*\*kwargs )

Parameters

-   **config** ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The MARIAN Model with a language modeling head. Can be used for translation. This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/marian/modeling_flax_marian.py#L1140)

( input\_ids: Array attention\_mask: typing.Optional\[jax.Array\] = None decoder\_input\_ids: typing.Optional\[jax.Array\] = None decoder\_attention\_mask: typing.Optional\[jax.Array\] = None position\_ids: typing.Optional\[jax.Array\] = None decoder\_position\_ids: typing.Optional\[jax.Array\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None train: bool = False params: dict = None dropout\_rng: PRNGKey = None ) → [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`jnp.ndarray` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **attention\_mask** (`jnp.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **decoder\_input\_ids** (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, _optional_) — Indices of decoder input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are decoder input IDs?](../glossary#decoder-input-ids)
    
    For translation and summarization training, `decoder_input_ids` should be provided. If no `decoder_input_ids` is provided, the model will create this tensor by shifting the `input_ids` to the right for denoising pre-training following the paper.
    
-   **decoder\_attention\_mask** (`jnp.ndarray` of shape `(batch_size, target_sequence_length)`, _optional_) — Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.
    
    If you want to change padding behavior, you should modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more information on the default strategy.
    
-   **position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **decoder\_position\_ids** (`numpy.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each decoder input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_flax\_outputs.FlaxSeq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxSeq2SeqLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([MarianConfig](/docs/transformers/v4.34.0/en/model_doc/marian#transformers.MarianConfig)) and inputs.

-   **logits** (`jnp.ndarray` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **past\_key\_values** (`tuple(tuple(jnp.ndarray))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(jnp.ndarray)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **decoder\_hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
    
-   **decoder\_attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **encoder\_last\_hidden\_state** (`jnp.ndarray` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Sequence of hidden-states at the output of the last layer of the encoder of the model.
    
-   **encoder\_hidden\_states** (`tuple(jnp.ndarray)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `jnp.ndarray` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
    
-   **encoder\_attentions** (`tuple(jnp.ndarray)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `jnp.ndarray` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The `FlaxMarianPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, FlaxMarianMTModel

>>> model = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
>>> tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

>>> text = "My friends are cool but they eat too many carbs."
>>> input_ids = tokenizer(text, max_length=64, return_tensors="jax").input_ids

>>> sequences = model.generate(input_ids, max_length=64, num_beams=2).sequences

>>> outputs = tokenizer.batch_decode(sequences, skip_special_tokens=True)
>>> 
```