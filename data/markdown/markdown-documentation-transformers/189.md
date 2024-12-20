# LayoutLM

## Overview

The LayoutLM model was proposed in the paper [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. It’s a simple but effective pretraining method of text and layout for document image understanding and information extraction tasks, such as form understanding and receipt understanding. It obtains state-of-the-art results on several downstream tasks:

-   form understanding: the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset (a collection of 199 annotated forms comprising more than 30,000 words).
-   receipt understanding: the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset (a collection of 626 receipts for training and 347 receipts for testing).
-   document image classification: the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset (a collection of 400,000 images belonging to one of 16 classes).

The abstract from the paper is the following:

_Pre-training techniques have been verified successfully in a variety of NLP tasks in recent years. Despite the widespread use of pretraining models for NLP applications, they almost exclusively focus on text-level manipulation, while neglecting layout and style information that is vital for document image understanding. In this paper, we propose the LayoutLM to jointly model interactions between text and layout information across scanned document images, which is beneficial for a great number of real-world document image understanding tasks such as information extraction from scanned documents. Furthermore, we also leverage image features to incorporate words’ visual information into LayoutLM. To the best of our knowledge, this is the first time that text and layout are jointly learned in a single framework for document-level pretraining. It achieves new state-of-the-art results in several downstream tasks, including form understanding (from 70.72 to 79.27), receipt understanding (from 94.02 to 95.24) and document image classification (from 93.07 to 94.42)._

Tips:

-   In addition to _input\_ids_, [forward()](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMModel.forward) also expects the input `bbox`, which are the bounding boxes (i.e. 2D-positions) of the input tokens. These can be obtained using an external OCR engine such as Google’s [Tesseract](https://github.com/tesseract-ocr/tesseract) (there’s a [Python wrapper](https://pypi.org/project/pytesseract/) available). Each bounding box should be in (x0, y0, x1, y1) format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on a 0-1000 scale. To normalize, you can use the following function:

```
def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
```

Here, `width` and `height` correspond to the width and height of the original document in which the token occurs. Those can be obtained using the Python Image Library (PIL) library for example, as follows:

```
from PIL import Image


image = Image.open(name_of_your_document).convert("RGB")

width, height = image.size
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with LayoutLM. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

Document Question Answering

-   A blog post on [fine-tuning LayoutLM for document-understanding using Keras & Hugging Face Transformers](https://www.philschmid.de/fine-tuning-layoutlm-keras).
    
-   A blog post on how to [fine-tune LayoutLM for document-understanding using only Hugging Face Transformers](https://www.philschmid.de/fine-tuning-layoutlm).
    
-   A notebook on how to [fine-tune LayoutLM on the FUNSD dataset with image embeddings](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Add_image_embeddings_to_LayoutLM.ipynb).
    
-   See also: [Document question answering task guide](../tasks/document_question_answering)
    

-   A notebook on how to [fine-tune LayoutLM for sequence classification on the RVL-CDIP dataset](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForSequenceClassification_on_RVL_CDIP.ipynb).
-   [Text classification task guide](../tasks/sequence_classification)

-   A notebook on how to [fine-tune LayoutLM for token classification on the FUNSD dataset](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LayoutLM/Fine_tuning_LayoutLMForTokenClassification_on_FUNSD.ipynb).
-   [Token classification task guide](../tasks/token_classification)

**Other resources**

-   [Masked language modeling task guide](../tasks/masked_language_modeling)

🚀 Deploy

-   A blog post on how to [Deploy LayoutLM with Hugging Face Inference Endpoints](https://www.philschmid.de/inference-endpoints-layoutlm).

## LayoutLMConfig

### class transformers.LayoutLMConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/configuration_layoutlm.py#L36)

( vocab\_size = 30522 hidden\_size = 768 num\_hidden\_layers = 12 num\_attention\_heads = 12 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 pad\_token\_id = 0 position\_embedding\_type = 'absolute' use\_cache = True max\_2d\_position\_embeddings = 1024 \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 30522) — Vocabulary size of the LayoutLM model. Defines the different tokens that can be represented by the _inputs\_ids_ passed to the forward method of [LayoutLMModel](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMModel).
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **num\_hidden\_layers** (`int`, _optional_, defaults to 12) — Number of hidden layers in the Transformer encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (i.e., feed-forward) layer in the Transformer encoder.
-   **hidden\_act** (`str` or `function`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 512) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **type\_vocab\_size** (`int`, _optional_, defaults to 2) — The vocabulary size of the `token_type_ids` passed into [LayoutLMModel](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMModel).
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **pad\_token\_id** (`int`, _optional_, defaults to 0) — The value used to pad input\_ids.
-   **position\_embedding\_type** (`str`, _optional_, defaults to `"absolute"`) — Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155). For more information on `"relative_key_query"`, please refer to _Method 4_ in [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
-   **use\_cache** (`bool`, _optional_, defaults to `True`) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True`.
-   **classifier\_dropout** (`float`, _optional_) — The dropout ratio for the classification head.
-   **max\_2d\_position\_embeddings** (`int`, _optional_, defaults to 1024) — The maximum value that the 2D position embedding might ever used. Typically set this to something large just in case (e.g., 1024).

This is the configuration class to store the configuration of a [LayoutLMModel](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMModel). It is used to instantiate a LayoutLM model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the LayoutLM [microsoft/layoutlm-base-uncased](https://huggingface.co/microsoft/layoutlm-base-uncased) architecture.

Configuration objects inherit from [BertConfig](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertConfig) and can be used to control the model outputs. Read the documentation from [BertConfig](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertConfig) for more information.

Examples:

```
>>> from transformers import LayoutLMConfig, LayoutLMModel

>>> 
>>> configuration = LayoutLMConfig()

>>> 
>>> model = LayoutLMModel(configuration)

>>> 
>>> configuration = model.config
```

## LayoutLMTokenizer

### class transformers.LayoutLMTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/tokenization_layoutlm.py#L75)

( vocab\_file do\_lower\_case = True do\_basic\_tokenize = True never\_split = None unk\_token = '\[UNK\]' sep\_token = '\[SEP\]' pad\_token = '\[PAD\]' cls\_token = '\[CLS\]' mask\_token = '\[MASK\]' tokenize\_chinese\_chars = True strip\_accents = None \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — File containing the vocabulary.
-   **do\_lower\_case** (`bool`, _optional_, defaults to `True`) — Whether or not to lowercase the input when tokenizing.
-   **do\_basic\_tokenize** (`bool`, _optional_, defaults to `True`) — Whether or not to do basic tokenization before WordPiece.
-   **never\_split** (`Iterable`, _optional_) — Collection of tokens which will never be split during tokenization. Only has an effect when `do_basic_tokenize=True`
-   **unk\_token** (`str`, _optional_, defaults to `"[UNK]"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **sep\_token** (`str`, _optional_, defaults to `"[SEP]"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **pad\_token** (`str`, _optional_, defaults to `"[PAD]"`) — The token used for padding, for example when batching sequences of different lengths.
-   **cls\_token** (`str`, _optional_, defaults to `"[CLS]"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **mask\_token** (`str`, _optional_, defaults to `"[MASK]"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
-   **tokenize\_chinese\_chars** (`bool`, _optional_, defaults to `True`) — Whether or not to tokenize Chinese characters.
    
    This should likely be deactivated for Japanese (see this [issue](https://github.com/huggingface/transformers/issues/328)).
    
-   **strip\_accents** (`bool`, _optional_) — Whether or not to strip all accents. If this option is not specified, then it will be determined by the value for `lowercase` (as in the original LayoutLM).

Construct a LayoutLM tokenizer. Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/tokenization_layoutlm.py#L208)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A LayoutLM sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/tokenization_layoutlm.py#L261)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A LayoutLM

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/tokenization_layoutlm.py#L233)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## LayoutLMTokenizerFast

### class transformers.LayoutLMTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/tokenization_layoutlm_fast.py#L62)

( vocab\_file = None tokenizer\_file = None do\_lower\_case = True unk\_token = '\[UNK\]' sep\_token = '\[SEP\]' pad\_token = '\[PAD\]' cls\_token = '\[CLS\]' mask\_token = '\[MASK\]' tokenize\_chinese\_chars = True strip\_accents = None \*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — File containing the vocabulary.
-   **do\_lower\_case** (`bool`, _optional_, defaults to `True`) — Whether or not to lowercase the input when tokenizing.
-   **unk\_token** (`str`, _optional_, defaults to `"[UNK]"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **sep\_token** (`str`, _optional_, defaults to `"[SEP]"`) — The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for sequence classification or for a text and a question for question answering. It is also used as the last token of a sequence built with special tokens.
-   **pad\_token** (`str`, _optional_, defaults to `"[PAD]"`) — The token used for padding, for example when batching sequences of different lengths.
-   **cls\_token** (`str`, _optional_, defaults to `"[CLS]"`) — The classifier token which is used when doing sequence classification (classification of the whole sequence instead of per-token classification). It is the first token of the sequence when built with special tokens.
-   **mask\_token** (`str`, _optional_, defaults to `"[MASK]"`) — The token used for masking values. This is the token used when training this model with masked language modeling. This is the token which the model will try to predict.
-   **clean\_text** (`bool`, _optional_, defaults to `True`) — Whether or not to clean the text before tokenization by removing any control characters and replacing all whitespaces by the classic one.
-   **tokenize\_chinese\_chars** (`bool`, _optional_, defaults to `True`) — Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see [this issue](https://github.com/huggingface/transformers/issues/328)).
-   **strip\_accents** (`bool`, _optional_) — Whether or not to strip all accents. If this option is not specified, then it will be determined by the value for `lowercase` (as in the original LayoutLM).
-   **wordpieces\_prefix** (`str`, _optional_, defaults to `"##"`) — The prefix for subwords.

Construct a “fast” LayoutLM tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/tokenization_layoutlm_fast.py#L150)

( token\_ids\_0 token\_ids\_1 = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A LayoutLM sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/tokenization_layoutlm_fast.py#L174)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A LayoutLM

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

## LayoutLMModel

### class transformers.LayoutLMModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L712)

( config )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top. The LayoutLM model was proposed in [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and Ming Zhou.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L738)

( input\_ids: typing.Optional\[torch.LongTensor\] = None bbox: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None encoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = None encoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, _optional_) — Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1) format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the position of the lower right corner. See [Overview](#Overview) for normalization.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`: `1` for tokens that are NOT MASKED, `0` for MASKED tokens.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`: `0` corresponds to a _sentence A_ token, `1` corresponds to a _sentence B_ token
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`: `1` indicates the head is **not masked**, `0` indicates the head is **masked**.
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — If set to `True`, the attentions tensors of all attention layers are returned. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — If set to `True`, the hidden states of all layers are returned. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — If set to `True`, the model will return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **last\_hidden\_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    
-   **past\_key\_values** (`tuple(tuple(torch.FloatTensor))`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
    
    Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    

The [LayoutLMModel](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, LayoutLMModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
>>> model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

>>> words = ["Hello", "world"]
>>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

>>> token_boxes = []
>>> for word, box in zip(words, normalized_word_boxes):
...     word_tokens = tokenizer.tokenize(word)
...     token_boxes.extend([box] * len(word_tokens))
>>> 
>>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

>>> encoding = tokenizer(" ".join(words), return_tensors="pt")
>>> input_ids = encoding["input_ids"]
>>> attention_mask = encoding["attention_mask"]
>>> token_type_ids = encoding["token_type_ids"]
>>> bbox = torch.tensor([token_boxes])

>>> outputs = model(
...     input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
... )

>>> last_hidden_states = outputs.last_hidden_state
```

## LayoutLMForMaskedLM

### class transformers.LayoutLMForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L861)

( config )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LayoutLM Model with a `language modeling` head on top. The LayoutLM model was proposed in [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and Ming Zhou.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L882)

( input\_ids: typing.Optional\[torch.LongTensor\] = None bbox: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None encoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = None encoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, _optional_) — Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1) format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the position of the lower right corner. See [Overview](#Overview) for normalization.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`: `1` for tokens that are NOT MASKED, `0` for MASKED tokens.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`: `0` corresponds to a _sentence A_ token, `1` corresponds to a _sentence B_ token
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`: `1` indicates the head is **not masked**, `0` indicates the head is **masked**.
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — If set to `True`, the attentions tensors of all attention layers are returned. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — If set to `True`, the hidden states of all layers are returned. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — If set to `True`, the model will return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

A [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Masked language modeling (MLM) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [LayoutLMForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, LayoutLMForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
>>> model = LayoutLMForMaskedLM.from_pretrained("microsoft/layoutlm-base-uncased")

>>> words = ["Hello", "[MASK]"]
>>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

>>> token_boxes = []
>>> for word, box in zip(words, normalized_word_boxes):
...     word_tokens = tokenizer.tokenize(word)
...     token_boxes.extend([box] * len(word_tokens))
>>> 
>>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

>>> encoding = tokenizer(" ".join(words), return_tensors="pt")
>>> input_ids = encoding["input_ids"]
>>> attention_mask = encoding["attention_mask"]
>>> token_type_ids = encoding["token_type_ids"]
>>> bbox = torch.tensor([token_boxes])

>>> labels = tokenizer("Hello world", return_tensors="pt")["input_ids"]

>>> outputs = model(
...     input_ids=input_ids,
...     bbox=bbox,
...     attention_mask=attention_mask,
...     token_type_ids=token_type_ids,
...     labels=labels,
... )

>>> loss = outputs.loss
```

## LayoutLMForSequenceClassification

### class transformers.LayoutLMForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L992)

( config )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LayoutLM Model with a sequence classification head on top (a linear layer on top of the pooled output) e.g. for document image classification tasks such as the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.

The LayoutLM model was proposed in [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and Ming Zhou.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L1006)

( input\_ids: typing.Optional\[torch.LongTensor\] = None bbox: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, _optional_) — Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1) format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the position of the lower right corner. See [Overview](#Overview) for normalization.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`: `1` for tokens that are NOT MASKED, `0` for MASKED tokens.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`: `0` corresponds to a _sentence A_ token, `1` corresponds to a _sentence B_ token
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`: `1` indicates the head is **not masked**, `0` indicates the head is **masked**.
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — If set to `True`, the attentions tensors of all attention layers are returned. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — If set to `True`, the hidden states of all layers are returned. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — If set to `True`, the model will return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [LayoutLMForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, LayoutLMForSequenceClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
>>> model = LayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased")

>>> words = ["Hello", "world"]
>>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

>>> token_boxes = []
>>> for word, box in zip(words, normalized_word_boxes):
...     word_tokens = tokenizer.tokenize(word)
...     token_boxes.extend([box] * len(word_tokens))
>>> 
>>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

>>> encoding = tokenizer(" ".join(words), return_tensors="pt")
>>> input_ids = encoding["input_ids"]
>>> attention_mask = encoding["attention_mask"]
>>> token_type_ids = encoding["token_type_ids"]
>>> bbox = torch.tensor([token_boxes])
>>> sequence_label = torch.tensor([1])

>>> outputs = model(
...     input_ids=input_ids,
...     bbox=bbox,
...     attention_mask=attention_mask,
...     token_type_ids=token_type_ids,
...     labels=sequence_label,
... )

>>> loss = outputs.loss
>>> logits = outputs.logits
```

## LayoutLMForTokenClassification

### class transformers.LayoutLMForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L1129)

( config )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for sequence labeling (information extraction) tasks such as the [FUNSD](https://guillaumejaume.github.io/FUNSD/) dataset and the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset.

The LayoutLM model was proposed in [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and Ming Zhou.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L1143)

( input\_ids: typing.Optional\[torch.LongTensor\] = None bbox: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **bbox** (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, _optional_) — Bounding boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1) format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the position of the lower right corner. See [Overview](#Overview) for normalization.
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`: `1` for tokens that are NOT MASKED, `0` for MASKED tokens.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`: `0` corresponds to a _sentence A_ token, `1` corresponds to a _sentence B_ token
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`: `1` indicates the head is **not masked**, `0` indicates the head is **masked**.
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert _input\_ids_ indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — If set to `True`, the attentions tensors of all attention layers are returned. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — If set to `True`, the hidden states of all layers are returned. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — If set to `True`, the model will return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

A [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [LayoutLMForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, LayoutLMForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
>>> model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

>>> words = ["Hello", "world"]
>>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

>>> token_boxes = []
>>> for word, box in zip(words, normalized_word_boxes):
...     word_tokens = tokenizer.tokenize(word)
...     token_boxes.extend([box] * len(word_tokens))
>>> 
>>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

>>> encoding = tokenizer(" ".join(words), return_tensors="pt")
>>> input_ids = encoding["input_ids"]
>>> attention_mask = encoding["attention_mask"]
>>> token_type_ids = encoding["token_type_ids"]
>>> bbox = torch.tensor([token_boxes])
>>> token_labels = torch.tensor([1, 1, 0, 0]).unsqueeze(0)  

>>> outputs = model(
...     input_ids=input_ids,
...     bbox=bbox,
...     attention_mask=attention_mask,
...     token_type_ids=token_type_ids,
...     labels=token_labels,
... )

>>> loss = outputs.loss
>>> logits = outputs.logits
```

## LayoutLMForQuestionAnswering

### class transformers.LayoutLMForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L1247)

( config has\_visual\_segment\_embedding = True )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

LayoutLM Model with a span classification head on top for extractive question-answering tasks such as [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the final hidden-states output to compute `span start logits` and `span end logits`).

The LayoutLM model was proposed in [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and Ming Zhou.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_layoutlm.py#L1261)

( input\_ids: typing.Optional\[torch.LongTensor\] = None bbox: typing.Optional\[torch.LongTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None position\_ids: typing.Optional\[torch.LongTensor\] = None head\_mask: typing.Optional\[torch.FloatTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None start\_positions: typing.Optional\[torch.LongTensor\] = None end\_positions: typing.Optional\[torch.LongTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

A [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **loss** (`torch.FloatTensor` of shape `(1,)`, _optional_, returned when `labels` is provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    
-   **start\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
    
-   **end\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
    
-   **hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    
-   **attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

start\_positions (`torch.LongTensor` of shape `(batch_size,)`, _optional_): Labels for position (index) of the start of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss. end\_positions (`torch.LongTensor` of shape `(batch_size,)`, _optional_): Labels for position (index) of the end of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss.

Example:

In the example below, we prepare a question + context pair for the LayoutLM model. It will give us a prediction of what it thinks the answer is (the span of the answer within the texts parsed from the image).

```
>>> from transformers import AutoTokenizer, LayoutLMForQuestionAnswering
>>> from datasets import load_dataset
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)
>>> model = LayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="1e3ebac")

>>> dataset = load_dataset("nielsr/funsd", split="train")
>>> example = dataset[0]
>>> question = "what's his name?"
>>> words = example["words"]
>>> boxes = example["bboxes"]

>>> encoding = tokenizer(
...     question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="pt"
... )
>>> bbox = []
>>> for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
...     if s == 1:
...         bbox.append(boxes[w])
...     elif i == tokenizer.sep_token_id:
...         bbox.append([1000] * 4)
...     else:
...         bbox.append([0] * 4)
>>> encoding["bbox"] = torch.tensor([bbox])

>>> word_ids = encoding.word_ids(0)
>>> outputs = model(**encoding)
>>> loss = outputs.loss
>>> start_scores = outputs.start_logits
>>> end_scores = outputs.end_logits
>>> start, end = word_ids[start_scores.argmax(-1)], word_ids[end_scores.argmax(-1)]
>>> print(" ".join(words[start : end + 1]))
M. Hamann P. Harper, P. Martinez
```

## TFLayoutLMModel

### class transformers.TFLayoutLMModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L912)

( \*args \*\*kwargs )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L918)

( input\_ids: TFModelInputType | None = None bbox: np.ndarray | tf.Tensor | None = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None encoder\_hidden\_states: np.ndarray | tf.Tensor | None = None encoder\_attention\_mask: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **bbox** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length, 4)`, _optional_) — Bounding Boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings- 1]`.
-   **attention\_mask** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

A [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **last\_hidden\_state** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the model.
    
-   **pooler\_output** (`tf.Tensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
    
    This output is usually _not_ a good summary of the semantic content of the input, you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
    
-   **past\_key\_values** (`List[tf.Tensor]`, _optional_, returned when `use_cache=True` is passed or when `config.use_cache=True`) — List of `tf.Tensor` of length `config.n_layers`, with each tensor of shape `(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).
    
    Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    
-   **cross\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights of the decoder’s cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
    

The [TFLayoutLMModel](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.TFLayoutLMModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TFLayoutLMModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
>>> model = TFLayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")

>>> words = ["Hello", "world"]
>>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

>>> token_boxes = []
>>> for word, box in zip(words, normalized_word_boxes):
...     word_tokens = tokenizer.tokenize(word)
...     token_boxes.extend([box] * len(word_tokens))
>>> 
>>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

>>> encoding = tokenizer(" ".join(words), return_tensors="tf")
>>> input_ids = encoding["input_ids"]
>>> attention_mask = encoding["attention_mask"]
>>> token_type_ids = encoding["token_type_ids"]
>>> bbox = tf.convert_to_tensor([token_boxes])

>>> outputs = model(
...     input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids
... )

>>> last_hidden_states = outputs.last_hidden_state
```

## TFLayoutLMForMaskedLM

### class transformers.TFLayoutLMForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L991)

( \*args \*\*kwargs )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

LayoutLM Model with a `language modeling` head on top.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1019)

( input\_ids: TFModelInputType | None = None bbox: np.ndarray | tf.Tensor | None = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **bbox** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length, 4)`, _optional_) — Bounding Boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings- 1]`.
-   **attention\_mask** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

A [transformers.modeling\_tf\_outputs.TFMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(n,)`, _optional_, where n is the number of non-masked labels, returned when `labels` is provided) — Masked language modeling (MLM) loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFLayoutLMForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.TFLayoutLMForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TFLayoutLMForMaskedLM
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
>>> model = TFLayoutLMForMaskedLM.from_pretrained("microsoft/layoutlm-base-uncased")

>>> words = ["Hello", "[MASK]"]
>>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

>>> token_boxes = []
>>> for word, box in zip(words, normalized_word_boxes):
...     word_tokens = tokenizer.tokenize(word)
...     token_boxes.extend([box] * len(word_tokens))
>>> 
>>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

>>> encoding = tokenizer(" ".join(words), return_tensors="tf")
>>> input_ids = encoding["input_ids"]
>>> attention_mask = encoding["attention_mask"]
>>> token_type_ids = encoding["token_type_ids"]
>>> bbox = tf.convert_to_tensor([token_boxes])

>>> labels = tokenizer("Hello world", return_tensors="tf")["input_ids"]

>>> outputs = model(
...     input_ids=input_ids,
...     bbox=bbox,
...     attention_mask=attention_mask,
...     token_type_ids=token_type_ids,
...     labels=labels,
... )

>>> loss = outputs.loss
```

## TFLayoutLMForSequenceClassification

### class transformers.TFLayoutLMForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1118)

( \*args \*\*kwargs )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

LayoutLM Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1136)

( input\_ids: TFModelInputType | None = None bbox: np.ndarray | tf.Tensor | None = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **bbox** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length, 4)`, _optional_) — Bounding Boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings- 1]`.
-   **attention\_mask** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, _optional_) — Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

A [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(batch_size, )`, _optional_, returned when `labels` is provided) — Classification (or regression if config.num\_labels==1) loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, config.num_labels)`) — Classification (or regression if config.num\_labels==1) scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFLayoutLMForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.TFLayoutLMForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, TFLayoutLMForSequenceClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
>>> model = TFLayoutLMForSequenceClassification.from_pretrained("microsoft/layoutlm-base-uncased")

>>> words = ["Hello", "world"]
>>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

>>> token_boxes = []
>>> for word, box in zip(words, normalized_word_boxes):
...     word_tokens = tokenizer.tokenize(word)
...     token_boxes.extend([box] * len(word_tokens))
>>> 
>>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

>>> encoding = tokenizer(" ".join(words), return_tensors="tf")
>>> input_ids = encoding["input_ids"]
>>> attention_mask = encoding["attention_mask"]
>>> token_type_ids = encoding["token_type_ids"]
>>> bbox = tf.convert_to_tensor([token_boxes])
>>> sequence_label = tf.convert_to_tensor([1])

>>> outputs = model(
...     input_ids=input_ids,
...     bbox=bbox,
...     attention_mask=attention_mask,
...     token_type_ids=token_type_ids,
...     labels=sequence_label,
... )

>>> loss = outputs.loss
>>> logits = outputs.logits
```

## TFLayoutLMForTokenClassification

### class transformers.TFLayoutLMForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1236)

( \*args \*\*kwargs )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1260)

( input\_ids: TFModelInputType | None = None bbox: np.ndarray | tf.Tensor | None = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None labels: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFTokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **bbox** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length, 4)`, _optional_) — Bounding Boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings- 1]`.
-   **attention\_mask** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **labels** (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

A [transformers.modeling\_tf\_outputs.TFTokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(n,)`, _optional_, where n is the number of unmasked labels, returned when `labels` is provided) — Classification loss.
    
-   **logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.num_labels)`) — Classification scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFLayoutLMForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.TFLayoutLMForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import tensorflow as tf
>>> from transformers import AutoTokenizer, TFLayoutLMForTokenClassification

>>> tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
>>> model = TFLayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

>>> words = ["Hello", "world"]
>>> normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]

>>> token_boxes = []
>>> for word, box in zip(words, normalized_word_boxes):
...     word_tokens = tokenizer.tokenize(word)
...     token_boxes.extend([box] * len(word_tokens))
>>> 
>>> token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

>>> encoding = tokenizer(" ".join(words), return_tensors="tf")
>>> input_ids = encoding["input_ids"]
>>> attention_mask = encoding["attention_mask"]
>>> token_type_ids = encoding["token_type_ids"]
>>> bbox = tf.convert_to_tensor([token_boxes])
>>> token_labels = tf.convert_to_tensor([1, 1, 0, 0])

>>> outputs = model(
...     input_ids=input_ids,
...     bbox=bbox,
...     attention_mask=attention_mask,
...     token_type_ids=token_type_ids,
...     labels=token_labels,
... )

>>> loss = outputs.loss
>>> logits = outputs.logits
```

## TFLayoutLMForQuestionAnswering

### class transformers.TFLayoutLMForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1359)

( \*args \*\*kwargs )

Parameters

-   **config** ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

LayoutLM Model with a span classification head on top for extractive question-answering tasks such as [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the final hidden-states output to compute `span start logits` and `span end logits`).

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/layoutlm/modeling_tf_layoutlm.py#L1380)

( input\_ids: TFModelInputType | None = None bbox: np.ndarray | tf.Tensor | None = None attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None position\_ids: np.ndarray | tf.Tensor | None = None head\_mask: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None start\_positions: np.ndarray | tf.Tensor | None = None end\_positions: np.ndarray | tf.Tensor | None = None training: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **bbox** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length, 4)`, _optional_) — Bounding Boxes of each input sequence tokens. Selected in the range `[0, config.max_2d_position_embeddings- 1]`.
-   **attention\_mask** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **position\_ids** (`Numpy array` or `tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.
    
    [What are position IDs?](../glossary#position-ids)
    
-   **head\_mask** (`Numpy array` or `tf.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, _optional_) — Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
    -   1 indicates the head is **not masked**,
    -   0 indicates the head is **masked**.
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **start\_positions** (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the start of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss.
-   **end\_positions** (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, _optional_) — Labels for position (index) of the end of the labelled span for computing the token classification loss. Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence are not taken into account for computing the loss.

A [transformers.modeling\_tf\_outputs.TFQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFQuestionAnsweringModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LayoutLMConfig](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.LayoutLMConfig)) and inputs.

-   **loss** (`tf.Tensor` of shape `(batch_size, )`, _optional_, returned when `start_positions` and `end_positions` are provided) — Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
    
-   **start\_logits** (`tf.Tensor` of shape `(batch_size, sequence_length)`) — Span-start scores (before SoftMax).
    
-   **end\_logits** (`tf.Tensor` of shape `(batch_size, sequence_length)`) — Span-end scores (before SoftMax).
    
-   **hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
    
    Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    
-   **attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
    Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    

The [TFLayoutLMForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/layoutlm#transformers.TFLayoutLMForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import tensorflow as tf
>>> from transformers import AutoTokenizer, TFLayoutLMForQuestionAnswering
>>> from datasets import load_dataset

>>> tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)
>>> model = TFLayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="1e3ebac")

>>> dataset = load_dataset("nielsr/funsd", split="train")
>>> example = dataset[0]
>>> question = "what's his name?"
>>> words = example["words"]
>>> boxes = example["bboxes"]

>>> encoding = tokenizer(
...     question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="tf"
... )
>>> bbox = []
>>> for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
...     if s == 1:
...         bbox.append(boxes[w])
...     elif i == tokenizer.sep_token_id:
...         bbox.append([1000] * 4)
...     else:
...         bbox.append([0] * 4)
>>> encoding["bbox"] = tf.convert_to_tensor([bbox])

>>> word_ids = encoding.word_ids(0)
>>> outputs = model(**encoding)
>>> loss = outputs.loss
>>> start_scores = outputs.start_logits
>>> end_scores = outputs.end_logits
>>> start, end = word_ids[tf.math.argmax(start_scores, -1)[0]], word_ids[tf.math.argmax(end_scores, -1)[0]]
>>> print(" ".join(words[start : end + 1]))
M. Hamann P. Harper, P. Martinez
```