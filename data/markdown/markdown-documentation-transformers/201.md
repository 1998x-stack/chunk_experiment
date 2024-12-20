# LXMERT

## Overview

The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan & Mohit Bansal. It is a series of bidirectional transformer encoders (one for the vision modality, one for the language modality, and then one to fuse both modalities) pretrained using a combination of masked language modeling, visual-language text alignment, ROI-feature regression, masked visual-attribute modeling, masked visual-object modeling, and visual-question answering objectives. The pretraining consists of multiple multi-modal datasets: MSCOCO, Visual-Genome + Visual-Genome Question Answering, VQA 2.0, and GQA.

The abstract from the paper is the following:

_Vision-and-language reasoning requires an understanding of visual concepts, language semantics, and, most importantly, the alignment and relationships between these two modalities. We thus propose the LXMERT (Learning Cross-Modality Encoder Representations from Transformers) framework to learn these vision-and-language connections. In LXMERT, we build a large-scale Transformer model that consists of three encoders: an object relationship encoder, a language encoder, and a cross-modality encoder. Next, to endow our model with the capability of connecting vision and language semantics, we pre-train the model with large amounts of image-and-sentence pairs, via five diverse representative pretraining tasks: masked language modeling, masked object prediction (feature regression and label classification), cross-modality matching, and image question answering. These tasks help in learning both intra-modality and cross-modality relationships. After fine-tuning from our pretrained parameters, our model achieves the state-of-the-art results on two visual question answering datasets (i.e., VQA and GQA). We also show the generalizability of our pretrained cross-modality model by adapting it to a challenging visual-reasoning task, NLVR, and improve the previous best result by 22% absolute (54% to 76%). Lastly, we demonstrate detailed ablation studies to prove that both our novel model components and pretraining strategies significantly contribute to our strong results; and also present several attention visualizations for the different encoders_

Tips:

-   Bounding boxes are not necessary to be used in the visual feature embeddings, any kind of visual-spacial features will work.
-   Both the language hidden states and the visual hidden states that LXMERT outputs are passed through the cross-modality layer, so they contain information from both modalities. To access a modality that only attends to itself, select the vision/language hidden states from the first input in the tuple.
-   The bidirectional cross-modality encoder attention only returns attention values when the language modality is used as the input and the vision modality is used as the context vector. Further, while the cross-modality encoder contains self-attention for each respective modality and cross-attention, only the cross attention is returned and both self attention outputs are disregarded.

This model was contributed by [eltoto1219](https://huggingface.co/eltoto1219). The original code can be found [here](https://github.com/airsplay/lxmert).

## Documentation resources

-   [Question answering task guide](../tasks/question_answering)

## LxmertConfig

### class transformers.LxmertConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/configuration_lxmert.py#L29)

( vocab\_size = 30522 hidden\_size = 768 num\_attention\_heads = 12 num\_qa\_labels = 9500 num\_object\_labels = 1600 num\_attr\_labels = 400 intermediate\_size = 3072 hidden\_act = 'gelu' hidden\_dropout\_prob = 0.1 attention\_probs\_dropout\_prob = 0.1 max\_position\_embeddings = 512 type\_vocab\_size = 2 initializer\_range = 0.02 layer\_norm\_eps = 1e-12 l\_layers = 9 x\_layers = 5 r\_layers = 5 visual\_feat\_dim = 2048 visual\_pos\_dim = 4 visual\_loss\_normalizer = 6.67 task\_matched = True task\_mask\_lm = True task\_obj\_predict = True task\_qa = True visual\_obj\_loss = True visual\_attr\_loss = True visual\_feat\_loss = True \*\*kwargs )

Parameters

-   **vocab\_size** (`int`, _optional_, defaults to 30522) — Vocabulary size of the LXMERT model. Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling [LxmertModel](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertModel) or [TFLxmertModel](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.TFLxmertModel).
-   **hidden\_size** (`int`, _optional_, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
-   **r\_layers** (`int`, _optional_, defaults to 5) — Number of hidden layers in the Transformer visual encoder.
-   **l\_layers** (`int`, _optional_, defaults to 9) — Number of hidden layers in the Transformer language encoder.
-   **x\_layers** (`int`, _optional_, defaults to 5) — Number of hidden layers in the Transformer cross modality encoder.
-   **num\_attention\_heads** (`int`, _optional_, defaults to 5) — Number of attention heads for each attention layer in the Transformer encoder.
-   **intermediate\_size** (`int`, _optional_, defaults to 3072) — Dimensionality of the “intermediate” (often named feed-forward) layer in the Transformer encoder.
-   **hidden\_act** (`str` or `Callable`, _optional_, defaults to `"gelu"`) — The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
-   **hidden\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
-   **attention\_probs\_dropout\_prob** (`float`, _optional_, defaults to 0.1) — The dropout ratio for the attention probabilities.
-   **max\_position\_embeddings** (`int`, _optional_, defaults to 512) — The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
-   **type\_vocab\_size** (`int`, _optional_, defaults to 2) — The vocabulary size of the _token\_type\_ids_ passed into [BertModel](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertModel).
-   **initializer\_range** (`float`, _optional_, defaults to 0.02) — The standard deviation of the truncated\_normal\_initializer for initializing all weight matrices.
-   **layer\_norm\_eps** (`float`, _optional_, defaults to 1e-12) — The epsilon used by the layer normalization layers.
-   **visual\_feat\_dim** (`int`, _optional_, defaults to 2048) — This represents the last dimension of the pooled-object features used as input for the model, representing the size of each object feature itself.
-   **visual\_pos\_dim** (`int`, _optional_, defaults to 4) — This represents the number of spacial features that are mixed into the visual features. The default is set to 4 because most commonly this will represent the location of a bounding box. i.e., (x, y, width, height)
-   **visual\_loss\_normalizer** (`float`, _optional_, defaults to 1/15) — This represents the scaling factor in which each visual loss is multiplied by if during pretraining, one decided to train with multiple vision-based loss objectives.
-   **num\_qa\_labels** (`int`, _optional_, defaults to 9500) — This represents the total number of different question answering (QA) labels there are. If using more than one dataset with QA, the user will need to account for the total number of labels that all of the datasets have in total.
-   **num\_object\_labels** (`int`, _optional_, defaults to 1600) — This represents the total number of semantically unique objects that lxmert will be able to classify a pooled-object feature as belonging too.
-   **num\_attr\_labels** (`int`, _optional_, defaults to 400) — This represents the total number of semantically unique attributes that lxmert will be able to classify a pooled-object feature as possessing.
-   **task\_matched** (`bool`, _optional_, defaults to `True`) — This task is used for sentence-image matching. If the sentence correctly describes the image the label will be 1. If the sentence does not correctly describe the image, the label will be 0.
-   **task\_mask\_lm** (`bool`, _optional_, defaults to `True`) — Whether or not to add masked language modeling (as used in pretraining models such as BERT) to the loss objective.
-   **task\_obj\_predict** (`bool`, _optional_, defaults to `True`) — Whether or not to add object prediction, attribute prediction and feature regression to the loss objective.
-   **task\_qa** (`bool`, _optional_, defaults to `True`) — Whether or not to add the question-answering loss to the objective
-   **visual\_obj\_loss** (`bool`, _optional_, defaults to `True`) — Whether or not to calculate the object-prediction loss objective
-   **visual\_attr\_loss** (`bool`, _optional_, defaults to `True`) — Whether or not to calculate the attribute-prediction loss objective
-   **visual\_feat\_loss** (`bool`, _optional_, defaults to `True`) — Whether or not to calculate the feature-regression loss objective
-   **output\_attentions** (`bool`, _optional_, defaults to `False`) — Whether or not the model should return the attentions from the vision, language, and cross-modality layers should be returned.
-   **output\_hidden\_states** (`bool`, _optional_, defaults to `False`) — Whether or not the model should return the hidden states from the vision, language, and cross-modality layers should be returned.

This is the configuration class to store the configuration of a [LxmertModel](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertModel) or a [TFLxmertModel](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.TFLxmertModel). It is used to instantiate a LXMERT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Lxmert [unc-nlp/lxmert-base-uncased](https://huggingface.co/unc-nlp/lxmert-base-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

## LxmertTokenizer

### class transformers.LxmertTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/tokenization_lxmert.py#L67)

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
    
-   **strip\_accents** (`bool`, _optional_) — Whether or not to strip all accents. If this option is not specified, then it will be determined by the value for `lowercase` (as in the original Lxmert).

Construct a Lxmert tokenizer. Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/tokenization_lxmert.py#L200)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A Lxmert sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

Converts a sequence of tokens (string) in a single string.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/tokenization_lxmert.py#L253)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Lxmert

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/tokenization_lxmert.py#L225)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None already\_has\_special\_tokens: bool = False ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

A list of integers in the range \[0, 1\]: 1 for a special token, 0 for a sequence token.

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` method.

## LxmertTokenizerFast

### class transformers.LxmertTokenizerFast

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/tokenization_lxmert_fast.py#L48)

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
-   **strip\_accents** (`bool`, _optional_) — Whether or not to strip all accents. If this option is not specified, then it will be determined by the value for `lowercase` (as in the original Lxmert).
-   **wordpieces\_prefix** (`str`, _optional_, defaults to `"##"`) — The prefix for subwords.

Construct a “fast” Lxmert tokenizer (backed by HuggingFace’s _tokenizers_ library). Based on WordPiece.

This tokenizer inherits from [PreTrainedTokenizerFast](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/tokenization_lxmert_fast.py#L136)

( token\_ids\_0 token\_ids\_1 = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A Lxmert sequence has the following format:

-   single sequence: `[CLS] X [SEP]`
-   pair of sequences: `[CLS] A [SEP] B [SEP]`

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/tokenization_lxmert_fast.py#L160)

( token\_ids\_0: typing.List\[int\] token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A Lxmert

sequence pair mask has the following format:

```
0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence    | second sequence |
```

If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

## Lxmert specific outputs

### class transformers.models.lxmert.modeling\_lxmert.LxmertModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_lxmert.py#L60)

( language\_output: typing.Optional\[torch.FloatTensor\] = None vision\_output: typing.Optional\[torch.FloatTensor\] = None pooled\_output: typing.Optional\[torch.FloatTensor\] = None language\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None vision\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None language\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None vision\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None cross\_encoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **language\_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the language encoder.
-   **vision\_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the visual encoder.
-   **pooled\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed by a Linear layer and a Tanh activation function. The Linear
-   **language\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Lxmert’s outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language, visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the “relation-ship” encoder”)

### class transformers.models.lxmert.modeling\_lxmert.LxmertForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_lxmert.py#L146)

( loss: typing.Optional\[torch.FloatTensor\] = None prediction\_logits: typing.Optional\[torch.FloatTensor\] = None cross\_relationship\_score: typing.Optional\[torch.FloatTensor\] = None question\_answering\_score: typing.Optional\[torch.FloatTensor\] = None language\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None vision\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None language\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None vision\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None cross\_encoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (_optional_, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
-   **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **cross\_relationship\_score** (`torch.FloatTensor` of shape `(batch_size, 2)`) — Prediction scores of the textual matching objective (classification) head (scores of True/False continuation before SoftMax).
-   **question\_answering\_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`) — Prediction scores of question answering objective (classification).
-   **language\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Output type of [LxmertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertForPreTraining).

### class transformers.models.lxmert.modeling\_lxmert.LxmertForQuestionAnsweringOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_lxmert.py#L106)

( loss: typing.Optional\[torch.FloatTensor\] = None question\_answering\_score: typing.Optional\[torch.FloatTensor\] = None language\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None vision\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None language\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None vision\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None cross\_encoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Parameters

-   **loss** (_optional_, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.k.
-   **question\_answering\_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`, _optional_) — Prediction scores of question answering objective (classification).
-   **language\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Output type of [LxmertForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering).

### class transformers.models.lxmert.modeling\_tf\_lxmert.TFLxmertModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_tf_lxmert.py#L61)

( language\_output: tf.Tensor | None = None vision\_output: tf.Tensor | None = None pooled\_output: tf.Tensor | None = None language\_hidden\_states: Tuple\[tf.Tensor\] | None = None vision\_hidden\_states: Tuple\[tf.Tensor\] | None = None language\_attentions: Tuple\[tf.Tensor\] | None = None vision\_attentions: Tuple\[tf.Tensor\] | None = None cross\_encoder\_attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **language\_output** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the language encoder.
-   **vision\_output** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the visual encoder.
-   **pooled\_output** (`tf.Tensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed by a Linear layer and a Tanh activation function. The Linear
-   **language\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Lxmert’s outputs that contain the last hidden states, pooled outputs, and attention probabilities for the language, visual, and, cross-modality encoders. (note: the visual encoder in Lxmert is referred to as the “relation-ship” encoder”)

### class transformers.models.lxmert.modeling\_tf\_lxmert.TFLxmertForPreTrainingOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_tf_lxmert.py#L107)

( loss: tf.Tensor | None = None prediction\_logits: tf.Tensor | None = None cross\_relationship\_score: tf.Tensor | None = None question\_answering\_score: tf.Tensor | None = None language\_hidden\_states: Tuple\[tf.Tensor\] | None = None vision\_hidden\_states: Tuple\[tf.Tensor\] | None = None language\_attentions: Tuple\[tf.Tensor\] | None = None vision\_attentions: Tuple\[tf.Tensor\] | None = None cross\_encoder\_attentions: Tuple\[tf.Tensor\] | None = None )

Parameters

-   **loss** (_optional_, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
-   **prediction\_logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **cross\_relationship\_score** (`tf.Tensor` of shape `(batch_size, 2)`) — Prediction scores of the textual matching objective (classification) head (scores of True/False continuation before SoftMax).
-   **question\_answering\_score** (`tf.Tensor` of shape `(batch_size, n_qa_answers)`) — Prediction scores of question answering objective (classification).
-   **language\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

Output type of [LxmertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertForPreTraining).

## LxmertModel

### class transformers.LxmertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_lxmert.py#L883)

( config )

Parameters

-   **config** ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.

The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It’s a vision and language transformer model, pretrained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MSCOCO captions, and Visual genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss for question answering attribute prediction, and object tag prediction.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_lxmert.py#L898)

( input\_ids: typing.Optional\[torch.LongTensor\] = None visual\_feats: typing.Optional\[torch.FloatTensor\] = None visual\_pos: typing.Optional\[torch.FloatTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None visual\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.lxmert.modeling\_lxmert.LxmertModelOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertModelOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **visual\_feats** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) — This input represents visual features. They ROI pooled object features from bounding boxes using a faster-RCNN model)
    
    These are currently not provided by the transformers library.
    
-   **visual\_pos** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`) — This input represents spacial features corresponding to their relative (via index) visual features. The pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to
    
    These are currently not provided by the transformers library.
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

A [transformers.models.lxmert.modeling\_lxmert.LxmertModelOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

-   **language\_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the language encoder.
-   **vision\_output** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the visual encoder.
-   **pooled\_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed by a Linear layer and a Tanh activation function. The Linear
-   **language\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The [LxmertModel](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LxmertModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
>>> model = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## LxmertForPreTraining

### class transformers.LxmertForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_lxmert.py#L1021)

( config )

Parameters

-   **config** ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Lxmert Model with a specified pretraining head on top.

The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It’s a vision and language transformer model, pretrained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MSCOCO captions, and Visual genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss for question answering attribute prediction, and object tag prediction.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_lxmert.py#L1150)

( input\_ids: typing.Optional\[torch.LongTensor\] = None visual\_feats: typing.Optional\[torch.FloatTensor\] = None visual\_pos: typing.Optional\[torch.FloatTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None visual\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.LongTensor\] = None obj\_labels: typing.Union\[typing.Dict\[str, typing.Tuple\[torch.FloatTensor, torch.FloatTensor\]\], NoneType\] = None matched\_label: typing.Optional\[torch.LongTensor\] = None ans: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None \*\*kwargs ) → [transformers.models.lxmert.modeling\_lxmert.LxmertForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **visual\_feats** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) — This input represents visual features. They ROI pooled object features from bounding boxes using a faster-RCNN model)
    
    These are currently not provided by the transformers library.
    
-   **visual\_pos** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`) — This input represents spacial features corresponding to their relative (via index) visual features. The pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to
    
    These are currently not provided by the transformers library.
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
-   **obj\_labels** (`Dict[Str -- Tuple[Torch.FloatTensor, Torch.FloatTensor]]`, _optional_): each key is named after each one of the visual losses and each element of the tuple is of the shape `(batch_size, num_features)` and `(batch_size, num_features, visual_feature_dim)` for each the label id and the label score respectively
-   **matched\_label** (`torch.LongTensor` of shape `(batch_size,)`, _optional_) — Labels for computing the whether or not the text input matches the image (classification) loss. Input should be a sequence pair (see `input_ids` docstring) Indices should be in `[0, 1]`:
    
    -   0 indicates that the sentence does not match the image,
    -   1 indicates that the sentence does match the image.
    
-   **ans** (`Torch.Tensor` of shape `(batch_size)`, _optional_) — a one hot representation hof the correct answer _optional_

A [transformers.models.lxmert.modeling\_lxmert.LxmertForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

-   **loss** (_optional_, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
-   **prediction\_logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **cross\_relationship\_score** (`torch.FloatTensor` of shape `(batch_size, 2)`) — Prediction scores of the textual matching objective (classification) head (scores of True/False continuation before SoftMax).
-   **question\_answering\_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`) — Prediction scores of question answering objective (classification).
-   **language\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The [LxmertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## LxmertForQuestionAnswering

### class transformers.LxmertForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_lxmert.py#L1289)

( config )

Parameters

-   **config** ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Lxmert Model with a visual-answering head on top for downstream QA tasks

The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It’s a vision and language transformer model, pretrained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MSCOCO captions, and Visual genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss for question answering attribute prediction, and object tag prediction.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_lxmert.py#L1381)

( input\_ids: typing.Optional\[torch.LongTensor\] = None visual\_feats: typing.Optional\[torch.FloatTensor\] = None visual\_pos: typing.Optional\[torch.FloatTensor\] = None attention\_mask: typing.Optional\[torch.FloatTensor\] = None visual\_attention\_mask: typing.Optional\[torch.FloatTensor\] = None token\_type\_ids: typing.Optional\[torch.LongTensor\] = None inputs\_embeds: typing.Optional\[torch.FloatTensor\] = None labels: typing.Optional\[torch.Tensor\] = None output\_attentions: typing.Optional\[bool\] = None output\_hidden\_states: typing.Optional\[bool\] = None return\_dict: typing.Optional\[bool\] = None ) → [transformers.models.lxmert.modeling\_lxmert.LxmertForQuestionAnsweringOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput) or `tuple(torch.FloatTensor)`

Parameters

-   **input\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) and [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **visual\_feats** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) — This input represents visual features. They ROI pooled object features from bounding boxes using a faster-RCNN model)
    
    These are currently not provided by the transformers library.
    
-   **visual\_pos** (`torch.FloatTensor` of shape `(batch_size, num_visual_features, visual_pos_dim)`) — This input represents spacial features corresponding to their relative (via index) visual features. The pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to
    
    These are currently not provided by the transformers library.
    
-   **attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **visual\_attention\_mask** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **inputs\_embeds** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.
-   **labels** (`Torch.Tensor` of shape `(batch_size)`, _optional_) — A one-hot representation of the correct answer

A [transformers.models.lxmert.modeling\_lxmert.LxmertForQuestionAnsweringOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

-   **loss** (_optional_, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.k.
-   **question\_answering\_score** (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`, _optional_) — Prediction scores of question answering objective (classification).
-   **language\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(torch.FloatTensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The [LxmertForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LxmertForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
>>> model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")

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

## TFLxmertModel

### class transformers.TFLxmertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_tf_lxmert.py#L927)

( \*args \*\*kwargs )

Parameters

-   **config** ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.

The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It’s a vision and language transformer model, pre-trained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss for question answering attribute prediction, and object tag prediction.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_tf_lxmert.py#L932)

( input\_ids: TFModelInputType | None = None visual\_feats: tf.Tensor | None = None visual\_pos: tf.Tensor | None = None attention\_mask: np.ndarray | tf.Tensor | None = None visual\_attention\_mask: np.ndarray | tf.Tensor | None = None token\_type\_ids: np.ndarray | tf.Tensor | None = None inputs\_embeds: np.ndarray | tf.Tensor | None = None output\_attentions: Optional\[bool\] = None output\_hidden\_states: Optional\[bool\] = None return\_dict: Optional\[bool\] = None training: bool = False ) → [transformers.models.lxmert.modeling\_tf\_lxmert.TFLxmertModelOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_tf_lxmert.TFLxmertModelOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **visual\_feats** (`tf.Tensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) — This input represents visual features. They ROI pooled object features from bounding boxes using a faster-RCNN model)
    
    These are currently not provided by the transformers library.
    
-   **visual\_pos** (`tf.Tensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) — This input represents spacial features corresponding to their relative (via index) visual features. The pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to
    
    These are currently not provided by the transformers library.
    
-   **attention\_mask** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **visual\_attention\_mask** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — MMask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).

A [transformers.models.lxmert.modeling\_tf\_lxmert.TFLxmertModelOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_tf_lxmert.TFLxmertModelOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

-   **language\_output** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the language encoder.
-   **vision\_output** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) — Sequence of hidden-states at the output of the last layer of the visual encoder.
-   **pooled\_output** (`tf.Tensor` of shape `(batch_size, hidden_size)`) — Last layer hidden-state of the first token of the sequence (classification, CLS, token) further processed by a Linear layer and a Tanh activation function. The Linear
-   **language\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The [TFLxmertModel](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.TFLxmertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFLxmertModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
>>> model = TFLxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFLxmertForPreTraining

### class transformers.TFLxmertForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_tf_lxmert.py#L1149)

( \*args \*\*kwargs )

Parameters

-   **config** ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Lxmert Model with a `language modeling` head on top.

The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It’s a vision and language transformer model, pre-trained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss for question answering attribute prediction, and object tag prediction.

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/lxmert/modeling_tf_lxmert.py#L1249)

( input\_ids: TFModelInputType | None = None visual\_feats: tf.Tensor | None = None visual\_pos: tf.Tensor | None = None attention\_mask: tf.Tensor | None = None visual\_attention\_mask: tf.Tensor | None = None token\_type\_ids: tf.Tensor | None = None inputs\_embeds: tf.Tensor | None = None masked\_lm\_labels: tf.Tensor | None = None obj\_labels: Dict\[str, Tuple\[tf.Tensor, tf.Tensor\]\] | None = None matched\_label: tf.Tensor | None = None ans: tf.Tensor | None = None output\_attentions: bool | None = None output\_hidden\_states: bool | None = None return\_dict: bool | None = None training: bool = False ) → [transformers.models.lxmert.modeling\_tf\_lxmert.TFLxmertForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_tf_lxmert.TFLxmertForPreTrainingOutput) or `tuple(tf.Tensor)`

Parameters

-   **input\_ids** (`np.ndarray` or `tf.Tensor` of shape `(batch_size, sequence_length)`) — Indices of input sequence tokens in the vocabulary.
    
    Indices can be obtained using [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) and [PreTrainedTokenizer.encode()](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast.encode) for details.
    
    [What are input IDs?](../glossary#input-ids)
    
-   **visual\_feats** (`tf.Tensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) — This input represents visual features. They ROI pooled object features from bounding boxes using a faster-RCNN model)
    
    These are currently not provided by the transformers library.
    
-   **visual\_pos** (`tf.Tensor` of shape `(batch_size, num_visual_features, visual_feat_dim)`) — This input represents spacial features corresponding to their relative (via index) visual features. The pre-trained LXMERT model expects these spacial features to be normalized bounding boxes on a scale of 0 to
    
    These are currently not provided by the transformers library.
    
-   **attention\_mask** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **visual\_attention\_mask** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — MMask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
    -   1 for tokens that are **not masked**,
    -   0 for tokens that are **masked**.
    
    [What are attention masks?](../glossary#attention-mask)
    
-   **token\_type\_ids** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
    
    -   0 corresponds to a _sentence A_ token,
    -   1 corresponds to a _sentence B_ token.
    
    [What are token type IDs?](../glossary#token-type-ids)
    
-   **inputs\_embeds** (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, _optional_) — Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model’s internal embedding lookup matrix.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail. This argument can be used only in eager mode, in graph mode the value in the config will be used instead.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple. This argument can be used in eager mode, in graph mode the value will always be set to True.
-   **training** (`bool`, _optional_, defaults to `False`) — Whether or not to use the model in training mode (some modules like dropout modules have different behaviors between training and evaluation).
-   **masked\_lm\_labels** (`tf.Tensor` of shape `(batch_size, sequence_length)`, _optional_) — Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
-   **obj\_labels** (`Dict[Str -- Tuple[tf.Tensor, tf.Tensor]]`, _optional_, defaults to `None`): each key is named after each one of the visual losses and each element of the tuple is of the shape `(batch_size, num_features)` and `(batch_size, num_features, visual_feature_dim)` for each the label id and the label score respectively
-   **matched\_label** (`tf.Tensor` of shape `(batch_size,)`, _optional_) — Labels for computing the whether or not the text input matches the image (classification) loss. Input should be a sequence pair (see `input_ids` docstring) Indices should be in `[0, 1]`:
    
    -   0 indicates that the sentence does not match the image,
    -   1 indicates that the sentence does match the image.
    
-   **ans** (`tf.Tensor` of shape `(batch_size)`, _optional_, defaults to `None`) — a one hot representation hof the correct answer _optional_

A [transformers.models.lxmert.modeling\_tf\_lxmert.TFLxmertForPreTrainingOutput](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.models.lxmert.modeling_tf_lxmert.TFLxmertForPreTrainingOutput) or a tuple of `tf.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([LxmertConfig](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.LxmertConfig)) and inputs.

-   **loss** (_optional_, returned when `labels` is provided, `tf.Tensor` of shape `(1,)`) — Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
-   **prediction\_logits** (`tf.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) — Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
-   **cross\_relationship\_score** (`tf.Tensor` of shape `(batch_size, 2)`) — Prediction scores of the textual matching objective (classification) head (scores of True/False continuation before SoftMax).
-   **question\_answering\_score** (`tf.Tensor` of shape `(batch_size, n_qa_answers)`) — Prediction scores of question answering objective (classification).
-   **language\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **vision\_hidden\_states** (`tuple(tf.Tensor)`, _optional_, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) — Tuple of `tf.Tensor` (one for input features + one for the output of each cross-modality layer) of shape `(batch_size, sequence_length, hidden_size)`.
-   **language\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **vision\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
-   **cross\_encoder\_attentions** (`tuple(tf.Tensor)`, _optional_, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) — Tuple of `tf.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

The [TFLxmertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/lxmert#transformers.TFLxmertForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.