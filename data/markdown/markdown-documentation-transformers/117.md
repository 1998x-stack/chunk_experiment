# BROS

## Overview

The BROS model was proposed in [BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents](https://arxiv.org/abs/2108.04539) by Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park.

BROS stands for _BERT Relying On Spatiality_. It is an encoder-only Transformer model that takes a sequence of tokens and their bounding boxes as inputs and outputs a sequence of hidden states. BROS encode relative spatial information instead of using absolute spatial information.

It is pre-trained with two objectives: a token-masked language modeling objective (TMLM) used in BERT, and a novel area-masked language modeling objective (AMLM) In TMLM, tokens are randomly masked, and the model predicts the masked tokens using spatial information and other unmasked tokens. AMLM is a 2D version of TMLM. It randomly masks text tokens and predicts with the same information as TMLM, but it masks text blocks (areas).

`BrosForTokenClassification` has a simple linear layer on top of BrosModel. It predicts the label of each token. `BrosSpadeEEForTokenClassification` has an `initial_token_classifier` and `subsequent_token_classifier` on top of BrosModel. `initial_token_classifier` is used to predict the first token of each entity, and `subsequent_token_classifier` is used to predict the next token of within entity. `BrosSpadeELForTokenClassification` has an `entity_linker` on top of BrosModel. `entity_linker` is used to predict the relation between two entities.

`BrosForTokenClassification` and `BrosSpadeEEForTokenClassification` essentially perform the same job. However, `BrosForTokenClassification` assumes input tokens are perfectly serialized (which is very challenging task since they exist in a 2D space), while `BrosSpadeEEForTokenClassification` allows for more flexibility in handling serialization errors as it predicts next connection tokens from one token.

`BrosSpadeELForTokenClassification` perform the intra-entity linking task. It predicts relation from one token (of one entity) to another token (of another entity) if these two entities share some relation.

BROS achieves comparable or better result on Key Information Extraction (KIE) benchmarks such as FUNSD, SROIE, CORD and SciTSR, without relying on explicit visual features.

The abstract from the paper is the following:

_Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space. Many recent studies try to solve the task by developing pre-trained language models focusing on combining visual features from document images with texts and their layout. On the other hand, this paper tackles the problem by going back to the basic: effective combination of text and layout. Specifically, we propose a pre-trained language model, named BROS (BERT Relying On Spatiality), that encodes relative positions of texts in 2D space and learns from unlabeled documents with area-masking strategy. With this optimized training scheme for understanding texts in 2D space, BROS shows comparable or better performance compared to previous methods on four KIE benchmarks (FUNSD, SROIE_, CORD, and SciTSR) without relying on visual features. This paper also reveals two real-world challenges in KIE tasks-(1) minimizing the error from incorrect text ordering and (2) efficient learning from fewer downstream examples-and demonstrates the superiority of BROS over previous methods.\*

Tips:

-   [forward()](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosModel.forward) requires `input_ids` and `bbox` (bounding box). Each bounding box should be in (x0, y0, x1, y1) format (top-left corner, bottom-right corner). Obtaining of Bounding boxes depends on external OCR system. The `x` coordinate should be normalized by document image width, and the `y` coordinate should be normalized by document image height.

```
def expand_and_normalize_bbox(bboxes, doc_width, doc_height):
    

    
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] / width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] / height
```

-   \[`~transformers.BrosForTokenClassification.forward`, `~transformers.BrosSpadeEEForTokenClassification.forward`, `~transformers.BrosSpadeEEForTokenClassification.forward`\] require not only `input_ids` and `bbox` but also `box_first_token_mask` for loss calculation. It is a mask to filter out non-first tokens of each box. You can obtain this mask by saving start token indices of bounding boxes when creating `input_ids` from words. You can make `box_first_token_mask` with following code,

```
def make_box_first_token_mask(bboxes, words, tokenizer, max_seq_length=512):

    box_first_token_mask = np.zeros(max_seq_length, dtype=np.bool_)

    
    input_ids_list: List[List[int]] = [tokenizer.encode(e, add_special_tokens=False) for e in words]

    
    tokens_length_list: List[int] = [len(l) for l in input_ids_list]

    box_end_token_indices = np.array(list(itertools.accumulate(tokens_length_list)))
    box_start_token_indices = box_end_token_indices - np.array(tokens_length_list)

    
    box_end_token_indices = box_end_token_indices[box_end_token_indices < max_seq_length - 1]
    if len(box_start_token_indices) > len(box_end_token_indices):
        box_start_token_indices = box_start_token_indices[: len(box_end_token_indices)]

    
    box_first_token_mask[box_start_token_indices] = True

    return box_first_token_mask

```

-   Demo scripts can be found [here](https://github.com/clovaai/bros).

This model was contributed by [jinho8345](https://huggingface.co/jinho8345). The original code can be found [here](https://github.com/clovaai/bros).

## BrosConfig

### class transformers.BrosConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/configuration_bros.py#L29)

( vocab\_size = 30522hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512type\_vocab\_size = 2initializer\_range = 0.02layer\_norm\_eps = 1e-12pad\_token\_id = 0dim\_bbox = 8bbox\_scale = 100.0n\_relations = 1classifier\_dropout\_prob = 0.1\*\*kwargs )

This is the configuration class to store the configuration of a [BrosModel](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosModel) or a `TFBrosModel`. It is used to instantiate a Bros model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Bros [jinho8345/bros-base-uncased](https://huggingface.co/jinho8345/bros-base-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import BrosConfig, BrosModel

>>> 
>>> configuration = BrosConfig()

>>> 
>>> model = BrosModel(configuration)

>>> 
>>> configuration = model.config
```

## BrosProcessor

### class transformers.BrosProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/processing_bros.py#L26)

( tokenizer = None\*\*kwargs )

Parameters

-   **tokenizer** (`BertTokenizerFast`) — An instance of \[‘BertTokenizerFast\`\]. The tokenizer is a required input.

Constructs a Bros processor which wraps a BERT tokenizer.

[BrosProcessor](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosProcessor) offers all the functionalities of [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast). See the docstring of [**call**()](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosProcessor.__call__) and `decode()` for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/processing_bros.py#L46)

( text: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonestride: int = 0pad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_token\_type\_ids: typing.Optional\[bool\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = Truereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None\*\*kwargs )

This method uses [BertTokenizerFast.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) to prepare text for the model.

Please refer to the docstring of the above two methods for more information.

## BrosModel

### class transformers.BrosModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/modeling_bros.py#L794)

( configadd\_pooling\_layer = True )

Parameters

-   **config** ([BrosConfig](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Bros Model transformer outputting raw hidden-states without any specific head on top. This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/modeling_bros.py#L821)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonebbox: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

The [BrosModel](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import torch
>>> from transformers import BrosProcessor, BrosModel

>>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

>>> model = BrosModel.from_pretrained("jinho8345/bros-base-uncased")

>>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
>>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
>>> encoding["bbox"] = bbox

>>> outputs = model(**encoding)
>>> last_hidden_states = outputs.last_hidden_state
```

## BrosForTokenClassification

### class transformers.BrosForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/modeling_bros.py#L972)

( config )

Parameters

-   **config** ([BrosConfig](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bros Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/modeling_bros.py#L988)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonebbox: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonebbox\_first\_token\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [BrosForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import torch
>>> from transformers import BrosProcessor, BrosForTokenClassification

>>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

>>> model = BrosForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

>>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
>>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
>>> encoding["bbox"] = bbox

>>> outputs = model(**encoding)
```

## BrosSpadeEEForTokenClassification

### class transformers.BrosSpadeEEForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/modeling_bros.py#L1079)

( config )

Parameters

-   **config** ([BrosConfig](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bros Model with a token classification head on top (initial\_token\_layers and subsequent\_token\_layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. The initial\_token\_classifier is used to predict the first token of each entity, and the subsequent\_token\_classifier is used to predict the subsequent tokens within an entity. Compared to BrosForTokenClassification, this model is more robust to serialization errors since it predicts next token from one token.

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/modeling_bros.py#L1107)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonebbox: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonebbox\_first\_token\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneinitial\_token\_labels: typing.Optional\[torch.Tensor\] = Nonesubsequent\_token\_labels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.bros.modeling_bros.BrosSpadeOutput` or `tuple(torch.FloatTensor)`

The [BrosSpadeEEForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosSpadeEEForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import torch
>>> from transformers import BrosProcessor, BrosSpadeEEForTokenClassification

>>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

>>> model = BrosSpadeEEForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

>>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
>>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
>>> encoding["bbox"] = bbox

>>> outputs = model(**encoding)
```

## BrosSpadeELForTokenClassification

### class transformers.BrosSpadeELForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/modeling_bros.py#L1222)

( config )

Parameters

-   **config** ([BrosConfig](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bros Model with a token classification head on top (a entity\_linker layer on top of the hidden-states output) e.g. for Entity-Linking. The entity\_linker is used to predict intra-entity links (one entity to another entity).

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bros/modeling_bros.py#L1239)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonebbox: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonebbox\_first\_token\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [BrosSpadeELForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/bros#transformers.BrosSpadeELForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import torch
>>> from transformers import BrosProcessor, BrosSpadeELForTokenClassification

>>> processor = BrosProcessor.from_pretrained("jinho8345/bros-base-uncased")

>>> model = BrosSpadeELForTokenClassification.from_pretrained("jinho8345/bros-base-uncased")

>>> encoding = processor("Hello, my dog is cute", add_special_tokens=False, return_tensors="pt")
>>> bbox = torch.tensor([[[0, 0, 1, 1]]]).repeat(1, encoding["input_ids"].shape[-1], 1)
>>> encoding["bbox"] = bbox

>>> outputs = model(**encoding)
```