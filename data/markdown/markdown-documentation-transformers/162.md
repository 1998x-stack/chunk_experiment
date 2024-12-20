# ESM

## Overview

This page provides code and pre-trained weights for Transformer protein language models from Meta AI's Fundamental AI Research Team, providing the state-of-the-art ESMFold and ESM-2, and the previously released ESM-1b and ESM-1v. Transformer protein language models were introduced in the paper \[Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences\](https://www.pnas.org/content/118/15/e2016239118) by Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, and Rob Fergus. The first version of this paper was \[preprinted in 2019\](https://www.biorxiv.org/content/10.1101/622803v1?versioned=true).

ESM-2 outperforms all tested single-sequence protein language models across a range of structure prediction tasks, and enables atomic resolution structure prediction. It was released with the paper [Language models of protein sequences at the scale of evolution enable accurate structure prediction](https://doi.org/10.1101/2022.07.20.500902) by Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Allan dos Santos Costa, Maryam Fazel-Zarandi, Tom Sercu, Sal Candido and Alexander Rives.

Also introduced in this paper was ESMFold. It uses an ESM-2 stem with a head that can predict folded protein structures with state-of-the-art accuracy. Unlike [AlphaFold2](https://www.nature.com/articles/s41586-021-03819-2), it relies on the token embeddings from the large pre-trained protein language model stem and does not perform a multiple sequence alignment (MSA) step at inference time, which means that ESMFold checkpoints are fully “standalone” - they do not require a database of known protein sequences and structures with associated external query tools to make predictions, and are much faster as a result.

The abstract from “Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences” is

_In the field of artificial intelligence, a combination of scale in data and model capacity enabled by unsupervised learning has led to major advances in representation learning and statistical generation. In the life sciences, the anticipated growth of sequencing promises unprecedented data on natural sequence diversity. Protein language modeling at the scale of evolution is a logical step toward predictive and generative artificial intelligence for biology. To this end, we use unsupervised learning to train a deep contextual language model on 86 billion amino acids across 250 million protein sequences spanning evolutionary diversity. The resulting model contains information about biological properties in its representations. The representations are learned from sequence data alone. The learned representation space has a multiscale organization reflecting structure from the level of biochemical properties of amino acids to remote homology of proteins. Information about secondary and tertiary structure is encoded in the representations and can be identified by linear projections. Representation learning produces features that generalize across a range of applications, enabling state-of-the-art supervised prediction of mutational effect and secondary structure and improving state-of-the-art features for long-range contact prediction._

The abstract from “Language models of protein sequences at the scale of evolution enable accurate structure prediction” is

_Large language models have recently been shown to develop emergent capabilities with scale, going beyond simple pattern matching to perform higher level reasoning and generate lifelike images and text. While language models trained on protein sequences have been studied at a smaller scale, little is known about what they learn about biology as they are scaled up. In this work we train models up to 15 billion parameters, the largest language models of proteins to be evaluated to date. We find that as models are scaled they learn information enabling the prediction of the three-dimensional structure of a protein at the resolution of individual atoms. We present ESMFold for high accuracy end-to-end atomic level structure prediction directly from the individual sequence of a protein. ESMFold has similar accuracy to AlphaFold2 and RoseTTAFold for sequences with low perplexity that are well understood by the language model. ESMFold inference is an order of magnitude faster than AlphaFold2, enabling exploration of the structural space of metagenomic proteins in practical timescales._

Tips:

-   ESM models are trained with a masked language modeling (MLM) objective.

The original code can be found [here](https://github.com/facebookresearch/esm) and was was developed by the Fundamental AI Research team at Meta AI. ESM-1b, ESM-1v and ESM-2 were contributed to huggingface by [jasonliu](https://huggingface.co/jasonliu) and [Matt](https://huggingface.co/Rocketknight1).

ESMFold was contributed to huggingface by [Matt](https://huggingface.co/Rocketknight1) and [Sylvain](https://huggingface.co/sgugger), with a big thank you to Nikita Smetanin, Roshan Rao and Tom Sercu for their help throughout the process!

The HuggingFace port of ESMFold uses portions of the [openfold](https://github.com/aqlaboratory/openfold) library. The `openfold` library is licensed under the Apache License 2.0.

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)

## EsmConfig

### class transformers.EsmConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/configuration_esm.py#L33)

( vocab\_size = Nonemask\_token\_id = Nonepad\_token\_id = Nonehidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 1026initializer\_range = 0.02layer\_norm\_eps = 1e-12position\_embedding\_type = 'absolute'use\_cache = Trueemb\_layer\_norm\_before = Nonetoken\_dropout = Falseis\_folding\_model = Falseesmfold\_config = Nonevocab\_list = None\*\*kwargs )

This is the configuration class to store the configuration of a `ESMModel`. It is used to instantiate a ESM model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ESM [facebook/esm-1b](https://huggingface.co/facebook/esm-1b) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import EsmModel, EsmConfig

>>> 

>>> 

>>> 
```

#### to\_dict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/configuration_esm.py#L160)

( ) → `Dict[str, any]`

Dictionary of all the attributes that make up this configuration instance,

Serializes this instance to a Python dictionary. Override the default [to\_dict()](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig.to_dict).

## EsmTokenizer

### class transformers.EsmTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/tokenization_esm.py#L47)

( vocab\_fileunk\_token = '<unk>'cls\_token = '<cls>'pad\_token = '<pad>'mask\_token = '<mask>'eos\_token = '<eos>'\*\*kwargs )

Constructs an ESM tokenizer.

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/tokenization_esm.py#L106)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None )

#### get\_special\_tokens\_mask

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/tokenization_esm.py#L120)

( token\_ids\_0: typing.Listtoken\_ids\_1: typing.Optional\[typing.List\] = Nonealready\_has\_special\_tokens: bool = False ) → A list of integers in the range \[0, 1\]

Parameters

-   **token\_ids\_0** (`List[int]`) — List of ids of the first sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — List of ids of the second sequence.
-   **already\_has\_special\_tokens** (`bool`, _optional_, defaults to `False`) — Whether or not the token list is already formatted with special tokens for the model.

Returns

A list of integers in the range \[0, 1\]

1 for a special token, 0 for a sequence token.

Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

#### create\_token\_type\_ids\_from\_sequences

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/tokenization_utils_base.py#L3305)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — The first tokenized sequence.
-   **token\_ids\_1** (`List[int]`, _optional_) — The second tokenized sequence.

The token type ids.

Create the token type IDs corresponding to the sequences passed. [What are token type IDs?](../glossary#token-type-ids)

Should be overridden in a subclass if the model has a special way of building those.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/tokenization_esm.py#L151)

( save\_directoryfilename\_prefix )

## EsmModel

### class transformers.EsmModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esm.py#L780)

( configadd\_pooling\_layer = True )

Parameters

-   **config** ([EsmConfig](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare ESM Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esm.py#L823)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

The [EsmModel](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, EsmModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
>>> model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## EsmForMaskedLM

### class transformers.EsmForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esm.py#L963)

( config )

Parameters

-   **config** ([EsmConfig](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ESM Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esm.py#L986)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

The [EsmForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, EsmForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
>>> model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

>>> inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> 
>>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

>>> outputs = model(**inputs, labels=labels)
```

## EsmForSequenceClassification

### class transformers.EsmForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esm.py#L1082)

( config )

Parameters

-   **config** ([EsmConfig](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ESM Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esm.py#L1093)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [EsmForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, EsmForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
>>> model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, EsmForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
>>> model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = EsmForSequenceClassification.from_pretrained(
...     "facebook/esm2_t6_8M_UR50D", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## EsmForTokenClassification

### class transformers.EsmForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esm.py#L1176)

( config )

Parameters

-   **config** ([EsmConfig](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ESM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esm.py#L1187)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [EsmForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, EsmForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
>>> model = EsmForTokenClassification.from_pretrained("facebook/esm2_t6_8M_UR50D")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> 
>>> 
>>> 
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
```

## EsmForProteinFolding

### class transformers.EsmForProteinFolding

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esmfold.py#L2020)

( config )

Parameters

-   **config** ([EsmConfig](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ESMForProteinFolding is the HuggingFace port of the original ESMFold model. It consists of an ESM-2 “stem” followed by a protein folding “head”, although unlike most other output heads, this “head” is similar in size and runtime to the rest of the model combined! It outputs a dictionary containing predicted structural information about the input protein(s).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_esmfold.py#L2084)

( input\_ids: Tensorattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonemasking\_pattern: typing.Optional\[torch.Tensor\] = Nonenum\_recycles: typing.Optional\[int\] = None ) → `transformers.models.esm.modeling_esmfold.EsmForProteinFoldingOutput` or `tuple(torch.FloatTensor)`

The [EsmForProteinFolding](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmForProteinFolding) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, EsmForProteinFolding

>>> model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
>>> inputs = tokenizer(["MLKNVQVQLV"], return_tensors="pt", add_special_tokens=False)  
>>> outputs = model(**inputs)
>>> folded_positions = outputs.positions
```

## TFEsmModel

### class transformers.TFEsmModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_tf_esm.py#L975)

( \*args\*\*kwargs )

Parameters

-   **config** ([EsmConfig](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare ESM Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular Keras model and refer to the TF/Keras documentation for all matters related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_tf_esm.py#L981)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneencoder\_hidden\_states: np.ndarray | tf.Tensor | None = Noneencoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonepast\_key\_values: Optional\[Tuple\[Tuple\[Union\[np.ndarray, tf.Tensor\]\]\]\] = Noneuse\_cache: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions) or `tuple(tf.Tensor)`

The [TFEsmModel](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.TFEsmModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFEsmModel
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
>>> model = TFEsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
>>> outputs = model(inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## TFEsmForMaskedLM

### class transformers.TFEsmForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_tf_esm.py#L1046)

( \*args\*\*kwargs )

Parameters

-   **config** ([EsmConfig](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

ESM Model with a `language modeling` head on top.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular Keras model and refer to the TF/Keras documentation for all matters related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_tf_esm.py#L1076)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Noneencoder\_hidden\_states: np.ndarray | tf.Tensor | None = Noneencoder\_attention\_mask: np.ndarray | tf.Tensor | None = Nonelabels: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFMaskedLMOutput) or `tuple(tf.Tensor)`

The [TFEsmForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.TFEsmForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFEsmForMaskedLM
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
>>> model = TFEsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

>>> inputs = tokenizer("The capital of France is <mask>.", return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
>>> selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

>>> predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
```

```
>>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
>>> 
>>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

>>> outputs = model(**inputs, labels=labels)
```

## TFEsmForSequenceClassification

### class transformers.TFEsmForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_tf_esm.py#L1194)

( \*args\*\*kwargs )

Parameters

-   **config** ([EsmConfig](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

ESM Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular Keras model and refer to the TF/Keras documentation for all matters related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_tf_esm.py#L1205)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonelabels: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or `tuple(tf.Tensor)`

The [TFEsmForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.TFEsmForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFEsmForSequenceClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
>>> model = TFEsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

>>> logits = model(**inputs).logits

>>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
```

```
>>> 
>>> num_labels = len(model.config.id2label)
>>> model = TFEsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D", num_labels=num_labels)

>>> labels = tf.constant(1)
>>> loss = model(**inputs, labels=labels).loss
```

## TFEsmForTokenClassification

### class transformers.TFEsmForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_tf_esm.py#L1268)

( \*args\*\*kwargs )

Parameters

-   **config** ([EsmConfig](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.EsmConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

ESM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular Keras model and refer to the TF/Keras documentation for all matters related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/esm/modeling_tf_esm.py#L1280)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneinputs\_embeds: np.ndarray | tf.Tensor | None = Nonelabels: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFTokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFTokenClassifierOutput) or `tuple(tf.Tensor)`

The [TFEsmForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/esm#transformers.TFEsmForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, TFEsmForTokenClassification
>>> import tensorflow as tf

>>> tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
>>> model = TFEsmForTokenClassification.from_pretrained("facebook/esm2_t6_8M_UR50D")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
... )

>>> logits = model(**inputs).logits
>>> predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

>>> 
>>> 
>>> 
>>> predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
```

```
>>> labels = predicted_token_class_ids
>>> loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
```