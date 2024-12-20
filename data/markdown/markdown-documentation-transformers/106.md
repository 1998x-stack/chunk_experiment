# BigBirdPegasus

## Overview

The BigBird model was proposed in [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) by Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others. BigBird, is a sparse-attention based transformer which extends Transformer based models, such as BERT to much longer sequences. In addition to sparse attention, BigBird also applies global attention as well as random attention to the input sequence. Theoretically, it has been shown that applying sparse, global, and random attention approximates full attention, while being computationally much more efficient for longer sequences. As a consequence of the capability to handle longer context, BigBird has shown improved performance on various long document NLP tasks, such as question answering and summarization, compared to BERT or RoBERTa.

The abstract from the paper is the following:

_Transformers-based models, such as BERT, have been one of the most successful deep learning models for NLP. Unfortunately, one of their core limitations is the quadratic dependency (mainly in terms of memory) on the sequence length due to their full attention mechanism. To remedy this, we propose, BigBird, a sparse attention mechanism that reduces this quadratic dependency to linear. We show that BigBird is a universal approximator of sequence functions and is Turing complete, thereby preserving these properties of the quadratic, full attention model. Along the way, our theoretical analysis reveals some of the benefits of having O(1) global tokens (such as CLS), that attend to the entire sequence as part of the sparse attention mechanism. The proposed sparse attention can handle sequences of length up to 8x of what was previously possible using similar hardware. As a consequence of the capability to handle longer context, BigBird drastically improves performance on various NLP tasks such as question answering and summarization. We also propose novel applications to genomics data._

Tips:

-   For an in-detail explanation on how BigBird’s attention works, see [this blog post](https://huggingface.co/blog/big-bird).
-   BigBird comes with 2 implementations: **original\_full** & **block\_sparse**. For the sequence length < 1024, using **original\_full** is advised as there is no benefit in using **block\_sparse** attention.
-   The code currently uses window size of 3 blocks and 2 global blocks.
-   Sequence length must be divisible by block size.
-   Current implementation supports only **ITC**.
-   Current implementation doesn’t support **num\_random\_blocks = 0**.
-   BigBirdPegasus uses the [PegasusTokenizer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/pegasus/tokenization_pegasus.py).
-   BigBird is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.

The original code can be found [here](https://github.com/google-research/bigbird).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## BigBirdPegasusConfig

### class transformers.BigBirdPegasusConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/configuration_bigbird_pegasus.py#L43)

( vocab\_size = 96103max\_position\_embeddings = 4096encoder\_layers = 16encoder\_ffn\_dim = 4096encoder\_attention\_heads = 16decoder\_layers = 16decoder\_ffn\_dim = 4096decoder\_attention\_heads = 16encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0use\_cache = Trueis\_encoder\_decoder = Trueactivation\_function = 'gelu\_new'd\_model = 1024dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02decoder\_start\_token\_id = 2classifier\_dropout = 0.0scale\_embedding = Truepad\_token\_id = 0bos\_token\_id = 2eos\_token\_id = 1attention\_type = 'block\_sparse'block\_size = 64num\_random\_blocks = 3use\_bias = False\*\*kwargs )

This is the configuration class to store the configuration of a [BigBirdPegasusModel](/docs/transformers/v4.34.0/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusModel). It is used to instantiate an BigBirdPegasus model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BigBirdPegasus [google/bigbird-pegasus-large-arxiv](https://huggingface.co/google/bigbird-pegasus-large-arxiv) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import BigBirdPegasusConfig, BigBirdPegasusModel

>>> 
>>> configuration = BigBirdPegasusConfig()

>>> 
>>> model = BigBirdPegasusModel(configuration)

>>> 
>>> configuration = model.config
```

## BigBirdPegasusModel

### class transformers.BigBirdPegasusModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2361)

( config: BigBirdPegasusConfig )

Parameters

-   **config** ([BigBirdPegasusConfig](/docs/transformers/v4.34.0/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare BigBirdPegasus Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2390)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [BigBirdPegasusModel](/docs/transformers/v4.34.0/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BigBirdPegasusModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
>>> model = BigBirdPegasusModel.from_pretrained("google/bigbird-pegasus-large-arxiv")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## BigBirdPegasusForConditionalGeneration

### class transformers.BigBirdPegasusForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2491)

( config: BigBirdPegasusConfig )

Parameters

-   **config** ([BigBirdPegasusConfig](/docs/transformers/v4.34.0/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The BigBirdPegasus Model with a language modeling head. Can be used for summarization. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2531)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [BigBirdPegasusForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Summarization example:

```
>>> from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration

>>> model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")
>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

>>> ARTICLE_TO_SUMMARIZE = (
...     "The dominant sequence transduction models are based on complex recurrent or convolutional neural "
...     "networks in an encoder-decoder configuration. The best performing models also connect the encoder "
...     "and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, "
...     "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. "
...     "Experiments on two machine translation tasks show these models to be superior in quality "
...     "while being more parallelizable and requiring significantly less time to train."
... )
>>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=4096, return_tensors="pt", truncation=True)

>>> 
>>> summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=15)
>>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'dominant sequence models are based on recurrent or convolutional neural networks .'
```

## BigBirdPegasusForSequenceClassification

### class transformers.BigBirdPegasusForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2667)

( config: BigBirdPegasusConfig\*\*kwargs )

Parameters

-   **config** ([BigBirdPegasusConfig](/docs/transformers/v4.34.0/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BigBirdPegasus model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2683)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [BigBirdPegasusForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, BigBirdPegasusForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
>>> model = BigBirdPegasusForSequenceClassification.from_pretrained("google/bigbird-pegasus-large-arxiv")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = BigBirdPegasusForSequenceClassification.from_pretrained("google/bigbird-pegasus-large-arxiv", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, BigBirdPegasusForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
>>> model = BigBirdPegasusForSequenceClassification.from_pretrained("google/bigbird-pegasus-large-arxiv", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = BigBirdPegasusForSequenceClassification.from_pretrained(
...     "google/bigbird-pegasus-large-arxiv", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## BigBirdPegasusForQuestionAnswering

### class transformers.BigBirdPegasusForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2796)

( config )

Parameters

-   **config** ([BigBirdPegasusConfig](/docs/transformers/v4.34.0/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

BigBirdPegasus Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2811)

( input\_ids: Tensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqQuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqQuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The [BigBirdPegasusForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/bigbird_pegasus#transformers.BigBirdPegasusForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, BigBirdPegasusForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
>>> model = BigBirdPegasusForQuestionAnswering.from_pretrained("google/bigbird-pegasus-large-arxiv")

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

## BigBirdPegasusForCausalLM

### class transformers.BigBirdPegasusForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2928)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py#L2961)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, BigBirdPegasusForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")
>>> model = BigBirdPegasusForCausalLM.from_pretrained(
...     "google/bigbird-pegasus-large-arxiv", add_cross_attention=False
... )
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
```