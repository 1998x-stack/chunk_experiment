**DISCLAIMER:** If you see something strange, file a [Github Issue](https://github.com/huggingface/transformers/issues/new?assignees=&labels=&template=bug-report.md&title) and assign [@gchhablani](https://www.github.com/gchhablani).

# Overview of PLBart

The PLBART model was proposed in [Unified Pre-training for Program Understanding and Generation](https://arxiv.org/abs/2103.06333) by Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang. This is a BART-like model which can be used to perform code-summarization, code-generation, and code-translation tasks. The pre-trained model `plbart-base` has been trained using multilingual denoising task on Java, Python and English.

According to the abstract

_Code summarization and generation empower conversion between programming language (PL) and natural language (NL), while code translation avails the migration of legacy code from one PL to another. This paper introduces PLBART, a sequence-to-sequence model capable of performing a broad spectrum of program and language understanding and generation tasks. PLBART is pre-trained on an extensive collection of Java and Python functions and associated NL text via denoising autoencoding. Experiments on code summarization in the English language, code generation, and code translation in seven programming languages show that PLBART outperforms or rivals state-of-the-art models. Moreover, experiments on discriminative tasks, e.g., program repair, clone detection, and vulnerable code detection, demonstrate PLBART’s effectiveness in program understanding. Furthermore, analysis reveals that PLBART learns program syntax, style (e.g., identifier naming convention), logical flow (e.g., if block inside an else block is equivalent to else if block) that are crucial to program semantics and thus excels even with limited annotations._

This model was contributed by [gchhablani](https://huggingface.co/gchhablani). The Authors’ code can be found [here](https://github.com/wasiahmad/PLBART).

### Training of PLBart

PLBart is a multilingual encoder-decoder (sequence-to-sequence) model primarily intended for code-to-text, text-to-code, code-to-code tasks. As the model is multilingual it expects the sequences in a different format. A special language id token is added in both the source and target text. The source text format is `X [eos, src_lang_code]` where `X` is the source text. The target text format is `[tgt_lang_code] X [eos]`. `bos` is never used.

However, for fine-tuning, in some cases no language token is provided in cases where a single language is used. Please refer to [the paper](https://arxiv.org/abs/2103.06333) to learn more about this.

In cases where the language code is needed, the regular [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) will encode source text format when you pass texts as the first argument or with the keyword argument `text`, and will encode target text format if it’s passed with the `text_target` keyword argument.

-   Supervised training

```
>>> from transformers import PLBartForConditionalGeneration, PLBartTokenizer

>>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base", src_lang="en_XX", tgt_lang="python")
>>> example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
>>> expected_translation_english = "Returns the maximum value of a b c."
>>> inputs = tokenizer(example_python_phrase, text_target=expected_translation_english, return_tensors="pt")
>>> model(**inputs)
```

-   Generation
    
    While generating the target text set the `decoder_start_token_id` to the target language id. The following example shows how to translate Python to English using the `uclanlp/plbart-python-en_XX` model.
    

```
>>> from transformers import PLBartForConditionalGeneration, PLBartTokenizer

>>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX")
>>> example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
>>> inputs = tokenizer(example_python_phrase, return_tensors="pt")
>>> model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-python-en_XX")
>>> translated_tokens = model.generate(**inputs, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"])
>>> tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
"Returns the maximum value of a b c."
```

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Translation task guide](../tasks/translation)
-   [Summarization task guide](../tasks/summarization)

## PLBartConfig

### class transformers.PLBartConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/configuration_plbart.py#L32)

( vocab\_size = 50005max\_position\_embeddings = 1024encoder\_layers = 6encoder\_ffn\_dim = 3072encoder\_attention\_heads = 12decoder\_layers = 6decoder\_ffn\_dim = 3072decoder\_attention\_heads = 12encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0use\_cache = Trueis\_encoder\_decoder = Trueactivation\_function = 'gelu'd\_model = 768dropout = 0.1attention\_dropout = 0.1activation\_dropout = 0.0init\_std = 0.02classifier\_dropout = 0.0scale\_embedding = Truepad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2forced\_eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [PLBartModel](/docs/transformers/v4.34.0/en/model_doc/plbart#transformers.PLBartModel). It is used to instantiate an PLBART model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the PLBART [uclanlp/plbart-base](https://huggingface.co/uclanlp/plbart-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import PLBartConfig, PLBartModel

>>> 
>>> configuration = PLBartConfig()

>>> 
>>> model = PLBartModel(configuration)

>>> 
>>> configuration = model.config
```

## PLBartTokenizer

### class transformers.PLBartTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/tokenization_plbart.py#L106)

( vocab\_filebos\_token = '<s>'eos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'language\_codes = 'base'tokenizer\_file = Nonesrc\_lang = Nonetgt\_lang = Nonesp\_model\_kwargs: typing.Union\[typing.Dict\[str, typing.Any\], NoneType\] = Noneadditional\_special\_tokens = None\*\*kwargs )

Construct an PLBART tokenizer.

Adapted from [RobertaTokenizer](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaTokenizer) and [XLNetTokenizer](/docs/transformers/v4.34.0/en/model_doc/xlnet#transformers.XLNetTokenizer). Based on [SentencePiece](https://github.com/google/sentencepiece).

The tokenization method is `<tokens> <eos> <language code>` for source language documents, and \`<language code>

<tokens> <eos>\` for target language documents.

Examples:

```
>>> from transformers import PLBartTokenizer

>>> tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-python-en_XX", src_lang="python", tgt_lang="en_XX")
>>> example_python_phrase = "def maximum(a,b,c):NEW_LINE_INDENTreturn max([a,b,c])"
>>> expected_translation_english = "Returns the maximum value of a b c."
>>> inputs = tokenizer(example_python_phrase, text_target=expected_translation_english, return_tensors="pt")
```

#### build\_inputs\_with\_special\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/tokenization_plbart.py#L333)

( token\_ids\_0: typing.List\[int\]token\_ids\_1: typing.Optional\[typing.List\[int\]\] = None ) → `List[int]`

Parameters

-   **token\_ids\_0** (`List[int]`) — List of IDs to which the special tokens will be added.
-   **token\_ids\_1** (`List[int]`, _optional_) — Optional second list of IDs for sequence pairs.

List of [input IDs](../glossary#input-ids) with the appropriate special tokens.

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. An PLBART sequence has the following format, where `X` represents the sequence:

-   `input_ids` (for encoder) `X [eos, src_lang_code]`
-   `decoder_input_ids`: (for decoder) `X [eos, tgt_lang_code]`

BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a separator.

## PLBartModel

### class transformers.PLBartModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/modeling_plbart.py#L1134)

( config: PLBartConfig )

Parameters

-   **config** ([PLBartConfig](/docs/transformers/v4.34.0/en/model_doc/plbart#transformers.PLBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare PLBART Model outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/modeling_plbart.py#L1162)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.LongTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqModelOutput) or `tuple(torch.FloatTensor)`

The [PLBartModel](/docs/transformers/v4.34.0/en/model_doc/plbart#transformers.PLBartModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, PLBartModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
>>> model = PLBartModel.from_pretrained("uclanlp/plbart-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## PLBartForConditionalGeneration

### class transformers.PLBartForConditionalGeneration

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/modeling_plbart.py#L1251)

( config: PLBartConfig )

Parameters

-   **config** ([PLBartConfig](/docs/transformers/v4.34.0/en/model_doc/plbart#transformers.PLBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The PLBART Model with a language modeling head. Can be used for code-to-text, text-to-code and code-to-code. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/modeling_plbart.py#L1290)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.LongTensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

The [PLBartForConditionalGeneration](/docs/transformers/v4.34.0/en/model_doc/plbart#transformers.PLBartForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Mask-filling example:

```
>>> from transformers import AutoTokenizer, PLBartForConditionalGeneration

>>> model = PLBartForConditionalGeneration.from_pretrained("uclanlp/plbart-base")
>>> tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")

>>> 
>>> TXT = "<s> Is 0 the <mask> Fibonacci number ? </s> en_XX"
>>> input_ids = tokenizer([TXT], add_special_tokens=False, return_tensors="pt").input_ids

>>> logits = model(input_ids).logits
>>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
>>> probs = logits[0, masked_index].softmax(dim=0)
>>> values, predictions = probs.topk(5)

>>> tokenizer.decode(predictions).split()
['first', 'same', 'highest', 'result', 'number']
```

## PLBartForSequenceClassification

### class transformers.PLBartForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/modeling_plbart.py#L1418)

( config: PLBartConfig\*\*kwargs )

Parameters

-   **config** ([PLBartConfig](/docs/transformers/v4.34.0/en/model_doc/plbart#transformers.PLBartConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

PLBart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for code classification.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/modeling_plbart.py#L1434)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_input\_ids: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonedecoder\_head\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_outputs: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.Seq2SeqSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqSequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [PLBartForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/plbart#transformers.PLBartForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, PLBartForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
>>> model = PLBartForSequenceClassification.from_pretrained("uclanlp/plbart-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = PLBartForSequenceClassification.from_pretrained("uclanlp/plbart-base", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, PLBartForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
>>> model = PLBartForSequenceClassification.from_pretrained("uclanlp/plbart-base", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = PLBartForSequenceClassification.from_pretrained(
...     "uclanlp/plbart-base", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## PLBartForCausalLM

### class transformers.PLBartForCausalLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/modeling_plbart.py#L1556)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/plbart/modeling_plbart.py#L1589)

( input\_ids: LongTensor = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonecross\_attn\_head\_mask: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.List\[torch.FloatTensor\]\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

Example:

```
>>> from transformers import AutoTokenizer, PLBartForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
>>> model = PLBartForCausalLM.from_pretrained("uclanlp/plbart-base", add_cross_attention=False)
>>> assert model.config.is_decoder, f"{model.__class__} has to be configured as a decoder."
>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> logits = outputs.logits
>>> expected_shape = [1, inputs.input_ids.shape[-1], model.config.vocab_size]
>>> list(logits.shape) == expected_shape
True
```