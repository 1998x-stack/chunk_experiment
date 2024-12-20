# LUKE

## Overview

The LUKE model was proposed in [LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention](https://arxiv.org/abs/2010.01057) by Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda and Yuji Matsumoto. It is based on RoBERTa and adds entity embeddings as well as an entity-aware self-attention mechanism, which helps improve performance on various downstream tasks involving reasoning about entities such as named entity recognition, extractive and cloze-style question answering, entity typing, and relation classification.

The abstract from the paper is the following:

_Entity representations are useful in natural language tasks involving entities. In this paper, we propose new pretrained contextualized representations of words and entities based on the bidirectional transformer. The proposed model treats words and entities in a given text as independent tokens, and outputs contextualized representations of them. Our model is trained using a new pretraining task based on the masked language model of BERT. The task involves predicting randomly masked words and entities in a large entity-annotated corpus retrieved from Wikipedia. We also propose an entity-aware self-attention mechanism that is an extension of the self-attention mechanism of the transformer, and considers the types of tokens (words or entities) when computing attention scores. The proposed model achieves impressive empirical performance on a wide range of entity-related tasks. In particular, it obtains state-of-the-art results on five well-known datasets: Open Entity (entity typing), TACRED (relation classification), CoNLL-2003 (named entity recognition), ReCoRD (cloze-style question answering), and SQuAD 1.1 (extractive question answering)._

Tips:

-   This implementation is the same as [RobertaModel](/docs/transformers/v4.34.0/en/model_doc/roberta#transformers.RobertaModel) with the addition of entity embeddings as well as an entity-aware self-attention mechanism, which improves performance on tasks involving reasoning about entities.
    
-   LUKE treats entities as input tokens; therefore, it takes `entity_ids`, `entity_attention_mask`, `entity_token_type_ids` and `entity_position_ids` as extra input. You can obtain those using [LukeTokenizer](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeTokenizer).
    
-   [LukeTokenizer](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeTokenizer) takes `entities` and `entity_spans` (character-based start and end positions of the entities in the input text) as extra input. `entities` typically consist of \[MASK\] entities or Wikipedia entities. The brief description when inputting these entities are as follows:
    
    -   _Inputting \[MASK\] entities to compute entity representations_: The \[MASK\] entity is used to mask entities to be predicted during pretraining. When LUKE receives the \[MASK\] entity, it tries to predict the original entity by gathering the information about the entity from the input text. Therefore, the \[MASK\] entity can be used to address downstream tasks requiring the information of entities in text such as entity typing, relation classification, and named entity recognition.
    -   _Inputting Wikipedia entities to compute knowledge-enhanced token representations_: LUKE learns rich information (or knowledge) about Wikipedia entities during pretraining and stores the information in its entity embedding. By using Wikipedia entities as input tokens, LUKE outputs token representations enriched by the information stored in the embeddings of these entities. This is particularly effective for tasks requiring real-world knowledge, such as question answering.
-   There are three head models for the former use case:
    
    -   [LukeForEntityClassification](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForEntityClassification), for tasks to classify a single entity in an input text such as entity typing, e.g. the [Open Entity dataset](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html). This model places a linear head on top of the output entity representation.
    -   [LukeForEntityPairClassification](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForEntityPairClassification), for tasks to classify the relationship between two entities such as relation classification, e.g. the [TACRED dataset](https://nlp.stanford.edu/projects/tacred/). This model places a linear head on top of the concatenated output representation of the pair of given entities.
    -   [LukeForEntitySpanClassification](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForEntitySpanClassification), for tasks to classify the sequence of entity spans, such as named entity recognition (NER). This model places a linear head on top of the output entity representations. You can address NER using this model by inputting all possible entity spans in the text to the model.
    
    [LukeTokenizer](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeTokenizer) has a `task` argument, which enables you to easily create an input to these head models by specifying `task="entity_classification"`, `task="entity_pair_classification"`, or `task="entity_span_classification"`. Please refer to the example code of each head models.
    
    A demo notebook on how to fine-tune [LukeForEntityPairClassification](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForEntityPairClassification) for relation classification can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LUKE).
    
    There are also 3 notebooks available, which showcase how you can reproduce the results as reported in the paper with the HuggingFace implementation of LUKE. They can be found [here](https://github.com/studio-ousia/luke/tree/master/notebooks).
    

Example:

```
>>> from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification

>>> model = LukeModel.from_pretrained("studio-ousia/luke-base")
>>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")


>>> text = "Beyoncé lives in Los Angeles."
>>> entity_spans = [(0, 7)]  
>>> inputs = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
>>> outputs = model(**inputs)
>>> word_last_hidden_state = outputs.last_hidden_state
>>> entity_last_hidden_state = outputs.entity_last_hidden_state


>>> entities = [
...     "Beyoncé",
...     "Los Angeles",
... ]  
>>> entity_spans = [(0, 7), (17, 28)]  
>>> inputs = tokenizer(text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
>>> outputs = model(**inputs)
>>> word_last_hidden_state = outputs.last_hidden_state
>>> entity_last_hidden_state = outputs.entity_last_hidden_state


>>> model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
>>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
>>> entity_spans = [(0, 7), (17, 28)]  
>>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> predicted_class_idx = int(logits[0].argmax())
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
```

This model was contributed by [ikuyamada](https://huggingface.co/ikuyamada) and [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/studio-ousia/luke).

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)
-   [Multiple choice task guide](../tasks/multiple_choice)

## LukeConfig

### class transformers.LukeConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/configuration_luke.py#L29)

( vocab\_size = 50267entity\_vocab\_size = 500000hidden\_size = 768entity\_emb\_size = 256num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512type\_vocab\_size = 2initializer\_range = 0.02layer\_norm\_eps = 1e-12use\_entity\_aware\_attention = Trueclassifier\_dropout = Nonepad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [LukeModel](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeModel). It is used to instantiate a LUKE model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the LUKE [studio-ousia/luke-base](https://huggingface.co/studio-ousia/luke-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import LukeConfig, LukeModel

>>> 
>>> configuration = LukeConfig()

>>> 
>>> model = LukeModel(configuration)

>>> 
>>> configuration = model.config
```

## LukeTokenizer

### class transformers.LukeTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/tokenization_luke.py#L193)

( vocab\_filemerges\_fileentity\_vocab\_filetask = Nonemax\_entity\_length = 32max\_mention\_length = 30entity\_token\_1 = '<ent>'entity\_token\_2 = '<ent2>'entity\_unk\_token = '\[UNK\]'entity\_pad\_token = '\[PAD\]'entity\_mask\_token = '\[MASK\]'entity\_mask2\_token = '\[MASK2\]'errors = 'replace'bos\_token = '<s>'eos\_token = '</s>'sep\_token = '</s>'cls\_token = '<s>'unk\_token = '<unk>'pad\_token = '<pad>'mask\_token = '<mask>'add\_prefix\_space = False\*\*kwargs )

Constructs a LUKE tokenizer, derived from the GPT-2 tokenizer, using byte-level Byte-Pair-Encoding.

This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will

be encoded differently whether it is at the beginning of the sentence (without space) or not:

```
>>> from transformers import LukeTokenizer

>>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
>>> tokenizer("Hello world")["input_ids"]
[0, 31414, 232, 2]

>>> tokenizer(" Hello world")["input_ids"]
[0, 20920, 232, 2]
```

You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods. It also creates entity sequences, namely `entity_ids`, `entity_attention_mask`, `entity_token_type_ids`, and `entity_position_ids` to be used by the LUKE model.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/tokenization_luke.py#L577)

( text: typing.Union\[str, typing.List\[str\]\]text\_pair: typing.Union\[str, typing.List\[str\], NoneType\] = Noneentity\_spans: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], typing.List\[typing.List\[typing.Tuple\[int, int\]\]\], NoneType\] = Noneentity\_spans\_pair: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], typing.List\[typing.List\[typing.Tuple\[int, int\]\]\], NoneType\] = Noneentities: typing.Union\[typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Noneentities\_pair: typing.Union\[typing.List\[str\], typing.List\[typing.List\[str\]\], NoneType\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonemax\_entity\_length: typing.Optional\[int\] = Nonestride: int = 0is\_split\_into\_words: typing.Optional\[bool\] = Falsepad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonereturn\_token\_type\_ids: typing.Optional\[bool\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = True\*\*kwargs ) → [BatchEncoding](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.BatchEncoding)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of sequences, depending on the task you want to prepare them for.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/tokenization_luke.py#L1692)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## LukeModel

### class transformers.LukeModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1024)

( config: LukeConfigadd\_pooling\_layer: bool = True )

Parameters

-   **config** ([LukeConfig](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare LUKE model transformer outputting raw hidden-states for both word tokens and entities without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1053)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneentity\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_position\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.luke.modeling_luke.BaseLukeModelOutputWithPooling` or `tuple(torch.FloatTensor)`

The [LukeModel](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, LukeModel

>>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
>>> model = LukeModel.from_pretrained("studio-ousia/luke-base")


>>> text = "Beyoncé lives in Los Angeles."
>>> entity_spans = [(0, 7)]  

>>> encoding = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
>>> outputs = model(**encoding)
>>> word_last_hidden_state = outputs.last_hidden_state
>>> entity_last_hidden_state = outputs.entity_last_hidden_state


>>> text = "Beyoncé lives in Los Angeles."
>>> entities = [
...     "Beyoncé",
...     "Los Angeles",
... ]  
>>> entity_spans = [
...     (0, 7),
...     (17, 28),
... ]  

>>> encoding = tokenizer(
...     text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt"
... )
>>> outputs = model(**encoding)
>>> word_last_hidden_state = outputs.last_hidden_state
>>> entity_last_hidden_state = outputs.entity_last_hidden_state
```

## LukeForMaskedLM

### class transformers.LukeForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1279)

( config )

Parameters

-   **config** ([LukeConfig](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LUKE model with a language modeling head and entity prediction head on top for masked language modeling and masked entity prediction.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1305)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneentity\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_position\_ids: typing.Optional\[torch.LongTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneentity\_labels: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.luke.modeling_luke.LukeMaskedLMOutput` or `tuple(torch.FloatTensor)`

The [LukeForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## LukeForEntityClassification

### class transformers.LukeForEntityClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1414)

( config )

Parameters

-   **config** ([LukeConfig](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LUKE model with a classification head on top (a linear layer on top of the hidden state of the first entity token) for entity classification tasks, such as Open Entity.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1427)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneentity\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_position\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.luke.modeling_luke.EntityClassificationOutput` or `tuple(torch.FloatTensor)`

The [LukeForEntityClassification](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForEntityClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, LukeForEntityClassification

>>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
>>> model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")

>>> text = "Beyoncé lives in Los Angeles."
>>> entity_spans = [(0, 7)]  
>>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: person
```

## LukeForEntityPairClassification

### class transformers.LukeForEntityPairClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1529)

( config )

Parameters

-   **config** ([LukeConfig](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LUKE model with a classification head on top (a linear layer on top of the hidden states of the two entity tokens) for entity pair classification tasks, such as TACRED.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1542)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneentity\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_position\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.luke.modeling_luke.EntityPairClassificationOutput` or `tuple(torch.FloatTensor)`

The [LukeForEntityPairClassification](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForEntityPairClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, LukeForEntityPairClassification

>>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
>>> model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")

>>> text = "Beyoncé lives in Los Angeles."
>>> entity_spans = [
...     (0, 7),
...     (17, 28),
... ]  
>>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
Predicted class: per:cities_of_residence
```

## LukeForEntitySpanClassification

### class transformers.LukeForEntitySpanClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1649)

( config )

Parameters

-   **config** ([LukeConfig](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LUKE model with a span classification head on top (a linear layer on top of the hidden states output) for tasks such as named entity recognition.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1662)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_attention\_mask: typing.Optional\[torch.LongTensor\] = Noneentity\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_position\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_start\_positions: typing.Optional\[torch.LongTensor\] = Noneentity\_end\_positions: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.luke.modeling_luke.EntitySpanClassificationOutput` or `tuple(torch.FloatTensor)`

The [LukeForEntitySpanClassification](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForEntitySpanClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoTokenizer, LukeForEntitySpanClassification

>>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
>>> model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")

>>> text = "Beyoncé lives in Los Angeles"


>>> word_start_positions = [0, 8, 14, 17, 21]  
>>> word_end_positions = [7, 13, 16, 20, 28]  
>>> entity_spans = []
>>> for i, start_pos in enumerate(word_start_positions):
...     for end_pos in word_end_positions[i:]:
...         entity_spans.append((start_pos, end_pos))

>>> inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> predicted_class_indices = logits.argmax(-1).squeeze().tolist()
>>> for span, predicted_class_idx in zip(entity_spans, predicted_class_indices):
...     if predicted_class_idx != 0:
...         print(text[span[0] : span[1]], model.config.id2label[predicted_class_idx])
Beyoncé PER
Los Angeles LOC
```

## LukeForSequenceClassification

### class transformers.LukeForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1795)

( config )

Parameters

-   **config** ([LukeConfig](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LUKE Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1808)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneentity\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_position\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.luke.modeling_luke.LukeSequenceClassifierOutput` or `tuple(torch.FloatTensor)`

The [LukeForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, LukeForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
>>> model = LukeForSequenceClassification.from_pretrained("studio-ousia/luke-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = LukeForSequenceClassification.from_pretrained("studio-ousia/luke-base", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, LukeForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
>>> model = LukeForSequenceClassification.from_pretrained("studio-ousia/luke-base", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = LukeForSequenceClassification.from_pretrained(
...     "studio-ousia/luke-base", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## LukeForMultipleChoice

### class transformers.LukeForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L2124)

( config )

Parameters

-   **config** ([LukeConfig](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LUKE Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L2137)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneentity\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_position\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.luke.modeling_luke.LukeMultipleChoiceModelOutput` or `tuple(torch.FloatTensor)`

The [LukeForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LukeForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
>>> model = LukeForMultipleChoice.from_pretrained("studio-ousia/luke-base")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> choice0 = "It is eaten with a fork and a knife."
>>> choice1 = "It is eaten while held in the hand."
>>> labels = torch.tensor(0).unsqueeze(0)  

>>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in encoding.items()}, labels=labels)  

>>> 
>>> loss = outputs.loss
>>> logits = outputs.logits
```

## LukeForTokenClassification

### class transformers.LukeForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1909)

( config )

Parameters

-   **config** ([LukeConfig](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LUKE Model with a token classification head on top (a linear layer on top of the hidden-states output). To solve Named-Entity Recognition (NER) task using LUKE, `LukeForEntitySpanClassification` is more suitable than this class.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L1923)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneentity\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_position\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.luke.modeling_luke.LukeTokenClassifierOutput` or `tuple(torch.FloatTensor)`

The [LukeForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LukeForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
>>> model = LukeForTokenClassification.from_pretrained("studio-ousia/luke-base")

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

## LukeForQuestionAnswering

### class transformers.LukeForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L2005)

( config )

Parameters

-   **config** ([LukeConfig](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The LUKE Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/luke/modeling_luke.py#L2017)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.FloatTensor\] = Noneentity\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneentity\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneentity\_position\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.luke.modeling_luke.LukeQuestionAnsweringModelOutput` or `tuple(torch.FloatTensor)`

The [LukeForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/luke#transformers.LukeForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, LukeForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
>>> model = LukeForQuestionAnswering.from_pretrained("studio-ousia/luke-base")

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