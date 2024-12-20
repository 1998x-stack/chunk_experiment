# VisualBERT

## Overview

The VisualBERT model was proposed in [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/pdf/1908.03557) by Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang. VisualBERT is a neural network trained on a variety of (image, text) pairs.

The abstract from the paper is the following:

_We propose VisualBERT, a simple and flexible framework for modeling a broad range of vision-and-language tasks. VisualBERT consists of a stack of Transformer layers that implicitly align elements of an input text and regions in an associated input image with self-attention. We further propose two visually-grounded language model objectives for pre-training VisualBERT on image caption data. Experiments on four vision-and-language tasks including VQA, VCR, NLVR2, and Flickr30K show that VisualBERT outperforms or rivals with state-of-the-art models while being significantly simpler. Further analysis demonstrates that VisualBERT can ground elements of language to image regions without any explicit supervision and is even sensitive to syntactic relationships, tracking, for example, associations between verbs and image regions corresponding to their arguments._

Tips:

1.  Most of the checkpoints provided work with the [VisualBertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertForPreTraining) configuration. Other checkpoints provided are the fine-tuned checkpoints for down-stream tasks - VQA (‘visualbert-vqa’), VCR (‘visualbert-vcr’), NLVR2 (‘visualbert-nlvr2’). Hence, if you are not working on these downstream tasks, it is recommended that you use the pretrained checkpoints.
    
2.  For the VCR task, the authors use a fine-tuned detector for generating visual embeddings, for all the checkpoints. We do not provide the detector and its weights as a part of the package, but it will be available in the research projects, and the states can be loaded directly into the detector provided.
    

## Usage

VisualBERT is a multi-modal vision and language model. It can be used for visual question answering, multiple choice, visual reasoning and region-to-phrase correspondence tasks. VisualBERT uses a BERT-like transformer to prepare embeddings for image-text pairs. Both the text and visual features are then projected to a latent space with identical dimension.

To feed images to the model, each image is passed through a pre-trained object detector and the regions and the bounding boxes are extracted. The authors use the features generated after passing these regions through a pre-trained CNN like ResNet as visual embeddings. They also add absolute position embeddings, and feed the resulting sequence of vectors to a standard BERT model. The text input is concatenated in the front of the visual embeddings in the embedding layer, and is expected to be bound by \[CLS\] and a \[SEP\] tokens, as in BERT. The segment IDs must also be set appropriately for the textual and visual parts.

The [BertTokenizer](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizer) is used to encode the text. A custom detector/image processor must be used to get the visual embeddings. The following example notebooks show how to use VisualBERT with Detectron-like models:

-   [VisualBERT VQA demo notebook](https://github.com/huggingface/transformers/tree/main/examples/research_projects/visual_bert) : This notebook contains an example on VisualBERT VQA.
    
-   [Generate Embeddings for VisualBERT (Colab Notebook)](https://colab.research.google.com/drive/1bLGxKdldwqnMVA5x4neY7-l_8fKGWQYI?usp=sharing) : This notebook contains an example on how to generate visual embeddings.
    

The following example shows how to get the last hidden state using [VisualBertModel](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertModel):

```
>>> import torch
>>> from transformers import BertTokenizer, VisualBertModel

>>> model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
>>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("What is the man eating?", return_tensors="pt")
>>> 
>>> visual_embeds = get_visual_embeddings(image_path)

>>> visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
>>> visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
>>> inputs.update(
...     {
...         "visual_embeds": visual_embeds,
...         "visual_token_type_ids": visual_token_type_ids,
...         "visual_attention_mask": visual_attention_mask,
...     }
... )
>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
```

This model was contributed by [gchhablani](https://huggingface.co/gchhablani). The original code can be found [here](https://github.com/uclanlp/visualbert).

## VisualBertConfig

### class transformers.VisualBertConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/configuration_visual_bert.py#L43)

( vocab\_size = 30522hidden\_size = 768visual\_embedding\_dim = 512num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512type\_vocab\_size = 2initializer\_range = 0.02layer\_norm\_eps = 1e-12bypass\_transformer = Falsespecial\_visual\_initialize = Truepad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [VisualBertModel](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertModel). It is used to instantiate an VisualBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the VisualBERT [uclanlp/visualbert-vqa-coco-pre](https://huggingface.co/uclanlp/visualbert-vqa-coco-pre) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import VisualBertConfig, VisualBertModel

>>> 
>>> configuration = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

>>> 
>>> model = VisualBertModel(configuration)

>>> 
>>> configuration = model.config
```

## VisualBertModel

### class transformers.VisualBertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L678)

( configadd\_pooling\_layer = True )

Parameters

-   **config** ([VisualBertConfig](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare VisualBert Model transformer outputting raw hidden-states without any specific head on top. This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The model can behave as an encoder (with only self-attention) following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L717)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonevisual\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneimage\_text\_alignment: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [VisualBertModel](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
from transformers import AutoTokenizer, VisualBertModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
```

## VisualBertForPreTraining

### class transformers.VisualBertForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L875)

( config )

Parameters

-   **config** ([VisualBertConfig](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

VisualBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `sentence-image prediction (classification)` head.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L893)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonevisual\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneimage\_text\_alignment: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonesentence\_image\_labels: typing.Optional\[torch.LongTensor\] = None ) → `transformers.models.visual_bert.modeling_visual_bert.VisualBertForPreTrainingOutput` or `tuple(torch.FloatTensor)`

The [VisualBertForPreTraining](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
from transformers import AutoTokenizer, VisualBertForPreTraining

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)
max_length = inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]
labels = tokenizer(
    "The capital of France is Paris.", return_tensors="pt", padding="max_length", max_length=max_length
)["input_ids"]
sentence_image_labels = torch.tensor(1).unsqueeze(0)  


outputs = model(**inputs, labels=labels, sentence_image_labels=sentence_image_labels)
loss = outputs.loss
prediction_logits = outputs.prediction_logits
seq_relationship_logits = outputs.seq_relationship_logits
```

## VisualBertForQuestionAnswering

### class transformers.VisualBertForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L1179)

( config )

Parameters

-   **config** ([VisualBertConfig](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

VisualBert Model with a classification/regression head on top (a dropout and a linear layer on top of the pooled output) for VQA.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L1191)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonevisual\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneimage\_text\_alignment: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [VisualBertForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
from transformers import AutoTokenizer, VisualBertForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa")

text = "Who is eating the apple?"
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

labels = torch.tensor([[0.0, 1.0]]).unsqueeze(0)  

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits
```

## VisualBertForMultipleChoice

### class transformers.VisualBertForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L1026)

( config )

Parameters

-   **config** ([VisualBertConfig](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

VisualBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for VCR tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L1037)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonevisual\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneimage\_text\_alignment: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = None ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

The [VisualBertForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
from transformers import AutoTokenizer, VisualBertForMultipleChoice
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."

visual_embeds = get_visual_embeddings(image)

visual_embeds = visual_embeds.expand(1, 2, *visual_embeds.shape)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

labels = torch.tensor(0).unsqueeze(0)  

encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors="pt", padding=True)

inputs_dict = {k: v.unsqueeze(0) for k, v in encoding.items()}
inputs_dict.update(
    {
        "visual_embeds": visual_embeds,
        "visual_attention_mask": visual_attention_mask,
        "visual_token_type_ids": visual_token_type_ids,
        "labels": labels,
    }
)
outputs = model(**inputs_dict)

loss = outputs.loss
logits = outputs.logits
```

## VisualBertForVisualReasoning

### class transformers.VisualBertForVisualReasoning

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L1305)

( config )

Parameters

-   **config** ([VisualBertConfig](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

VisualBert Model with a sequence classification head on top (a dropout and a linear layer on top of the pooled output) for Visual Reasoning e.g. for NLVR task.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L1317)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonevisual\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneimage\_text\_alignment: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Nonelabels: typing.Optional\[torch.LongTensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [VisualBertForVisualReasoning](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertForVisualReasoning) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
from transformers import AutoTokenizer, VisualBertForVisualReasoning
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2")

text = "Who is eating the apple?"
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)

inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

labels = torch.tensor(1).unsqueeze(0)  

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits
```

## VisualBertForRegionToPhraseAlignment

### class transformers.VisualBertForRegionToPhraseAlignment

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L1465)

( config )

Parameters

-   **config** ([VisualBertConfig](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

VisualBert Model with a Masked Language Modeling head and an attention layer on top for Region-to-Phrase Alignment e.g. for Flickr30 Entities task.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/visual_bert/modeling_visual_bert.py#L1479)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.LongTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.LongTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_embeds: typing.Optional\[torch.FloatTensor\] = Nonevisual\_attention\_mask: typing.Optional\[torch.LongTensor\] = Nonevisual\_token\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneimage\_text\_alignment: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = Noneregion\_to\_phrase\_position: typing.Optional\[torch.LongTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [VisualBertForRegionToPhraseAlignment](/docs/transformers/v4.34.0/en/model_doc/visual_bert#transformers.VisualBertForRegionToPhraseAlignment) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
from transformers import AutoTokenizer, VisualBertForRegionToPhraseAlignment
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = VisualBertForRegionToPhraseAlignment.from_pretrained("uclanlp/visualbert-vqa-coco-pre")

text = "Who is eating the apple?"
inputs = tokenizer(text, return_tensors="pt")
visual_embeds = get_visual_embeddings(image).unsqueeze(0)
visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
region_to_phrase_position = torch.ones((1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2]))

inputs.update(
    {
        "region_to_phrase_position": region_to_phrase_position,
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)

labels = torch.ones(
    (1, inputs["input_ids"].shape[-1] + visual_embeds.shape[-2], visual_embeds.shape[-2])
)  

outputs = model(**inputs, labels=labels)
loss = outputs.loss
scores = outputs.logits
```