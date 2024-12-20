# QDQBERT

## Overview

The QDQBERT model can be referenced in [Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation](https://arxiv.org/abs/2004.09602) by Hao Wu, Patrick Judd, Xiaojie Zhang, Mikhail Isaev and Paulius Micikevicius.

The abstract from the paper is the following:

_Quantization techniques can reduce the size of Deep Neural Networks and improve inference latency and throughput by taking advantage of high throughput integer instructions. In this paper we review the mathematical aspects of quantization parameters and evaluate their choices on a wide range of neural network models for different application domains, including vision, speech, and language. We focus on quantization techniques that are amenable to acceleration by processors with high-throughput integer math pipelines. We also present a workflow for 8-bit quantization that is able to maintain accuracy within 1% of the floating-point baseline on all networks studied, including models that are more difficult to quantize, such as MobileNets and BERT-large._

Tips:

-   QDQBERT model adds fake quantization operations (pair of QuantizeLinear/DequantizeLinear ops) to (i) linear layer inputs and weights, (ii) matmul inputs, (iii) residual add inputs, in BERT model.
    
-   QDQBERT requires the dependency of [Pytorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization). To install `pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com`
    
-   QDQBERT model can be loaded from any checkpoint of HuggingFace BERT model (for example _bert-base-uncased_), and perform Quantization Aware Training/Post Training Quantization.
    
-   A complete example of using QDQBERT model to perform Quatization Aware Training and Post Training Quantization for SQUAD task can be found at [transformers/examples/research\_projects/quantization-qdqbert/](examples/research_projects/quantization-qdqbert/).
    

This model was contributed by [shangz](https://huggingface.co/shangz).

### Set default quantizers

QDQBERT model adds fake quantization operations (pair of QuantizeLinear/DequantizeLinear ops) to BERT by `TensorQuantizer` in [Pytorch Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization). `TensorQuantizer` is the module for quantizing tensors, with `QuantDescriptor` defining how the tensor should be quantized. Refer to [Pytorch Quantization Toolkit userguide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html) for more details.

Before creating QDQBERT model, one has to set the default `QuantDescriptor` defining default tensor quantizers.

Example:

```
>>> import pytorch_quantization.nn as quant_nn
>>> from pytorch_quantization.tensor_quant import QuantDescriptor

>>> 
>>> input_desc = QuantDescriptor(num_bits=8, calib_method="max")
>>> 
>>> weight_desc = QuantDescriptor(num_bits=8, axis=((0,)))
>>> quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
>>> quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)
```

### Calibration

Calibration is the terminology of passing data samples to the quantizer and deciding the best scaling factors for tensors. After setting up the tensor quantizers, one can use the following example to calibrate the model:

```
>>> 
>>> for name, module in model.named_modules():
...     if name.endswith("_input_quantizer"):
...         module.enable_calib()
...         module.disable_quant()  

>>> 
>>> model(x)
>>> 

>>> 
>>> for name, module in model.named_modules():
...     if name.endswith("_input_quantizer"):
...         module.load_calib_amax()
...         module.enable_quant()

>>> 
>>> model.cuda()

>>> 
>>> 
```

### Export to ONNX

The goal of exporting to ONNX is to deploy inference by [TensorRT](https://developer.nvidia.com/tensorrt). Fake quantization will be broken into a pair of QuantizeLinear/DequantizeLinear ONNX ops. After setting static member of TensorQuantizer to use Pytorch’s own fake quantization functions, fake quantized model can be exported to ONNX, follow the instructions in [torch.onnx](https://pytorch.org/docs/stable/onnx.html). Example:

```
>>> from pytorch_quantization.nn import TensorQuantizer

>>> TensorQuantizer.use_fb_fake_quant = True

>>> 
>>> ...
>>> 
>>> torch.onnx.export(...)
```

## Documentation resources

-   [Text classification task guide](../tasks/sequence_classification)
-   [Token classification task guide](../tasks/token_classification)
-   [Question answering task guide](../tasks/question_answering)
-   [Causal language modeling task guide](../tasks/language_modeling)
-   [Masked language modeling task guide](../tasks/masked_language_modeling)
-   [Multiple choice task guide](../tasks/multiple_choice)

## QDQBertConfig

### class transformers.QDQBertConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/configuration_qdqbert.py#L29)

( vocab\_size = 30522hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.1attention\_probs\_dropout\_prob = 0.1max\_position\_embeddings = 512type\_vocab\_size = 2initializer\_range = 0.02layer\_norm\_eps = 1e-12use\_cache = Truepad\_token\_id = 1bos\_token\_id = 0eos\_token\_id = 2\*\*kwargs )

This is the configuration class to store the configuration of a [QDQBertModel](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertModel). It is used to instantiate an QDQBERT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BERT [bert-base-uncased](https://huggingface.co/bert-base-uncased) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import QDQBertModel, QDQBertConfig

>>> 
>>> configuration = QDQBertConfig()

>>> 
>>> model = QDQBertModel(configuration)

>>> 
>>> configuration = model.config
```

## QDQBertModel

### class transformers.QDQBertModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L835)

( configadd\_pooling\_layer: bool = True )

Parameters

-   **config** ([QDQBertConfig](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare QDQBERT Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of cross-attention is added between the self-attention layers, following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L875)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPoolingAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions) or `tuple(torch.FloatTensor)`

The [QDQBertModel](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, QDQBertModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = QDQBertModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
```

## QDQBertLMHeadModel

### class transformers.QDQBertLMHeadModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1016)

( config )

Parameters

-   **config** ([QDQBertConfig](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

QDQBERT Model with a `language modeling` head on top for CLM fine-tuning.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1037)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.LongTensor\]\]\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

The [QDQBertLMHeadModel](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertLMHeadModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, QDQBertLMHeadModel, QDQBertConfig
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
>>> config = QDQBertConfig.from_pretrained("bert-base-cased")
>>> config.is_decoder = True
>>> model = QDQBertLMHeadModel.from_pretrained("bert-base-cased", config=config)

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
>>> outputs = model(**inputs)

>>> prediction_logits = outputs.logits
```

## QDQBertForMaskedLM

### class transformers.QDQBertForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1170)

( config )

Parameters

-   **config** ([QDQBertConfig](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

QDQBERT Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1194)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.FloatTensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

The [QDQBertForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, QDQBertForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = QDQBertForMaskedLM.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

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

## QDQBertForSequenceClassification

### class transformers.QDQBertForSequenceClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1384)

( config )

Parameters

-   **config** ([QDQBertConfig](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1396)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [QDQBertForSequenceClassification](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example of single-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, QDQBertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = QDQBertForSequenceClassification.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = QDQBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
```

Example of multi-label classification:

```
>>> import torch
>>> from transformers import AutoTokenizer, QDQBertForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = QDQBertForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> 
>>> num_labels = len(model.config.id2label)
>>> model = QDQBertForSequenceClassification.from_pretrained(
...     "bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## QDQBertForNextSentencePrediction

### class transformers.QDQBertForNextSentencePrediction

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1280)

( config )

Parameters

-   **config** ([QDQBertConfig](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bert Model with a `next sentence prediction (classification)` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1290)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs ) → [transformers.modeling\_outputs.NextSentencePredictorOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.NextSentencePredictorOutput) or `tuple(torch.FloatTensor)`

The [QDQBertForNextSentencePrediction](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertForNextSentencePrediction) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, QDQBertForNextSentencePrediction
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = QDQBertForNextSentencePrediction.from_pretrained("bert-base-uncased")

>>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
>>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
>>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

>>> outputs = model(**encoding, labels=torch.LongTensor([1]))
>>> logits = outputs.logits
>>> assert logits[0, 0] < logits[0, 1]  
```

## QDQBertForMultipleChoice

### class transformers.QDQBertForMultipleChoice

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1481)

( config )

Parameters

-   **config** ([QDQBertConfig](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1492)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MultipleChoiceModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MultipleChoiceModelOutput) or `tuple(torch.FloatTensor)`

The [QDQBertForMultipleChoice](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertForMultipleChoice) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, QDQBertForMultipleChoice
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = QDQBertForMultipleChoice.from_pretrained("bert-base-uncased")

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

## QDQBertForTokenClassification

### class transformers.QDQBertForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1572)

( config )

Parameters

-   **config** ([QDQBertConfig](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

QDQBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1584)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [QDQBertForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, QDQBertForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = QDQBertForTokenClassification.from_pretrained("bert-base-uncased")

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

## QDQBertForQuestionAnswering

### class transformers.QDQBertForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1650)

( config )

Parameters

-   **config** ([QDQBertConfig](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

QDQBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/qdqbert/modeling_qdqbert.py#L1661)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonestart\_positions: typing.Optional\[torch.LongTensor\] = Noneend\_positions: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.QuestionAnsweringModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput) or `tuple(torch.FloatTensor)`

The [QDQBertForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/qdqbert#transformers.QDQBertForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoTokenizer, QDQBertForQuestionAnswering
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = QDQBertForQuestionAnswering.from_pretrained("bert-base-uncased")

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