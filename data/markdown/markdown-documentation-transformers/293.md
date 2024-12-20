# ViLT

## Overview

The ViLT model was proposed in [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) by Wonjae Kim, Bokyung Son, Ildoo Kim. ViLT incorporates text embeddings into a Vision Transformer (ViT), allowing it to have a minimal design for Vision-and-Language Pre-training (VLP).

The abstract from the paper is the following:

_Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks. Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision (e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model, Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of times faster than previous VLP models, yet with competitive or better downstream task performance._

Tips:

-   The quickest way to get started with ViLT is by checking the [example notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ViLT) (which showcase both inference and fine-tuning on custom data).
-   ViLT is a model that takes both `pixel_values` and `input_ids` as input. One can use [ViltProcessor](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltProcessor) to prepare data for the model. This processor wraps a image processor (for the image modality) and a tokenizer (for the language modality) into one.
-   ViLT is trained with images of various sizes: the authors resize the shorter edge of input images to 384 and limit the longer edge to under 640 while preserving the aspect ratio. To make batching of images possible, the authors use a `pixel_mask` that indicates which pixel values are real and which are padding. [ViltProcessor](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltProcessor) automatically creates this for you.
-   The design of ViLT is very similar to that of a standard Vision Transformer (ViT). The only difference is that the model includes additional embedding layers for the language modality.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vilt_architecture.jpg) ViLT architecture. Taken from the [original paper](https://arxiv.org/abs/2102.03334).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/dandelin/ViLT).

Tips:

-   The PyTorch version of this model is only available in torch 1.10 and higher.

## ViltConfig

### class transformers.ViltConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/configuration_vilt.py#L28)

( vocab\_size = 30522type\_vocab\_size = 2modality\_type\_vocab\_size = 2max\_position\_embeddings = 40hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-12image\_size = 384patch\_size = 32num\_channels = 3qkv\_bias = Truemax\_image\_length = -1tie\_word\_embeddings = Falsenum\_images = -1\*\*kwargs )

This is the configuration class to store the configuration of a `ViLTModel`. It is used to instantiate an ViLT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ViLT [dandelin/vilt-b32-mlm](https://huggingface.co/dandelin/vilt-b32-mlm) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ViLTModel, ViLTConfig

>>> 
>>> configuration = ViLTConfig()

>>> 
>>> model = ViLTModel(configuration)

>>> 
>>> configuration = model.config
```

## ViltFeatureExtractor

Preprocess an image or a batch of images.

## ViltImageProcessor

### class transformers.ViltImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/image_processing_vilt.py#L120)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Nonesize\_divisor: int = 32resample: Resampling = <Resampling.BICUBIC: 3>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: bool = True\*\*kwargs )

Constructs a ViLT image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/image_processing_vilt.py#L338)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Nonesize\_divisor: typing.Optional\[int\] = Noneresample: Resampling = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[transformers.image\_utils.ChannelDimension, str, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## ViltProcessor

### class transformers.ViltProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/processing_vilt.py#L27)

( image\_processor = Nonetokenizer = None\*\*kwargs )

Parameters

-   **image\_processor** (`ViltImageProcessor`) — An instance of [ViltImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltImageProcessor). The image processor is a required input.
-   **tokenizer** (`BertTokenizerFast`) — An instance of \[‘BertTokenizerFast\`\]. The tokenizer is a required input.

Constructs a ViLT processor which wraps a BERT tokenizer and ViLT image processor into a single processor.

[ViltProcessor](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltProcessor) offers all the functionalities of [ViltImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltImageProcessor) and [BertTokenizerFast](/docs/transformers/v4.34.0/en/model_doc/bert#transformers.BertTokenizerFast). See the docstring of [**call**()](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltProcessor.__call__) and `decode()` for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/processing_vilt.py#L63)

( imagestext: typing.Union\[str, typing.List\[str\], typing.List\[typing.List\[str\]\]\] = Noneadd\_special\_tokens: bool = Truepadding: typing.Union\[bool, str, transformers.utils.generic.PaddingStrategy\] = Falsetruncation: typing.Union\[bool, str, transformers.tokenization\_utils\_base.TruncationStrategy\] = Nonemax\_length: typing.Optional\[int\] = Nonestride: int = 0pad\_to\_multiple\_of: typing.Optional\[int\] = Nonereturn\_token\_type\_ids: typing.Optional\[bool\] = Nonereturn\_attention\_mask: typing.Optional\[bool\] = Nonereturn\_overflowing\_tokens: bool = Falsereturn\_special\_tokens\_mask: bool = Falsereturn\_offsets\_mapping: bool = Falsereturn\_length: bool = Falseverbose: bool = Truereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = None\*\*kwargs )

This method uses [ViltImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) method to prepare image(s) for the model, and [BertTokenizerFast.**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) to prepare text for the model.

Please refer to the docstring of the above two methods for more information.

## ViltModel

### class transformers.ViltModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L725)

( configadd\_pooling\_layer = True )

Parameters

-   **config** ([ViltConfig](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare ViLT Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`\_ subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L753)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Nonepixel\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneimage\_embeds: typing.Optional\[torch.FloatTensor\] = Noneimage\_token\_type\_idx: typing.Optional\[int\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [ViltModel](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import ViltProcessor, ViltModel
>>> from PIL import Image
>>> import requests

>>> 
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "hello world"

>>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
>>> model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")

>>> inputs = processor(image, text, return_tensors="pt")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## ViltForMaskedLM

### class transformers.ViltForMaskedLM

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L891)

( config )

Parameters

-   **config** ([ViltConfig](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViLT Model with a language modeling head on top as done during pretraining.

This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`\_ subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L909)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Nonepixel\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneimage\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

The [ViltForMaskedLM](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import ViltProcessor, ViltForMaskedLM
>>> import requests
>>> from PIL import Image
>>> import re
>>> import torch

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "a bunch of [MASK] laying on a [MASK]."

>>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
>>> model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")

>>> 
>>> encoding = processor(image, text, return_tensors="pt")

>>> 
>>> outputs = model(**encoding)

>>> tl = len(re.findall("\[MASK\]", text))
>>> inferred_token = [text]

>>> 
>>> with torch.no_grad():
...     for i in range(tl):
...         encoded = processor.tokenizer(inferred_token)
...         input_ids = torch.tensor(encoded.input_ids)
...         encoded = encoded["input_ids"][0][1:-1]
...         outputs = model(input_ids=input_ids, pixel_values=encoding.pixel_values)
...         mlm_logits = outputs.logits[0]  
...         
...         mlm_logits = mlm_logits[1 : input_ids.shape[1] - 1, :]
...         mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
...         
...         mlm_values[torch.tensor(encoded) != 103] = 0
...         select = mlm_values.argmax().item()
...         encoded[select] = mlm_ids[select].item()
...         inferred_token = [processor.decode(encoded)]

>>> selected_token = ""
>>> encoded = processor.tokenizer(inferred_token)
>>> output = processor.decode(encoded.input_ids[0], skip_special_tokens=True)
>>> print(output)
a bunch of cats laying on a couch.
```

## ViltForQuestionAnswering

### class transformers.ViltForQuestionAnswering

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L1067)

( config )

Parameters

-   **config** ([ViltConfig](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the \[CLS\] token) for visual question answering, e.g. for VQAv2.

This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`\_ subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L1085)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Nonepixel\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneimage\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [ViltForQuestionAnswering](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltForQuestionAnswering) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import ViltProcessor, ViltForQuestionAnswering
>>> import requests
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> text = "How many cats are there?"

>>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
>>> model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

>>> 
>>> encoding = processor(image, text, return_tensors="pt")

>>> 
>>> outputs = model(**encoding)
>>> logits = outputs.logits
>>> idx = logits.argmax(-1).item()
>>> print("Predicted answer:", model.config.id2label[idx])
Predicted answer: 2
```

## ViltForImagesAndTextClassification

### class transformers.ViltForImagesAndTextClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L1281)

( config )

Vilt Model transformer with a classifier head on top for natural language visual reasoning, e.g. NLVR2.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L1300)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Nonepixel\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneimage\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.vilt.modeling_vilt.ViltForImagesAndTextClassificationOutput` or `tuple(torch.FloatTensor)`

The [ViltForImagesAndTextClassification](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltForImagesAndTextClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import ViltProcessor, ViltForImagesAndTextClassification
>>> import requests
>>> from PIL import Image

>>> image1 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg", stream=True).raw)
>>> image2 = Image.open(requests.get("https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg", stream=True).raw)
>>> text = "The left image contains twice the number of dogs as the right image."

>>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")
>>> model = ViltForImagesAndTextClassification.from_pretrained("dandelin/vilt-b32-finetuned-nlvr2")

>>> 
>>> encoding = processor([image1, image2], text, return_tensors="pt")

>>> 
>>> outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))
>>> logits = outputs.logits
>>> idx = logits.argmax(-1).item()
>>> print("Predicted answer:", model.config.id2label[idx])
Predicted answer: True
```

## ViltForImageAndTextRetrieval

### class transformers.ViltForImageAndTextRetrieval

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L1180)

( config )

Parameters

-   **config** ([ViltConfig](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the \[CLS\] token) for image-to-text or text-to-image retrieval, e.g. MSCOCO and F30K.

This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`\_ subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L1192)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Nonepixel\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneimage\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

The [ViltForImageAndTextRetrieval](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltForImageAndTextRetrieval) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import ViltProcessor, ViltForImageAndTextRetrieval
>>> import requests
>>> from PIL import Image

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

>>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
>>> model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

>>> 
>>> scores = dict()
>>> for text in texts:
...     
...     encoding = processor(image, text, return_tensors="pt")
...     outputs = model(**encoding)
...     scores[text] = outputs.logits[0, :].item()
```

## ViltForTokenClassification

### class transformers.ViltForTokenClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L1423)

( config )

Parameters

-   **config** ([ViltConfig](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViLT Model with a token classification head on top (a linear layer on top of the final hidden-states of the text tokens) e.g. for Named-Entity-Recognition (NER) tasks.

This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`\_ subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vilt/modeling_vilt.py#L1436)

( input\_ids: typing.Optional\[torch.LongTensor\] = Noneattention\_mask: typing.Optional\[torch.FloatTensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Nonepixel\_mask: typing.Optional\[torch.LongTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneimage\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.TokenClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

The [ViltForTokenClassification](/docs/transformers/v4.34.0/en/model_doc/vilt#transformers.ViltForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.