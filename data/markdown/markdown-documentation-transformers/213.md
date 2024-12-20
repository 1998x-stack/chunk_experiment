# MGP-STR

## Overview

The MGP-STR model was proposed in [Multi-Granularity Prediction for Scene Text Recognition](https://arxiv.org/abs/2209.03592) by Peng Wang, Cheng Da, and Cong Yao. MGP-STR is a conceptually **simple** yet **powerful** vision Scene Text Recognition (STR) model, which is built upon the [Vision Transformer (ViT)](vit). To integrate linguistic knowledge, Multi-Granularity Prediction (MGP) strategy is proposed to inject information from the language modality into the model in an implicit way.

The abstract from the paper is the following:

_Scene text recognition (STR) has been an active research topic in computer vision for years. To tackle this challenging problem, numerous innovative methods have been successively proposed and incorporating linguistic knowledge into STR models has recently become a prominent trend. In this work, we first draw inspiration from the recent progress in Vision Transformer (ViT) to construct a conceptually simple yet powerful vision STR model, which is built upon ViT and outperforms previous state-of-the-art models for scene text recognition, including both pure vision models and language-augmented methods. To integrate linguistic knowledge, we further propose a Multi-Granularity Prediction strategy to inject information from the language modality into the model in an implicit way, i.e. , subword representations (BPE and WordPiece) widely-used in NLP are introduced into the output space, in addition to the conventional character level representation, while no independent language model (LM) is adopted. The resultant algorithm (termed MGP-STR) is able to push the performance envelop of STR to an even higher level. Specifically, it achieves an average recognition accuracy of 93.35% on standard benchmarks._

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/mgp_str_architecture.png) MGP-STR architecture. Taken from the [original paper](https://arxiv.org/abs/2209.03592).

Tips:

-   MGP-STR is trained on two synthetic datasets [MJSynth]((http://www.robots.ox.ac.uk/~vgg/data/text/)) (MJ) and SynthText([http://www.robots.ox.ac.uk/~vgg/data/scenetext/](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)) (ST) without fine-tuning on other datasets. It achieves state-of-the-art results on six standard Latin scene text benchmarks, including 3 regular text datasets (IC13, SVT, IIIT) and 3 irregular ones (IC15, SVTP, CUTE).
-   This model was contributed by [yuekun](https://huggingface.co/yuekun). The original code can be found [here](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR).

## Inference

[MgpstrModel](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrModel) accepts images as input and generates three types of predictions, which represent textual information at different granularities. The three types of predictions are fused to give the final prediction result.

The [ViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor) class is responsible for preprocessing the input image and [MgpstrTokenizer](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrTokenizer) decodes the generated character tokens to the target string. The [MgpstrProcessor](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrProcessor) wraps [ViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor) and [MgpstrTokenizer](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrTokenizer) into a single instance to both extract the input features and decode the predicted token ids.

-   Step-by-step Optical Character Recognition (OCR)

```
>>> from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition
>>> import requests
>>> from PIL import Image

>>> processor = MgpstrProcessor.from_pretrained('alibaba-damo/mgp-str-base')
>>> model = MgpstrForSceneTextRecognition.from_pretrained('alibaba-damo/mgp-str-base')

>>> 
>>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> pixel_values = processor(images=image, return_tensors="pt").pixel_values
>>> outputs = model(pixel_values)

>>> generated_text = processor.batch_decode(outputs.logits)['generated_text']
```

## MgpstrConfig

### class transformers.MgpstrConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/configuration_mgp_str.py#L28)

( image\_size = \[32, 128\]patch\_size = 4num\_channels = 3max\_token\_length = 27num\_character\_labels = 38num\_bpe\_labels = 50257num\_wordpiece\_labels = 30522hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12mlp\_ratio = 4.0qkv\_bias = Truedistilled = Falselayer\_norm\_eps = 1e-05drop\_rate = 0.0attn\_drop\_rate = 0.0drop\_path\_rate = 0.0output\_a3\_attentions = Falseinitializer\_range = 0.02\*\*kwargs )

This is the configuration class to store the configuration of an [MgpstrModel](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrModel). It is used to instantiate an MGP-STR model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the MGP-STR [alibaba-damo/mgp-str-base](https://huggingface.co/alibaba-damo/mgp-str-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import MgpstrConfig, MgpstrForSceneTextRecognition

>>> 
>>> configuration = MgpstrConfig()

>>> 
>>> model = MgpstrForSceneTextRecognition(configuration)

>>> 
>>> configuration = model.config
```

## MgpstrTokenizer

### class transformers.MgpstrTokenizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/tokenization_mgp_str.py#L38)

( vocab\_fileunk\_token = '\[GO\]'bos\_token = '\[GO\]'eos\_token = '\[s\]'pad\_token = '\[GO\]'\*\*kwargs )

Parameters

-   **vocab\_file** (`str`) — Path to the vocabulary file.
-   **unk\_token** (`str`, _optional_, defaults to `"[GO]"`) — The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.
-   **bos\_token** (`str`, _optional_, defaults to `"[GO]"`) — The beginning of sequence token.
-   **eos\_token** (`str`, _optional_, defaults to `"[s]"`) — The end of sequence token.
-   **pad\_token** (`str` or `tokenizers.AddedToken`, _optional_, , defaults to `"[GO]"`) — A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.

Construct a MGP-STR char tokenizer.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) which contains most of the main methods. Users should refer to this superclass for more information regarding those methods.

#### save\_vocabulary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/tokenization_mgp_str.py#L100)

( save\_directory: strfilename\_prefix: typing.Optional\[str\] = None )

## MgpstrProcessor

### class transformers.MgpstrProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/processing_mgp_str.py#L39)

( image\_processor = Nonetokenizer = None\*\*kwargs )

Parameters

-   **image\_processor** (`ViTImageProcessor`) — An instance of `ViTImageProcessor`. The image processor is a required input.
-   **tokenizer** ([MgpstrTokenizer](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrTokenizer)) — The tokenizer is a required input.

Constructs a MGP-STR processor which wraps an image processor and MGP-STR tokenizers into a single

[MgpstrProcessor](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrProcessor) offers all the functionalities of `ViTImageProcessor`\] and [MgpstrTokenizer](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrTokenizer). See the [**call**()](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrProcessor.__call__) and [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrProcessor.batch_decode) for more information.

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/processing_mgp_str.py#L78)

( text = Noneimages = Nonereturn\_tensors = None\*\*kwargs )

When used in normal mode, this method forwards all its arguments to ViTImageProcessor’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) and returns its output. This method also forwards the `text` and `kwargs` arguments to MgpstrTokenizer’s [**call**()](/docs/transformers/v4.34.0/en/model_doc/vits#transformers.VitsTokenizer.__call__) if `text` is not `None` to encode the text. Please refer to the doctsring of the above methods for more information.

#### batch\_decode

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/processing_mgp_str.py#L101)

( sequences ) → `Dict[str, any]`

Parameters

-   **sequences** (`torch.Tensor`) — List of tokenized input ids.

Dictionary of all the outputs of the decoded results. generated\_text (`List[str]`): The final results after fusion of char, bpe, and wp. scores (`List[float]`): The final scores after fusion of char, bpe, and wp. char\_preds (`List[str]`): The list of character decoded sentences. bpe\_preds (`List[str]`): The list of bpe decoded sentences. wp\_preds (`List[str]`): The list of wp decoded sentences.

Convert a list of lists of token ids into a list of strings by calling decode.

This method forwards all its arguments to PreTrainedTokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

## MgpstrModel

### class transformers.MgpstrModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/modeling_mgp_str.py#L372)

( config: MgpstrConfig )

Parameters

-   **config** ([MgpstrConfig](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare MGP-STR Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/modeling_mgp_str.py#L382)

( pixel\_values: FloatTensoroutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None )

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [ViTImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

The [MgpstrModel](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

## MgpstrForSceneTextRecognition

### class transformers.MgpstrForSceneTextRecognition

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/modeling_mgp_str.py#L424)

( config: MgpstrConfig )

Parameters

-   **config** ([MgpstrConfig](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

MGP-STR Model transformer with three classification heads on top (three A^3 modules and three linear layer on top of the transformer encoder output) for scene text recognition (STR) .

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/mgp_str/modeling_mgp_str.py#L442)

( pixel\_values: FloatTensoroutput\_attentions: typing.Optional\[bool\] = Noneoutput\_a3\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.mgp_str.modeling_mgp_str.MgpstrModelOutput` or `tuple(torch.FloatTensor)`

The [MgpstrForSceneTextRecognition](/docs/transformers/v4.34.0/en/model_doc/mgp-str#transformers.MgpstrForSceneTextRecognition) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import (
...     MgpstrProcessor,
...     MgpstrForSceneTextRecognition,
... )
>>> import requests
>>> from PIL import Image

>>> 
>>> url = "https://i.postimg.cc/ZKwLg2Gw/367-14.png"
>>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

>>> processor = MgpstrProcessor.from_pretrained("alibaba-damo/mgp-str-base")
>>> pixel_values = processor(images=image, return_tensors="pt").pixel_values

>>> model = MgpstrForSceneTextRecognition.from_pretrained("alibaba-damo/mgp-str-base")

>>> 
>>> outputs = model(pixel_values)
>>> out_strs = processor.batch_decode(outputs.logits)
>>> out_strs["generated_text"]
'["ticket"]'
```