# GroupViT

## Overview

The GroupViT model was proposed in [GroupViT: Semantic Segmentation Emerges from Text Supervision](https://arxiv.org/abs/2202.11094) by Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, Xiaolong Wang. Inspired by [CLIP](clip), GroupViT is a vision-language model that can perform zero-shot semantic segmentation on any given vocabulary categories.

The abstract from the paper is the following:

_Grouping and recognition are important components of visual scene understanding, e.g., for object detection and semantic segmentation. With end-to-end deep learning systems, grouping of image regions usually happens implicitly via top-down supervision from pixel-level recognition labels. Instead, in this paper, we propose to bring back the grouping mechanism into deep networks, which allows semantic segments to emerge automatically with only text supervision. We propose a hierarchical Grouping Vision Transformer (GroupViT), which goes beyond the regular grid structure representation and learns to group image regions into progressively larger arbitrary-shaped segments. We train GroupViT jointly with a text encoder on a large-scale image-text dataset via contrastive losses. With only text supervision and without any pixel-level annotations, GroupViT learns to group together semantic regions and successfully transfers to the task of semantic segmentation in a zero-shot manner, i.e., without any further fine-tuning. It achieves a zero-shot accuracy of 52.3% mIoU on the PASCAL VOC 2012 and 22.4% mIoU on PASCAL Context datasets, and performs competitively to state-of-the-art transfer-learning methods requiring greater levels of supervision._

Tips:

-   You may specify `output_segmentation=True` in the forward of `GroupViTModel` to get the segmentation logits of input texts.

This model was contributed by [xvjiarui](https://huggingface.co/xvjiarui). The TensorFlow version was contributed by [ariG23498](https://huggingface.co/ariG23498) with the help of [Yih-Dar SHIEH](https://huggingface.co/ydshieh), [Amy Roberts](https://huggingface.co/amyeroberts), and [Joao Gante](https://huggingface.co/joaogante). The original code can be found [here](https://github.com/NVlabs/GroupViT).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with GroupViT.

-   The quickest way to get started with GroupViT is by checking the [example notebooks](https://github.com/xvjiarui/GroupViT/blob/main/demo/GroupViT_hf_inference_notebook.ipynb) (which showcase zero-shot segmentation inference).
-   One can also check out the [HuggingFace Spaces demo](https://huggingface.co/spaces/xvjiarui/GroupViT) to play with GroupViT.

## GroupViTConfig

### class transformers.GroupViTConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/configuration_groupvit.py#L271)

( text\_config = Nonevision\_config = Noneprojection\_dim = 256projection\_intermediate\_dim = 4096logit\_scale\_init\_value = 2.6592\*\*kwargs )

Parameters

-   **text\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [GroupViTTextConfig](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTTextConfig).
-   **vision\_config** (`dict`, _optional_) — Dictionary of configuration options used to initialize [GroupViTVisionConfig](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTVisionConfig).
-   **projection\_dim** (`int`, _optional_, defaults to 256) — Dimentionality of text and vision projection layers.
-   **projection\_intermediate\_dim** (`int`, _optional_, defaults to 4096) — Dimentionality of intermediate layer of text and vision projection layers.
-   **logit\_scale\_init\_value** (`float`, _optional_, defaults to 2.6592) — The inital value of the _logit\_scale_ parameter. Default is used as per the original GroupViT implementation.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[GroupViTConfig](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTConfig) is the configuration class to store the configuration of a [GroupViTModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTModel). It is used to instantiate a GroupViT model according to the specified arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the GroupViT [nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

#### from\_text\_vision\_configs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/configuration_groupvit.py#L396)

( text\_config: GroupViTTextConfigvision\_config: GroupViTVisionConfig\*\*kwargs ) → [GroupViTConfig](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTConfig)

An instance of a configuration object

Instantiate a [GroupViTConfig](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTConfig) (or a derived class) from groupvit text model configuration and groupvit vision model configuration.

## GroupViTTextConfig

### class transformers.GroupViTTextConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/configuration_groupvit.py#L38)

( vocab\_size = 49408hidden\_size = 256intermediate\_size = 1024num\_hidden\_layers = 12num\_attention\_heads = 4max\_position\_embeddings = 77hidden\_act = 'quick\_gelu'layer\_norm\_eps = 1e-05dropout = 0.0attention\_dropout = 0.0initializer\_range = 0.02initializer\_factor = 1.0pad\_token\_id = 1bos\_token\_id = 49406eos\_token\_id = 49407\*\*kwargs )

This is the configuration class to store the configuration of a [GroupViTTextModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTTextModel). It is used to instantiate an GroupViT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the GroupViT [nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import GroupViTTextConfig, GroupViTTextModel

>>> 
>>> configuration = GroupViTTextConfig()

>>> model = GroupViTTextModel(configuration)

>>> 
>>> configuration = model.config
```

## GroupViTVisionConfig

### class transformers.GroupViTVisionConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/configuration_groupvit.py#L146)

( hidden\_size = 384intermediate\_size = 1536depths = \[6, 3, 3\]num\_hidden\_layers = 12num\_group\_tokens = \[64, 8, 0\]num\_output\_groups = \[64, 8, 8\]num\_attention\_heads = 6image\_size = 224patch\_size = 16num\_channels = 3hidden\_act = 'gelu'layer\_norm\_eps = 1e-05dropout = 0.0attention\_dropout = 0.0initializer\_range = 0.02initializer\_factor = 1.0assign\_eps = 1.0assign\_mlp\_ratio = \[0.5, 4\]\*\*kwargs )

This is the configuration class to store the configuration of a [GroupViTVisionModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTVisionModel). It is used to instantiate an GroupViT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the GroupViT [nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import GroupViTVisionConfig, GroupViTVisionModel

>>> 
>>> configuration = GroupViTVisionConfig()

>>> model = GroupViTVisionModel(configuration)

>>> 
>>> configuration = model.config
```

## GroupViTModel

### class transformers.GroupViTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_groupvit.py#L1346)

( config: GroupViTConfig )

Parameters

-   **config** ([GroupViTConfig](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_groupvit.py#L1488)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonereturn\_loss: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneoutput\_segmentation: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.groupvit.modeling_groupvit.GroupViTModelOutput` or `tuple(torch.FloatTensor)`

The [GroupViTModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, GroupViTModel

>>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_groupvit.py#L1392)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → text\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The [GroupViTModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import CLIPTokenizer, GroupViTModel

>>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_groupvit.py#L1439)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

Parameters

-   **pixel\_values** (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`) — Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor). See [CLIPImageProcessor.**call**()](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTFeatureExtractor.__call__) for details.
-   **output\_attentions** (`bool`, _optional_) — Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.
-   **output\_hidden\_states** (`bool`, _optional_) — Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.
-   **return\_dict** (`bool`, _optional_) — Whether or not to return a [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

Returns

image\_features (`torch.FloatTensor` of shape `(batch_size, output_dim`)

The image embeddings obtained by applying the projection layer to the pooled output of [GroupViTVisionModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTVisionModel).

The [GroupViTModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, GroupViTModel

>>> model = GroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> image_features = model.get_image_features(**inputs)
```

## GroupViTTextModel

### class transformers.GroupViTTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_groupvit.py#L1182)

( config: GroupViTTextConfig )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_groupvit.py#L1197)

( input\_ids: typing.Optional\[torch.Tensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [GroupViTTextModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import CLIPTokenizer, GroupViTTextModel

>>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> model = GroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## GroupViTVisionModel

### class transformers.GroupViTVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_groupvit.py#L1293)

( config: GroupViTVisionConfig )

The [GroupViTVisionModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, GroupViTVisionModel

>>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> model = GroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="pt")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## TFGroupViTModel

### class transformers.TFGroupViTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1728)

( \*args\*\*kwargs )

Parameters

-   **config** ([GroupViTConfig](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.GroupViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TF 2.0 models accepts two formats as inputs:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional arguments.

This second option is useful when using `tf.keras.Model.fit` method which currently requires having all the tensors in the first argument of the model call function: `model(inputs)`.

If you choose this second option, there are three possibilities you can use to gather all the input Tensors in the first positional argument :

-   a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1820)

( input\_ids: TFModelInputType | None = Nonepixel\_values: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Nonereturn\_loss: Optional\[bool\] = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Noneoutput\_segmentation: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.models.groupvit.modeling_tf_groupvit.TFGroupViTModelOutput` or `tuple(tf.Tensor)`

The [TFGroupViTModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.TFGroupViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFGroupViTModel
>>> import tensorflow as tf

>>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
... )

>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = tf.math.softmax(logits_per_image, axis=1)  
```

#### get\_text\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1736)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → text\_features (`tf.Tensor` of shape `(batch_size, output_dim`)

The [TFGroupViTModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.TFGroupViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import CLIPTokenizer, TFGroupViTModel

>>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
>>> text_features = model.get_text_features(**inputs)
```

#### get\_image\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1777)

( pixel\_values: TFModelInputType | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → image\_features (`tf.Tensor` of shape `(batch_size, output_dim`)

The [TFGroupViTModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.TFGroupViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFGroupViTModel

>>> model = TFGroupViTModel.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="tf")

>>> image_features = model.get_image_features(**inputs)
```

## TFGroupViTTextModel

### class transformers.TFGroupViTTextModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1620)

( \*args\*\*kwargs )

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1629)

( input\_ids: TFModelInputType | None = Noneattention\_mask: np.ndarray | tf.Tensor | None = Noneposition\_ids: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or `tuple(tf.Tensor)`

The [TFGroupViTTextModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.TFGroupViTTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import CLIPTokenizer, TFGroupViTTextModel

>>> tokenizer = CLIPTokenizer.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> model = TFGroupViTTextModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```

## TFGroupViTVisionModel

### class transformers.TFGroupViTVisionModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/groupvit/modeling_tf_groupvit.py#L1673)

( \*args\*\*kwargs )

The [TFGroupViTVisionModel](/docs/transformers/v4.34.0/en/model_doc/groupvit#transformers.TFGroupViTVisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import AutoProcessor, TFGroupViTVisionModel

>>> processor = AutoProcessor.from_pretrained("nvidia/groupvit-gcc-yfcc")
>>> model = TFGroupViTVisionModel.from_pretrained("nvidia/groupvit-gcc-yfcc")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = processor(images=image, return_tensors="tf")

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
>>> pooled_output = outputs.pooler_output  
```