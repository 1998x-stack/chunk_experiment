# SegFormer

## Overview

The SegFormer model was proposed in [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) by Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo. The model consists of a hierarchical Transformer encoder and a lightweight all-MLP decode head to achieve great results on image segmentation benchmarks such as ADE20K and Cityscapes.

The abstract from the paper is the following:

_We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perception (MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a novel hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding, thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution differs from training. 2) SegFormer avoids complex decoders. The proposed MLP decoder aggregates information from different layers, and thus combining both local attention and global attention to render powerful representations. We show that this simple and lightweight design is the key to efficient segmentation on Transformers. We scale our approach up to obtain a series of models from SegFormer-B0 to SegFormer-B5, reaching significantly better performance and efficiency than previous counterparts. For example, SegFormer-B4 achieves 50.3% mIoU on ADE20K with 64M parameters, being 5x smaller and 2.2% better than the previous best method. Our best model, SegFormer-B5, achieves 84.0% mIoU on Cityscapes validation set and shows excellent zero-shot robustness on Cityscapes-C._

The figure below illustrates the architecture of SegFormer. Taken from the [original paper](https://arxiv.org/abs/2105.15203).

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/segformer_architecture.png)

This model was contributed by [nielsr](https://huggingface.co/nielsr). The TensorFlow version of the model was contributed by [sayakpaul](https://huggingface.co/sayakpaul). The original code can be found [here](https://github.com/NVlabs/SegFormer).

Tips:

-   SegFormer consists of a hierarchical Transformer encoder, and a lightweight all-MLP decoder head. [SegformerModel](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerModel) is the hierarchical Transformer encoder (which in the paper is also referred to as Mix Transformer or MiT). [SegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) adds the all-MLP decoder head on top to perform semantic segmentation of images. In addition, there’s [SegformerForImageClassification](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForImageClassification) which can be used to - you guessed it - classify images. The authors of SegFormer first pre-trained the Transformer encoder on ImageNet-1k to classify images. Next, they throw away the classification head, and replace it by the all-MLP decode head. Next, they fine-tune the model altogether on ADE20K, Cityscapes and COCO-stuff, which are important benchmarks for semantic segmentation. All checkpoints can be found on the [hub](https://huggingface.co/models?other=segformer).
-   The quickest way to get started with SegFormer is by checking the [example notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SegFormer) (which showcase both inference and fine-tuning on custom data). One can also check out the [blog post](https://huggingface.co/blog/fine-tune-segformer) introducing SegFormer and illustrating how it can be fine-tuned on custom data.
-   TensorFlow users should refer to [this repository](https://github.com/deep-diver/segformer-tf-transformers) that shows off-the-shelf inference and fine-tuning.
-   One can also check out [this interactive demo on Hugging Face Spaces](https://huggingface.co/spaces/chansung/segformer-tf-transformers) to try out a SegFormer model on custom images.
-   SegFormer works on any input size, as it pads the input to be divisible by `config.patch_sizes`.
-   One can use [SegformerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerImageProcessor) to prepare images and corresponding segmentation maps for the model. Note that this image processor is fairly basic and does not include all data augmentations used in the original paper. The original preprocessing pipelines (for the ADE20k dataset for instance) can be found [here](https://github.com/NVlabs/SegFormer/blob/master/local_configs/_base_/datasets/ade20k_repeat.py). The most important preprocessing step is that images and segmentation maps are randomly cropped and padded to the same size, such as 512x512 or 640x640, after which they are normalized.
-   One additional thing to keep in mind is that one can initialize [SegformerImageProcessor](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerImageProcessor) with `reduce_labels` set to `True` or `False`. In some datasets (like ADE20k), the 0 index is used in the annotated segmentation maps for background. However, ADE20k doesn’t include the “background” class in its 150 labels. Therefore, `reduce_labels` is used to reduce all labels by 1, and to make sure no loss is computed for the background class (i.e. it replaces 0 in the annotated maps by 255, which is the _ignore\_index_ of the loss function used by [SegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation)). However, other datasets use the 0 index as background class and include this class as part of all labels. In that case, `reduce_labels` should be set to `False`, as loss should also be computed for the background class.
-   As most models, SegFormer comes in different sizes, the details of which can be found in the table below (taken from Table 7 of the [original paper](https://arxiv.org/abs/2105.15203)).

| **Model variant** | **Depths** | **Hidden sizes** | **Decoder hidden size** | **Params (M)** | **ImageNet-1k Top 1** |
| --- | --- | --- | --- | --- | --- |
| MiT-b0 | \[2, 2, 2, 2\] | \[32, 64, 160, 256\] | 256 | 3.7 | 70.5 |
| MiT-b1 | \[2, 2, 2, 2\] | \[64, 128, 320, 512\] | 256 | 14.0 | 78.7 |
| MiT-b2 | \[3, 4, 6, 3\] | \[64, 128, 320, 512\] | 768 | 25.4 | 81.6 |
| MiT-b3 | \[3, 4, 18, 3\] | \[64, 128, 320, 512\] | 768 | 45.2 | 83.1 |
| MiT-b4 | \[3, 8, 27, 3\] | \[64, 128, 320, 512\] | 768 | 62.6 | 83.6 |
| MiT-b5 | \[3, 6, 40, 3\] | \[64, 128, 320, 512\] | 768 | 82.0 | 83.8 |

Note that MiT in the above table refers to the Mix Transformer encoder backbone introduced in SegFormer. For SegFormer’s results on the segmentation datasets like ADE20k, refer to the [paper](https://arxiv.org/abs/2105.15203).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SegFormer.

-   [SegformerForImageClassification](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   [Image classification task guide](../tasks/image_classification)

Semantic segmentation:

-   [SegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/semantic-segmentation).
-   A blog on fine-tuning SegFormer on a custom dataset can be found [here](https://huggingface.co/blog/fine-tune-segformer).
-   More demo notebooks on SegFormer (both inference + fine-tuning on a custom dataset) can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/SegFormer).
-   [TFSegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.TFSegformerForSemanticSegmentation) is supported by this [example notebook](https://github.com/huggingface/notebooks/blob/main/examples/semantic_segmentation-tf.ipynb).
-   [Semantic segmentation task guide](../tasks/semantic_segmentation)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## SegformerConfig

### class transformers.SegformerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/configuration_segformer.py#L38)

( num\_channels = 3num\_encoder\_blocks = 4depths = \[2, 2, 2, 2\]sr\_ratios = \[8, 4, 2, 1\]hidden\_sizes = \[32, 64, 160, 256\]patch\_sizes = \[7, 3, 3, 3\]strides = \[4, 2, 2, 2\]num\_attention\_heads = \[1, 2, 5, 8\]mlp\_ratios = \[4, 4, 4, 4\]hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0classifier\_dropout\_prob = 0.1initializer\_range = 0.02drop\_path\_rate = 0.1layer\_norm\_eps = 1e-06decoder\_hidden\_size = 256semantic\_loss\_ignore\_index = 255\*\*kwargs )

This is the configuration class to store the configuration of a [SegformerModel](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerModel). It is used to instantiate an SegFormer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the SegFormer [nvidia/segformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import SegformerModel, SegformerConfig

>>> 
>>> configuration = SegformerConfig()

>>> 
>>> model = SegformerModel(configuration)

>>> 
>>> configuration = model.config
```

## SegformerFeatureExtractor

( imagessegmentation\_maps = None\*\*kwargs )

Preprocesses a batch of images and optionally segmentation maps.

Overrides the `__call__` method of the `Preprocessor` class so that both images and segmentation maps can be passed in as positional arguments.

( outputstarget\_sizes: typing.List\[typing.Tuple\] = None ) → semantic\_segmentation

Parameters

-   **outputs** ([SegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple]` of length `batch_size`, _optional_) — List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

`List[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [SegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## SegformerImageProcessor

### class transformers.SegformerImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/image_processing_segformer.py#L49)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_reduce\_labels: bool = False\*\*kwargs )

Constructs a Segformer image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/image_processing_segformer.py#L305)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]segmentation\_maps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\], NoneType\] = Nonedo\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample: Resampling = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_reduce\_labels: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

#### post\_process\_semantic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/image_processing_segformer.py#L441)

( outputstarget\_sizes: typing.List\[typing.Tuple\] = None ) → semantic\_segmentation

Parameters

-   **outputs** ([SegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple]` of length `batch_size`, _optional_) — List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

Returns

semantic\_segmentation

`List[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [SegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## SegformerModel

### class transformers.SegformerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_segformer.py#L510)

( config )

Parameters

-   **config** ([SegformerConfig](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

The [SegformerModel](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, SegformerModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
>>> model = SegformerModel.from_pretrained("nvidia/mit-b0")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 256, 16, 16]
```

## SegformerDecodeHead

### class transformers.SegformerDecodeHead

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_segformer.py#L681)

( config )

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_segformer.py#L706)

( encoder\_hidden\_states: FloatTensor )

## SegformerForImageClassification

### class transformers.SegformerForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_segformer.py#L575)

( config )

Parameters

-   **config** ([SegformerConfig](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden states) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_segformer.py#L588)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.segformer.modeling_segformer.SegFormerImageClassifierOutput` or `tuple(torch.FloatTensor)`

The [SegformerForImageClassification](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, SegformerForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
>>> model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## SegformerForSemanticSegmentation

### class transformers.SegformerForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_segformer.py#L743)

( config )

Parameters

-   **config** ([SegformerConfig](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_segformer.py#L752)

( pixel\_values: FloatTensorlabels: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

The [SegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
>>> from PIL import Image
>>> import requests

>>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
>>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits  
>>> list(logits.shape)
[1, 150, 128, 128]
```

## TFSegformerDecodeHead

### class transformers.TFSegformerDecodeHead

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_tf_segformer.py#L690)

( \*args\*\*kwargs )

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_tf_segformer.py#L712)

( encoder\_hidden\_states: tf.Tensortraining: bool = False )

## TFSegformerModel

### class transformers.TFSegformerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_tf_segformer.py#L574)

( \*args\*\*kwargs )

Parameters

-   **config** ([SegformerConfig](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

The [TFSegformerModel](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.TFSegformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFSegformerModel
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
>>> model = TFSegformerModel.from_pretrained("nvidia/mit-b0")

>>> inputs = image_processor(image, return_tensors="tf")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 256, 16, 16]
```

## TFSegformerForImageClassification

### class transformers.TFSegformerForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_tf_segformer.py#L616)

( \*args\*\*kwargs )

Parameters

-   **config** ([SegformerConfig](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden states) e.g. for ImageNet.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

The [TFSegformerForImageClassification](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.TFSegformerForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFSegformerForImageClassification
>>> import tensorflow as tf
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
>>> model = TFSegformerForImageClassification.from_pretrained("nvidia/mit-b0")

>>> inputs = image_processor(image, return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> predicted_label = int(tf.math.argmax(logits, axis=-1))
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## TFSegformerForSemanticSegmentation

### class transformers.TFSegformerForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_tf_segformer.py#L749)

( \*args\*\*kwargs )

Parameters

-   **config** ([SegformerConfig](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.SegformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/segformer/modeling_tf_segformer.py#L775)

( pixel\_values: tf.Tensorlabels: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = None ) → `transformers.modeling_tf_outputs.TFSemanticSegmenterOutput` or `tuple(tf.Tensor)`

The [TFSegformerForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/segformer#transformers.TFSegformerForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TFSegformerForSemanticSegmentation
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
>>> model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

>>> inputs = image_processor(images=image, return_tensors="tf")
>>> outputs = model(**inputs, training=False)
>>> 
>>> logits = outputs.logits
>>> list(logits.shape)
[1, 150, 128, 128]
```