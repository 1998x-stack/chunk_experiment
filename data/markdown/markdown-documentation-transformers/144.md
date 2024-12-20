# DETR

## Overview

The DETR model was proposed in [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) by Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov and Sergey Zagoruyko. DETR consists of a convolutional backbone followed by an encoder-decoder Transformer which can be trained end-to-end for object detection. It greatly simplifies a lot of the complexity of models like Faster-R-CNN and Mask-R-CNN, which use things like region proposals, non-maximum suppression procedure and anchor generation. Moreover, DETR can also be naturally extended to perform panoptic segmentation, by simply adding a mask head on top of the decoder outputs.

The abstract from the paper is the following:

_We present a new method that views object detection as a direct set prediction problem. Our approach streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task. The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. The new model is conceptually simple and does not require a specialized library, unlike many other modern detectors. DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster RCNN baseline on the challenging COCO object detection dataset. Moreover, DETR can be easily generalized to produce panoptic segmentation in a unified manner. We show that it significantly outperforms competitive baselines._

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/facebookresearch/detr).

Here’s a TLDR explaining how [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) works:

First, an image is sent through a pre-trained convolutional backbone (in the paper, the authors use ResNet-50/ResNet-101). Let’s assume we also add a batch dimension. This means that the input to the backbone is a tensor of shape `(batch_size, 3, height, width)`, assuming the image has 3 color channels (RGB). The CNN backbone outputs a new lower-resolution feature map, typically of shape `(batch_size, 2048, height/32, width/32)`. This is then projected to match the hidden dimension of the Transformer of DETR, which is `256` by default, using a `nn.Conv2D` layer. So now, we have a tensor of shape `(batch_size, 256, height/32, width/32).` Next, the feature map is flattened and transposed to obtain a tensor of shape `(batch_size, seq_len, d_model)` = `(batch_size, width/32*height/32, 256)`. So a difference with NLP models is that the sequence length is actually longer than usual, but with a smaller `d_model` (which in NLP is typically 768 or higher).

Next, this is sent through the encoder, outputting `encoder_hidden_states` of the same shape (you can consider these as image features). Next, so-called **object queries** are sent through the decoder. This is a tensor of shape `(batch_size, num_queries, d_model)`, with `num_queries` typically set to 100 and initialized with zeros. These input embeddings are learnt positional encodings that the authors refer to as object queries, and similarly to the encoder, they are added to the input of each attention layer. Each object query will look for a particular object in the image. The decoder updates these embeddings through multiple self-attention and encoder-decoder attention layers to output `decoder_hidden_states` of the same shape: `(batch_size, num_queries, d_model)`. Next, two heads are added on top for object detection: a linear layer for classifying each object query into one of the objects or “no object”, and a MLP to predict bounding boxes for each query.

The model is trained using a **bipartite matching loss**: so what we actually do is compare the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a “no object” as class and “no bounding box” as bounding box). The [Hungarian matching algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) is used to find an optimal one-to-one mapping of each of the N queries to each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and [generalized IoU loss](https://giou.stanford.edu/) (for the bounding boxes) are used to optimize the parameters of the model.

DETR can be naturally extended to perform panoptic segmentation (which unifies semantic segmentation and instance segmentation). [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) adds a segmentation mask head on top of [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection). The mask head can be trained either jointly, or in a two steps process, where one first trains a [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) model to detect bounding boxes around both “things” (instances) and “stuff” (background things like trees, roads, sky), then freeze all the weights and train only the mask head for 25 epochs. Experimentally, these two approaches give similar results. Note that predicting boxes is required for the training to be possible, since the Hungarian matching is computed using distances between boxes.

Tips:

-   DETR uses so-called **object queries** to detect objects in an image. The number of queries determines the maximum number of objects that can be detected in a single image, and is set to 100 by default (see parameter `num_queries` of [DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig)). Note that it’s good to have some slack (in COCO, the authors used 100, while the maximum number of objects in a COCO image is ~70).
-   The decoder of DETR updates the query embeddings in parallel. This is different from language models like GPT-2, which use autoregressive decoding instead of parallel. Hence, no causal attention mask is used.
-   DETR adds position embeddings to the hidden states at each self-attention and cross-attention layer before projecting to queries and keys. For the position embeddings of the image, one can choose between fixed sinusoidal or learned absolute position embeddings. By default, the parameter `position_embedding_type` of [DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig) is set to `"sine"`.
-   During training, the authors of DETR did find it helpful to use auxiliary losses in the decoder, especially to help the model output the correct number of objects of each class. If you set the parameter `auxiliary_loss` of [DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig) to `True`, then prediction feedforward neural networks and Hungarian losses are added after each decoder layer (with the FFNs sharing parameters).
-   If you want to train the model in a distributed environment across multiple nodes, then one should update the _num\_boxes_ variable in the _DetrLoss_ class of _modeling\_detr.py_. When training on multiple nodes, this should be set to the average number of target boxes across all nodes, as can be seen in the original implementation [here](https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/detr.py#L227-L232).
-   [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) and [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) can be initialized with any convolutional backbone available in the [timm library](https://github.com/rwightman/pytorch-image-models). Initializing with a MobileNet backbone for example can be done by setting the `backbone` attribute of [DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig) to `"tf_mobilenetv3_small_075"`, and then initializing the model with that config.
-   DETR resizes the input images such that the shortest side is at least a certain amount of pixels while the longest is at most 1333 pixels. At training time, scale augmentation is used such that the shortest side is randomly set to at least 480 and at most 800 pixels. At inference time, the shortest side is set to 800. One can use [DetrImageProcessor](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrImageProcessor) to prepare images (and optional annotations in COCO format) for the model. Due to this resizing, images in a batch can have different sizes. DETR solves this by padding images up to the largest size in a batch, and by creating a pixel mask that indicates which pixels are real/which are padding. Alternatively, one can also define a custom `collate_fn` in order to batch images together, using `~transformers.DetrImageProcessor.pad_and_create_pixel_mask`.
-   The size of the images will determine the amount of memory being used, and will thus determine the `batch_size`. It is advised to use a batch size of 2 per GPU. See [this Github thread](https://github.com/facebookresearch/detr/issues/150) for more info.

There are three ways to instantiate a DETR model (depending on what you prefer):

Option 1: Instantiate DETR with pre-trained weights for entire model

```
>>> from transformers import DetrForObjectDetection

>>> model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
```

Option 2: Instantiate DETR with randomly initialized weights for Transformer, but pre-trained weights for backbone

```
>>> from transformers import DetrConfig, DetrForObjectDetection

>>> config = DetrConfig()
>>> model = DetrForObjectDetection(config)
```

Option 3: Instantiate DETR with randomly initialized weights for backbone + Transformer

```
>>> config = DetrConfig(use_pretrained_backbone=False)
>>> model = DetrForObjectDetection(config)
```

As a summary, consider the following table:

| Task | Object detection | Instance segmentation | Panoptic segmentation |
| --- | --- | --- | --- |
| **Description** | Predicting bounding boxes and class labels around objects in an image | Predicting masks around objects (i.e. instances) in an image | Predicting masks around both objects (i.e. instances) as well as “stuff” (i.e. background things like trees and roads) in an image |
| **Model** | [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) | [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) | [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) |
| **Example dataset** | COCO detection | COCO detection, COCO panoptic | COCO panoptic |
| **Format of annotations to provide to** [DetrImageProcessor](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrImageProcessor) | {‘image\_id’: `int`, ‘annotations’: `List[Dict]`} each Dict being a COCO object annotation | {‘image\_id’: `int`, ‘annotations’: `List[Dict]`} (in case of COCO detection) or {‘file\_name’: `str`, ‘image\_id’: `int`, ‘segments\_info’: `List[Dict]`} (in case of COCO panoptic) | {‘file\_name’: `str`, ‘image\_id’: `int`, ‘segments\_info’: `List[Dict]`} and masks\_path (path to directory containing PNG files of the masks) |
| **Postprocessing** (i.e. converting the output of the model to COCO API) | `post_process()` | `post_process_segmentation()` | `post_process_segmentation()`, `post_process_panoptic()` |
| **evaluators** | `CocoEvaluator` with `iou_types="bbox"` | `CocoEvaluator` with `iou_types="bbox"` or `"segm"` | `CocoEvaluator` with `iou_tupes="bbox"` or `"segm"`, `PanopticEvaluator` |

In short, one should prepare the data either in COCO detection or COCO panoptic format, then use [DetrImageProcessor](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrImageProcessor) to create `pixel_values`, `pixel_mask` and optional `labels`, which can then be used to train (or fine-tune) a model. For evaluation, one should first convert the outputs of the model using one of the postprocessing methods of [DetrImageProcessor](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrImageProcessor). These can be be provided to either `CocoEvaluator` or `PanopticEvaluator`, which allow you to calculate metrics like mean Average Precision (mAP) and Panoptic Quality (PQ). The latter objects are implemented in the [original repository](https://github.com/facebookresearch/detr). See the [example notebooks](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR) for more info regarding evaluation.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with DETR.

-   All example notebooks illustrating fine-tuning [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) and [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) on a custom dataset an be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/DETR).
-   See also: [Object detection task guide](../tasks/object_detection)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DETR specific outputs

### class transformers.models.detr.modeling\_detr.DetrModelOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/modeling_detr.py#L94)

( last\_hidden\_state: FloatTensor = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.FloatTensor\]\]\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneintermediate\_hidden\_states: typing.Optional\[torch.FloatTensor\] = None )

Base class for outputs of the DETR encoder-decoder model. This class adds one attribute to Seq2SeqModelOutput, namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them gone through a layernorm. This is useful when training the model with auxiliary decoding losses.

### class transformers.models.detr.modeling\_detr.DetrObjectDetectionOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/modeling_detr.py#L134)

( loss: typing.Optional\[torch.FloatTensor\] = Noneloss\_dict: typing.Optional\[typing.Dict\] = Nonelogits: FloatTensor = Nonepred\_boxes: FloatTensor = Noneauxiliary\_outputs: typing.Optional\[typing.List\[typing.Dict\]\] = Nonelast\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection).

### class transformers.models.detr.modeling\_detr.DetrSegmentationOutput

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/modeling_detr.py#L197)

( loss: typing.Optional\[torch.FloatTensor\] = Noneloss\_dict: typing.Optional\[typing.Dict\] = Nonelogits: FloatTensor = Nonepred\_boxes: FloatTensor = Nonepred\_masks: FloatTensor = Noneauxiliary\_outputs: typing.Optional\[typing.List\[typing.Dict\]\] = Nonelast\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonedecoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Nonecross\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_last\_hidden\_state: typing.Optional\[torch.FloatTensor\] = Noneencoder\_hidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneencoder\_attentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Output type of [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation).

## DetrConfig

### class transformers.DetrConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/configuration_detr.py#L36)

( use\_timm\_backbone = Truebackbone\_config = Nonenum\_channels = 3num\_queries = 100encoder\_layers = 6encoder\_ffn\_dim = 2048encoder\_attention\_heads = 8decoder\_layers = 6decoder\_ffn\_dim = 2048decoder\_attention\_heads = 8encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0is\_encoder\_decoder = Trueactivation\_function = 'relu'd\_model = 256dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02init\_xavier\_std = 1.0auxiliary\_loss = Falseposition\_embedding\_type = 'sine'backbone = 'resnet50'use\_pretrained\_backbone = Truedilation = Falseclass\_cost = 1bbox\_cost = 5giou\_cost = 2mask\_loss\_coefficient = 1dice\_loss\_coefficient = 1bbox\_loss\_coefficient = 5giou\_loss\_coefficient = 2eos\_coefficient = 0.1\*\*kwargs )

This is the configuration class to store the configuration of a [DetrModel](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrModel). It is used to instantiate a DETR model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the DETR [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import DetrConfig, DetrModel

>>> 
>>> configuration = DetrConfig()

>>> 
>>> model = DetrModel(configuration)

>>> 
>>> configuration = model.config
```

#### from\_backbone\_config

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/configuration_detr.py#L238)

( backbone\_config: PretrainedConfig\*\*kwargs ) → [DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig)

Parameters

-   **backbone\_config** ([PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig)) — The backbone configuration.

An instance of a configuration object

Instantiate a [DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig) (or a derived class) from a pre-trained backbone model configuration.

## DetrImageProcessor

### class transformers.DetrImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/image_processing_detr.py#L746)

( format: typing.Union\[str, transformers.models.detr.image\_processing\_detr.AnnotionFormat\] = <AnnotionFormat.COCO\_DETECTION: 'coco\_detection'>do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\]\] = Noneimage\_std: typing.Union\[float, typing.List\[float\]\] = Nonedo\_pad: bool = True\*\*kwargs )

Constructs a Detr image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/image_processing_detr.py#L1074)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]annotations: typing.Union\[typing.Dict\[str, typing.Union\[int, str, typing.List\[typing.Dict\]\]\], typing.List\[typing.Dict\[str, typing.Union\[int, str, typing.List\[typing.Dict\]\]\]\], NoneType\] = Nonereturn\_segmentation\_masks: bool = Nonemasks\_path: typing.Union\[str, pathlib.Path, NoneType\] = Nonedo\_resize: typing.Optional\[bool\] = Nonesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Union\[int, float, NoneType\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_pad: typing.Optional\[bool\] = Noneformat: typing.Union\[str, transformers.models.detr.image\_processing\_detr.AnnotionFormat, NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or a batch of images so that it can be used by the model.

#### post\_process\_object\_detection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/image_processing_detr.py#L1591)

( outputsthreshold: float = 0.5target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = None ) → `List[Dict]`

Parameters

-   **outputs** (`DetrObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size `(height, width)` of each image in the batch. If unset, predictions will not be resized.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

#### post\_process\_semantic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/image_processing_detr.py#L1644)

( outputstarget\_sizes: typing.List\[typing.Tuple\[int, int\]\] = None ) → `List[torch.Tensor]`

Parameters

-   **outputs** ([DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple[int, int]]`, _optional_) — A list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If unset, predictions will not be resized.

Returns

`List[torch.Tensor]`

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) into semantic segmentation maps. Only supports PyTorch.

#### post\_process\_instance\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/image_processing_detr.py#L1692)

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8target\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = Nonereturn\_coco\_annotation: typing.Optional\[bool\] = False ) → `List[Dict]`

Converts the output of [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) into instance segmentation predictions. Only supports PyTorch.

#### post\_process\_panoptic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/image_processing_detr.py#L1776)

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8label\_ids\_to\_fuse: typing.Optional\[typing.Set\[int\]\] = Nonetarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[Dict]`

Converts the output of [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) into image panoptic segmentation predictions. Only supports PyTorch.

## DetrFeatureExtractor

Preprocess an image or a batch of images.

( outputsthreshold: float = 0.5target\_sizes: typing.Union\[transformers.utils.generic.TensorType, typing.List\[typing.Tuple\]\] = None ) → `List[Dict]`

Parameters

-   **outputs** (`DetrObjectDetectionOutput`) — Raw outputs of the model.
-   **threshold** (`float`, _optional_) — Score threshold to keep object detection predictions.
-   **target\_sizes** (`torch.Tensor` or `List[Tuple[int, int]]`, _optional_) — Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size `(height, width)` of each image in the batch. If unset, predictions will not be resized.

A list of dictionaries, each dictionary containing the scores, labels and boxes for an image in the batch as predicted by the model.

Converts the raw output of [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) into final bounding boxes in (top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y) format. Only supports PyTorch.

( outputstarget\_sizes: typing.List\[typing.Tuple\[int, int\]\] = None ) → `List[torch.Tensor]`

Parameters

-   **outputs** ([DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple[int, int]]`, _optional_) — A list of tuples (`Tuple[int, int]`) containing the target size (height, width) of each image in the batch. If unset, predictions will not be resized.

A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) into semantic segmentation maps. Only supports PyTorch.

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8target\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = Nonereturn\_coco\_annotation: typing.Optional\[bool\] = False ) → `List[Dict]`

Converts the output of [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) into instance segmentation predictions. Only supports PyTorch.

( outputsthreshold: float = 0.5mask\_threshold: float = 0.5overlap\_mask\_area\_threshold: float = 0.8label\_ids\_to\_fuse: typing.Optional\[typing.Set\[int\]\] = Nonetarget\_sizes: typing.Union\[typing.List\[typing.Tuple\[int, int\]\], NoneType\] = None ) → `List[Dict]`

Converts the output of [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) into image panoptic segmentation predictions. Only supports PyTorch.

## DetrModel

### class transformers.DetrModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/modeling_detr.py#L1325)

( config: DetrConfig )

Parameters

-   **config** ([DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/modeling_detr.py#L1359)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.detr.modeling\_detr.DetrModelOutput](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrModelOutput) or `tuple(torch.FloatTensor)`

The [DetrModel](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DetrModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
>>> model = DetrModel.from_pretrained("facebook/detr-resnet-50")

>>> 
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> 
>>> outputs = model(**inputs)

>>> 
>>> 
>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 100, 256]
```

## DetrForObjectDetection

### class transformers.DetrForObjectDetection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/modeling_detr.py#L1493)

( config: DetrConfig )

Parameters

-   **config** ([DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/modeling_detr.py#L1519)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[typing.List\[dict\]\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.detr.modeling\_detr.DetrObjectDetectionOutput](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrObjectDetectionOutput) or `tuple(torch.FloatTensor)`

The [DetrForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, DetrForObjectDetection
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
>>> model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)

>>> 
>>> target_sizes = torch.tensor([image.size[::-1]])
>>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
...     0
... ]

>>> for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
...     box = [round(i, 2) for i in box.tolist()]
...     print(
...         f"Detected {model.config.id2label[label.item()]} with confidence "
...         f"{round(score.item(), 3)} at location {box}"
...     )
Detected remote with confidence 0.998 at location [40.16, 70.81, 175.55, 117.98]
Detected remote with confidence 0.996 at location [333.24, 72.55, 368.33, 187.66]
Detected couch with confidence 0.995 at location [-0.02, 1.15, 639.73, 473.76]
Detected cat with confidence 0.999 at location [13.24, 52.05, 314.02, 470.93]
Detected cat with confidence 0.999 at location [345.4, 23.85, 640.37, 368.72]
```

## DetrForSegmentation

### class transformers.DetrForSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/modeling_detr.py#L1667)

( config: DetrConfig )

Parameters

-   **config** ([DetrConfig](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top, for tasks such as COCO panoptic.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/detr/modeling_detr.py#L1689)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.LongTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[typing.List\[dict\]\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.detr.modeling\_detr.DetrSegmentationOutput](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.models.detr.modeling_detr.DetrSegmentationOutput) or `tuple(torch.FloatTensor)`

The [DetrForSegmentation](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrForSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> import io
>>> import requests
>>> from PIL import Image
>>> import torch
>>> import numpy

>>> from transformers import AutoImageProcessor, DetrForSegmentation
>>> from transformers.image_transforms import rgb_to_id

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
>>> model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

>>> 
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> 
>>> outputs = model(**inputs)

>>> 
>>> 
>>> result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[(300, 500)])

>>> 
>>> panoptic_seg = result[0]["segmentation"]
>>> 
>>> panoptic_segments_info = result[0]["segments_info"]
```