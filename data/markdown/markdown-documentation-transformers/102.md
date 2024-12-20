# BEiT

## Overview

The BEiT model was proposed in [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) by Hangbo Bao, Li Dong and Furu Wei. Inspired by BERT, BEiT is the first paper that makes self-supervised pre-training of Vision Transformers (ViTs) outperform supervised pre-training. Rather than pre-training the model to predict the class of an image (as done in the [original ViT paper](https://arxiv.org/abs/2010.11929)), BEiT models are pre-trained to predict visual tokens from the codebook of OpenAI’s [DALL-E model](https://arxiv.org/abs/2102.12092) given masked patches.

The abstract from the paper is the following:

_We introduce a self-supervised vision representation model BEiT, which stands for Bidirectional Encoder representation from Image Transformers. Following BERT developed in the natural language processing area, we propose a masked image modeling task to pretrain vision Transformers. Specifically, each image has two views in our pre-training, i.e, image patches (such as 16x16 pixels), and visual tokens (i.e., discrete tokens). We first “tokenize” the original image into visual tokens. Then we randomly mask some image patches and fed them into the backbone Transformer. The pre-training objective is to recover the original visual tokens based on the corrupted image patches. After pre-training BEiT, we directly fine-tune the model parameters on downstream tasks by appending task layers upon the pretrained encoder. Experimental results on image classification and semantic segmentation show that our model achieves competitive results with previous pre-training methods. For example, base-size BEiT achieves 83.2% top-1 accuracy on ImageNet-1K, significantly outperforming from-scratch DeiT training (81.8%) with the same setup. Moreover, large-size BEiT obtains 86.3% only using ImageNet-1K, even outperforming ViT-L with supervised pre-training on ImageNet-22K (85.2%)._

Tips:

-   BEiT models are regular Vision Transformers, but pre-trained in a self-supervised way rather than supervised. They outperform both the [original model (ViT)](vit) as well as [Data-efficient Image Transformers (DeiT)](deit) when fine-tuned on ImageNet-1K and CIFAR-100. You can check out demo notebooks regarding inference as well as fine-tuning on custom data [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer) (you can just replace [ViTFeatureExtractor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTFeatureExtractor) by [BeitImageProcessor](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitImageProcessor) and [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification) by [BeitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForImageClassification)).
-   There’s also a demo notebook available which showcases how to combine DALL-E’s image tokenizer with BEiT for performing masked image modeling. You can find it [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/BEiT).
-   As the BEiT models expect each image to be of the same size (resolution), one can use [BeitImageProcessor](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitImageProcessor) to resize (or rescale) and normalize images for the model.
-   Both the patch resolution and image resolution used during pre-training or fine-tuning are reflected in the name of each checkpoint. For example, `microsoft/beit-base-patch16-224` refers to a base-sized architecture with patch resolution of 16x16 and fine-tuning resolution of 224x224. All checkpoints can be found on the [hub](https://huggingface.co/models?search=microsoft/beit).
-   The available checkpoints are either (1) pre-trained on [ImageNet-22k](http://www.image-net.org/) (a collection of 14 million images and 22k classes) only, (2) also fine-tuned on ImageNet-22k or (3) also fine-tuned on [ImageNet-1k](http://www.image-net.org/challenges/LSVRC/2012/) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes).
-   BEiT uses relative position embeddings, inspired by the T5 model. During pre-training, the authors shared the relative position bias among the several self-attention layers. During fine-tuning, each layer’s relative position bias is initialized with the shared relative position bias obtained after pre-training. Note that, if one wants to pre-train a model from scratch, one needs to either set the `use_relative_position_bias` or the `use_relative_position_bias` attribute of [BeitConfig](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitConfig) to `True` in order to add position embeddings.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/beit_architecture.jpg) BEiT pre-training. Taken from the [original paper.](https://arxiv.org/abs/2106.08254)

This model was contributed by [nielsr](https://huggingface.co/nielsr). The JAX/FLAX version of this model was contributed by [kamalkraj](https://huggingface.co/kamalkraj). The original code can be found [here](https://github.com/microsoft/unilm/tree/master/beit).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with BEiT.

-   [BeitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

**Semantic segmentation**

-   [Semantic segmentation task guide](../tasks/semantic_segmentation)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## BEiT specific outputs

### class transformers.models.beit.modeling\_beit.BeitModelOutputWithPooling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_beit.py#L68)

( last\_hidden\_state: FloatTensor = Nonepooler\_output: FloatTensor = Nonehidden\_states: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = Noneattentions: typing.Optional\[typing.Tuple\[torch.FloatTensor\]\] = None )

Class for outputs of [BeitModel](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitModel).

### class transformers.models.beit.modeling\_flax\_beit.FlaxBeitModelOutputWithPooling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_flax_beit.py#L45)

( last\_hidden\_state: Array = Nonepooler\_output: Array = Nonehidden\_states: typing.Optional\[typing.Tuple\[jax.Array\]\] = Noneattentions: typing.Optional\[typing.Tuple\[jax.Array\]\] = None )

Class for outputs of [FlaxBeitModel](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.FlaxBeitModel).

## BeitConfig

### class transformers.BeitConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/configuration_beit.py#L36)

( vocab\_size = 8192hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-12image\_size = 224patch\_size = 16num\_channels = 3use\_mask\_token = Falseuse\_absolute\_position\_embeddings = Falseuse\_relative\_position\_bias = Falseuse\_shared\_relative\_position\_bias = Falselayer\_scale\_init\_value = 0.1drop\_path\_rate = 0.1use\_mean\_pooling = Trueout\_indices = \[3, 5, 7, 11\]pool\_scales = \[1, 2, 3, 6\]use\_auxiliary\_head = Trueauxiliary\_loss\_weight = 0.4auxiliary\_channels = 256auxiliary\_num\_convs = 1auxiliary\_concat\_input = Falsesemantic\_loss\_ignore\_index = 255\*\*kwargs )

This is the configuration class to store the configuration of a [BeitModel](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitModel). It is used to instantiate an BEiT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the BEiT [microsoft/beit-base-patch16-224-pt22k](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k) architecture.

Example:

```
>>> from transformers import BeitConfig, BeitModel

>>> 
>>> configuration = BeitConfig()

>>> 
>>> model = BeitModel(configuration)

>>> 
>>> configuration = model.config
```

## BeitFeatureExtractor

( imagessegmentation\_maps = None\*\*kwargs )

( outputstarget\_sizes: typing.List\[typing.Tuple\] = None ) → semantic\_segmentation

Parameters

-   **outputs** ([BeitForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForSemanticSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple]` of length `batch_size`, _optional_) — List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

`List[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [BeitForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## BeitImageProcessor

### class transformers.BeitImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/image_processing_beit.py#L49)

( do\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BICUBIC: 3>do\_center\_crop: bool = Truecrop\_size: typing.Dict\[str, int\] = Nonerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_rescale: bool = Truedo\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_reduce\_labels: bool = False\*\*kwargs )

Constructs a BEiT image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/image_processing_beit.py#L312)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]segmentation\_maps: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\], NoneType\] = Nonedo\_resize: bool = Nonesize: typing.Dict\[str, int\] = Noneresample: Resampling = Nonedo\_center\_crop: bool = Nonecrop\_size: typing.Dict\[str, int\] = Nonedo\_rescale: bool = Nonerescale\_factor: float = Nonedo\_normalize: bool = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonedo\_reduce\_labels: typing.Optional\[bool\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: ChannelDimension = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

#### post\_process\_semantic\_segmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/image_processing_beit.py#L464)

( outputstarget\_sizes: typing.List\[typing.Tuple\] = None ) → semantic\_segmentation

Parameters

-   **outputs** ([BeitForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForSemanticSegmentation)) — Raw outputs of the model.
-   **target\_sizes** (`List[Tuple]` of length `batch_size`, _optional_) — List of tuples corresponding to the requested final size (height, width) of each prediction. If unset, predictions will not be resized.

Returns

semantic\_segmentation

`List[torch.Tensor]` of length `batch_size`, where each item is a semantic segmentation map of shape (height, width) corresponding to the target\_sizes entry (if `target_sizes` is specified). Each entry of each `torch.Tensor` correspond to a semantic class id.

Converts the output of [BeitForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForSemanticSegmentation) into semantic segmentation maps. Only supports PyTorch.

## BeitModel

### class transformers.BeitModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_beit.py#L618)

( config: BeitConfigadd\_pooling\_layer: bool = True )

Parameters

-   **config** ([BeitConfig](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Beit Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_beit.py#L645)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.models.beit.modeling\_beit.BeitModelOutputWithPooling](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.models.beit.modeling_beit.BeitModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [BeitModel](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, BeitModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
>>> model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 197, 768]
```

## BeitForMaskedImageModeling

### class transformers.BeitForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_beit.py#L733)

( config: BeitConfig )

Parameters

-   **config** ([BeitConfig](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Beit Model transformer with a ‘language’ modeling head on top. BEiT does masked image modeling by predicting visual tokens of a Vector-Quantize Variational Autoencoder (VQ-VAE), whereas other vision models like ViT and DeiT predict RGB pixel values. As a result, this class is incompatible with [AutoModelForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModelForMaskedImageModeling), so you will need to use [BeitForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForMaskedImageModeling) directly if you wish to do masked image modeling with BEiT. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_beit.py#L747)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.MaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

The [BeitForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForMaskedImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, BeitForMaskedImageModeling
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
>>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
>>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
>>> 
>>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss, logits = outputs.loss, outputs.logits
>>> list(logits.shape)
[1, 196, 8192]
```

## BeitForImageClassification

### class transformers.BeitForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_beit.py#L833)

( config: BeitConfig )

Parameters

-   **config** ([BeitConfig](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final hidden states of the patch tokens) e.g. for ImageNet.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_beit.py#L846)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

The [BeitForImageClassification](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, BeitForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
>>> model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
tabby, tabby cat
```

## BeitForSemanticSegmentation

### class transformers.BeitForSemanticSegmentation

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_beit.py#L1156)

( config: BeitConfig )

Parameters

-   **config** ([BeitConfig](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Beit Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_beit.py#L1202)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.SemanticSegmenterOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.SemanticSegmenterOutput) or `tuple(torch.FloatTensor)`

The [BeitForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitForSemanticSegmentation) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, BeitForSemanticSegmentation
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")
>>> model = BeitForSemanticSegmentation.from_pretrained("microsoft/beit-base-finetuned-ade-640-640")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> 
>>> logits = outputs.logits
```

## FlaxBeitModel

### class transformers.FlaxBeitModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_flax_beit.py#L744)

( config: BeitConfiginput\_shape = Noneseed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BeitConfig](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare Beit Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

The `FlaxBeitPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, FlaxBeitModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
>>> model = FlaxBeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")

>>> inputs = image_processor(images=image, return_tensors="np")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## FlaxBeitForMaskedImageModeling

### class transformers.FlaxBeitForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_flax_beit.py#L828)

( config: BeitConfiginput\_shape = Noneseed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BeitConfig](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Beit Model transformer with a ‘language’ modeling head on top (to predict visual tokens).

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_flax_beit.py#L631)

( pixel\_valuesbool\_masked\_pos = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_flax\_outputs.FlaxMaskedLMOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_flax_outputs.FlaxMaskedLMOutput) or `tuple(torch.FloatTensor)`

The `FlaxBeitPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

bool\_masked\_pos (`numpy.ndarray` of shape `(batch_size, num_patches)`): Boolean masked positions. Indicates which patches are masked (1) and which aren’t (0).

Examples:

```
>>> from transformers import AutoImageProcessor, BeitForMaskedImageModeling
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
>>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

>>> inputs = image_processor(images=image, return_tensors="np")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
```

## FlaxBeitForImageClassification

### class transformers.FlaxBeitForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/beit/modeling_flax_beit.py#L915)

( config: BeitConfiginput\_shape = Noneseed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([BeitConfig](/docs/transformers/v4.34.0/en/model_doc/beit#transformers.BeitConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final hidden states of the patch tokens) e.g. for ImageNet.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

The `FlaxBeitPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, FlaxBeitForImageClassification
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
>>> model = FlaxBeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")

>>> inputs = image_processor(images=image, return_tensors="np")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
>>> 
>>> predicted_class_idx = logits.argmax(-1).item()
>>> print("Predicted class:", model.config.id2label[predicted_class_idx])
```