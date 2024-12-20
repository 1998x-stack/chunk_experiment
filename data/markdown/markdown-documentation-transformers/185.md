# ImageGPT

## Overview

The ImageGPT model was proposed in [Generative Pretraining from Pixels](https://openai.com/blog/image-gpt) by Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever. ImageGPT (iGPT) is a GPT-2-like model trained to predict the next pixel value, allowing for both unconditional and conditional image generation.

The abstract from the paper is the following:

_Inspired by progress in unsupervised representation learning for natural language, we examine whether similar models can learn useful representations for images. We train a sequence Transformer to auto-regressively predict pixels, without incorporating knowledge of the 2D input structure. Despite training on low-resolution ImageNet without labels, we find that a GPT-2 scale model learns strong image representations as measured by linear probing, fine-tuning, and low-data classification. On CIFAR-10, we achieve 96.3% accuracy with a linear probe, outperforming a supervised Wide ResNet, and 99.0% accuracy with full fine-tuning, matching the top supervised pre-trained models. We are also competitive with self-supervised benchmarks on ImageNet when substituting pixels for a VQVAE encoding, achieving 69.0% top-1 accuracy on a linear probe of our features._

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/imagegpt_architecture.png) Summary of the approach. Taken from the \[original paper\](https://cdn.openai.com/papers/Generative\_Pretraining\_from\_Pixels\_V2.pdf).

This model was contributed by [nielsr](https://huggingface.co/nielsr), based on [this issue](https://github.com/openai/image-gpt/issues/7). The original code can be found [here](https://github.com/openai/image-gpt).

Tips:

-   ImageGPT is almost exactly the same as [GPT-2](gpt2), with the exception that a different activation function is used (namely “quick gelu”), and the layer normalization layers don’t mean center the inputs. ImageGPT also doesn’t have tied input- and output embeddings.
-   As the time- and memory requirements of the attention mechanism of Transformers scales quadratically in the sequence length, the authors pre-trained ImageGPT on smaller input resolutions, such as 32x32 and 64x64. However, feeding a sequence of 32x32x3=3072 tokens from 0..255 into a Transformer is still prohibitively large. Therefore, the authors applied k-means clustering to the (R,G,B) pixel values with k=512. This way, we only have a 32\*32 = 1024-long sequence, but now of integers in the range 0..511. So we are shrinking the sequence length at the cost of a bigger embedding matrix. In other words, the vocabulary size of ImageGPT is 512, + 1 for a special “start of sentence” (SOS) token, used at the beginning of every sequence. One can use [ImageGPTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTImageProcessor) to prepare images for the model.
-   Despite being pre-trained entirely unsupervised (i.e. without the use of any labels), ImageGPT produces fairly performant image features useful for downstream tasks, such as image classification. The authors showed that the features in the middle of the network are the most performant, and can be used as-is to train a linear model (such as a sklearn logistic regression model for example). This is also referred to as “linear probing”. Features can be easily obtained by first forwarding the image through the model, then specifying `output_hidden_states=True`, and then average-pool the hidden states at whatever layer you like.
-   Alternatively, one can further fine-tune the entire model on a downstream dataset, similar to BERT. For this, you can use [ImageGPTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification).
-   ImageGPT comes in different sizes: there’s ImageGPT-small, ImageGPT-medium and ImageGPT-large. The authors did also train an XL variant, which they didn’t release. The differences in size are summarized in the following table:

| **Model variant** | **Depths** | **Hidden sizes** | **Decoder hidden size** | **Params (M)** | **ImageNet-1k Top 1** |
| --- | --- | --- | --- | --- | --- |
| MiT-b0 | \[2, 2, 2, 2\] | \[32, 64, 160, 256\] | 256 | 3.7 | 70.5 |
| MiT-b1 | \[2, 2, 2, 2\] | \[64, 128, 320, 512\] | 256 | 14.0 | 78.7 |
| MiT-b2 | \[3, 4, 6, 3\] | \[64, 128, 320, 512\] | 768 | 25.4 | 81.6 |
| MiT-b3 | \[3, 4, 18, 3\] | \[64, 128, 320, 512\] | 768 | 45.2 | 83.1 |
| MiT-b4 | \[3, 8, 27, 3\] | \[64, 128, 320, 512\] | 768 | 62.6 | 83.6 |
| MiT-b5 | \[3, 6, 40, 3\] | \[64, 128, 320, 512\] | 768 | 82.0 | 83.8 |

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ImageGPT.

-   Demo notebooks for ImageGPT can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ImageGPT).
-   [ImageGPTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   See also: [Image classification task guide](../tasks/image_classification)

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ImageGPTConfig

### class transformers.ImageGPTConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/imagegpt/configuration_imagegpt.py#L37)

( vocab\_size = 513n\_positions = 1024n\_embd = 512n\_layer = 24n\_head = 8n\_inner = Noneactivation\_function = 'quick\_gelu'resid\_pdrop = 0.1embd\_pdrop = 0.1attn\_pdrop = 0.1layer\_norm\_epsilon = 1e-05initializer\_range = 0.02scale\_attn\_weights = Trueuse\_cache = Truetie\_word\_embeddings = Falsescale\_attn\_by\_inverse\_layer\_idx = Falsereorder\_and\_upcast\_attn = False\*\*kwargs )

This is the configuration class to store the configuration of a [ImageGPTModel](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTModel) or a `TFImageGPTModel`. It is used to instantiate a GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ImageGPT [openai/imagegpt-small](https://huggingface.co/openai/imagegpt-small) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ImageGPTConfig, ImageGPTModel

>>> 
>>> configuration = ImageGPTConfig()

>>> 
>>> model = ImageGPTModel(configuration)

>>> 
>>> configuration = model.config
```

## ImageGPTFeatureExtractor

Preprocess an image or a batch of images.

## ImageGPTImageProcessor

### class transformers.ImageGPTImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/imagegpt/image_processing_imagegpt.py#L58)

( clusters: typing.Union\[typing.List\[typing.List\[int\]\], numpy.ndarray, NoneType\] = Nonedo\_resize: bool = Truesize: typing.Dict\[str, int\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_normalize: bool = Truedo\_color\_quantize: bool = True\*\*kwargs )

Parameters

-   **clusters** (`np.ndarray` or `List[List[int]]`, _optional_) — The color clusters to use, of shape `(n_clusters, 3)` when color quantizing. Can be overriden by `clusters` in `preprocess`.
-   **do\_resize** (`bool`, _optional_, defaults to `True`) — Whether to resize the image’s dimensions to `(size["height"], size["width"])`. Can be overridden by `do_resize` in `preprocess`.
-   **size** (`Dict[str, int]` _optional_, defaults to `{"height" -- 256, "width": 256}`): Size of the image after resizing. Can be overridden by `size` in `preprocess`.
-   **resample** (`PILImageResampling`, _optional_, defaults to `PILImageResampling.BICUBIC`) — Resampling filter to use if resizing the image. Can be overridden by `resample` in `preprocess`.
-   **do\_normalize** (`bool`, _optional_, defaults to `True`) — Whether to normalize the image pixel value to between \[-1, 1\]. Can be overridden by `do_normalize` in `preprocess`.
-   **do\_color\_quantize** (`bool`, _optional_, defaults to `True`) — Whether to color quantize the image. Can be overridden by `do_color_quantize` in `preprocess`.

Constructs a ImageGPT image processor. This image processor can be used to resize images to a smaller resolution (such as 32x32 or 64x64), normalize them and finally color quantize them to obtain sequences of “pixel values” (color clusters).

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/imagegpt/image_processing_imagegpt.py#L175)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: bool = Nonesize: typing.Dict\[str, int\] = Noneresample: Resampling = Nonedo\_normalize: bool = Nonedo\_color\_quantize: typing.Optional\[bool\] = Noneclusters: typing.Union\[typing.List\[typing.List\[int\]\], numpy.ndarray, NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## ImageGPTModel

### class transformers.ImageGPTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/imagegpt/modeling_imagegpt.py#L616)

( config: ImageGPTConfig )

Parameters

-   **config** ([ImageGPTConfig](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare ImageGPT Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/imagegpt/modeling_imagegpt.py#L649)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs: typing.Any ) → [transformers.modeling\_outputs.BaseModelOutputWithPastAndCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions) or `tuple(torch.FloatTensor)`

The [ImageGPTModel](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, ImageGPTModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
>>> model = ImageGPTModel.from_pretrained("openai/imagegpt-small")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## ImageGPTForCausalImageModeling

### class transformers.ImageGPTForCausalImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/imagegpt/modeling_imagegpt.py#L895)

( config: ImageGPTConfig )

Parameters

-   **config** ([ImageGPTConfig](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/imagegpt/modeling_imagegpt.py#L943)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Noneencoder\_hidden\_states: typing.Optional\[torch.Tensor\] = Noneencoder\_attention\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs: typing.Any ) → [transformers.modeling\_outputs.CausalLMOutputWithCrossAttentions](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithCrossAttentions) or `tuple(torch.FloatTensor)`

The [ImageGPTForCausalImageModeling](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTForCausalImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, ImageGPTForCausalImageModeling
>>> import torch
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
>>> model = ImageGPTForCausalImageModeling.from_pretrained("openai/imagegpt-small")
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> model.to(device)
>>> 
>>> batch_size = 4
>>> context = torch.full((batch_size, 1), model.config.vocab_size - 1)  
>>> context = context.to(device)
>>> output = model.generate(
...     input_ids=context, max_length=model.config.n_positions + 1, temperature=1.0, do_sample=True, top_k=40
... )

>>> clusters = image_processor.clusters
>>> height = image_processor.size["height"]
>>> width = image_processor.size["width"]

>>> samples = output[:, 1:].cpu().detach().numpy()
>>> samples_img = [
...     np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [height, width, 3]).astype(np.uint8) for s in samples
... ]  
>>> f, axes = plt.subplots(1, batch_size, dpi=300)

>>> for img, ax in zip(samples_img, axes):
...     ax.axis("off")
...     ax.imshow(img)
```

## ImageGPTForImageClassification

### class transformers.ImageGPTForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/imagegpt/modeling_imagegpt.py#L1086)

( config: ImageGPTConfig )

Parameters

-   **config** ([ImageGPTConfig](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ImageGPT Model transformer with an image classification head on top (linear layer). [ImageGPTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification) average-pools the hidden states in order to do the classification.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/imagegpt/modeling_imagegpt.py#L1096)

( input\_ids: typing.Optional\[torch.Tensor\] = Nonepast\_key\_values: typing.Optional\[typing.Tuple\[typing.Tuple\[torch.Tensor\]\]\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Nonetoken\_type\_ids: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneinputs\_embeds: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneuse\_cache: typing.Optional\[bool\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None\*\*kwargs: typing.Any ) → `transformers.modeling_outputs.SequenceClassifierOutputWithPast` or `tuple(torch.FloatTensor)`

The [ImageGPTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/imagegpt#transformers.ImageGPTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, ImageGPTForImageClassification
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
>>> model = ImageGPTForImageClassification.from_pretrained("openai/imagegpt-small")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> logits = outputs.logits
```