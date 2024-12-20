# VisionTextDualEncoder

## Overview

The [VisionTextDualEncoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel) can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model as the vision encoder (_e.g._ [ViT](vit), [BEiT](beit), [DeiT](deit)) and any pretrained text autoencoding model as the text encoder (_e.g._ [RoBERTa](roberta), [BERT](bert)). Two projection layers are added on top of both the vision and text encoder to project the output embeddings to a shared latent space. The projection layers are randomly initialized so the model should be fine-tuned on a downstream task. This model can be used to align the vision-text embeddings using CLIP like contrastive image-text training and then can be used for zero-shot vision tasks such image-classification or retrieval.

In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvement on new zero-shot vision tasks such as image classification or retrieval.

## VisionTextDualEncoderConfig

### class transformers.VisionTextDualEncoderConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_text_dual_encoder/configuration_vision_text_dual_encoder.py#L27)

( projection\_dim = 512logit\_scale\_init\_value = 2.6592\*\*kwargs )

Parameters

-   **text\_config** (`dict`) — Dictionary of configuration options that defines text model config.
-   **vision\_config** (`dict`) — Dictionary of configuration options that defines vison model config.
-   **projection\_dim** (`int`, _optional_, defaults to 512) — Dimentionality of text and vision projection layers.
-   **logit\_scale\_init\_value** (`float`, _optional_, defaults to 2.6592) — The inital value of the _logit\_scale_ paramter. Default is used as per the original CLIP implementation.
-   **kwargs** (_optional_) — Dictionary of keyword arguments.

[VisionTextDualEncoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig) is the configuration class to store the configuration of a [VisionTextDualEncoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel). It is used to instantiate [VisionTextDualEncoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel) model according to the specified arguments, defining the text model and vision model configs.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel

>>> 
>>> config_vision = ViTConfig()
>>> config_text = BertConfig()

>>> config = VisionTextDualEncoderConfig.from_vision_text_configs(config_vision, config_text, projection_dim=512)

>>> 
>>> model = VisionTextDualEncoderModel(config=config)

>>> 
>>> config_vision = model.config.vision_config
>>> config_text = model.config.text_config

>>> 
>>> model.save_pretrained("vit-bert")

>>> 
>>> vision_text_config = VisionTextDualEncoderConfig.from_pretrained("vit-bert")
>>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert", config=vision_text_config)
```

## VisionTextDualEncoderProcessor

### class transformers.VisionTextDualEncoderProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_text_dual_encoder/processing_vision_text_dual_encoder.py#L25)

( image\_processor = Nonetokenizer = None\*\*kwargs )

Parameters

-   **image\_processor** ([AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor)) — The image processor is a required input.
-   **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) — The tokenizer is a required input.

Constructs a VisionTextDualEncoder processor which wraps an image processor and a tokenizer into a single processor.

[VisionTextDualEncoderProcessor](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderProcessor) offers all the functionalities of [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) and [AutoTokenizer](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoTokenizer). See the `__call__()` and [decode()](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderProcessor.decode) for more information.

This method forwards all its arguments to VisionTextDualEncoderTokenizer’s [batch\_decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.batch_decode). Please refer to the docstring of this method for more information.

This method forwards all its arguments to VisionTextDualEncoderTokenizer’s [decode()](/docs/transformers/v4.34.0/en/model_doc/speecht5#transformers.SpeechT5Tokenizer.decode). Please refer to the docstring of this method for more information.

## VisionTextDualEncoderModel

### class transformers.VisionTextDualEncoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py#L162)

( config: typing.Optional\[transformers.models.vision\_text\_dual\_encoder.configuration\_vision\_text\_dual\_encoder.VisionTextDualEncoderConfig\] = Nonevision\_model: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = Nonetext\_model: typing.Optional\[transformers.modeling\_utils.PreTrainedModel\] = None )

Parameters

-   **config** ([VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded via the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) method. The projection layers are automatically added to the model and should be fine-tuned on a downstream task, like contrastive image-text modeling.

In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment on new zero-shot vision tasks such as image classification or retrieval.

After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_text_dual_encoder/modeling_vision_text_dual_encoder.py#L293)

( input\_ids: typing.Optional\[torch.LongTensor\] = Nonepixel\_values: typing.Optional\[torch.FloatTensor\] = Noneattention\_mask: typing.Optional\[torch.Tensor\] = Noneposition\_ids: typing.Optional\[torch.LongTensor\] = Nonereturn\_loss: typing.Optional\[bool\] = Nonetoken\_type\_ids: typing.Optional\[torch.LongTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clip.modeling_clip.CLIPOutput` or `tuple(torch.FloatTensor)`

The [VisionTextDualEncoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import (
...     VisionTextDualEncoderModel,
...     VisionTextDualEncoderProcessor,
...     AutoImageProcessor,
...     AutoTokenizer,
... )

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
>>> processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
>>> model = VisionTextDualEncoderModel.from_vision_text_pretrained(
...     "google/vit-base-patch16-224", "bert-base-uncased"
... )

>>> 
>>> urls = [
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
... ]
>>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="pt", padding=True
... )
>>> outputs = model(
...     input_ids=inputs.input_ids,
...     attention_mask=inputs.attention_mask,
...     pixel_values=inputs.pixel_values,
...     return_loss=True,
... )
>>> loss, logits_per_image = outputs.loss, outputs.logits_per_image  

>>> 
>>> model.save_pretrained("vit-bert")
>>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert")

>>> 
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = logits_per_image.softmax(dim=1)  
```

## FlaxVisionTextDualEncoderModel

### class transformers.FlaxVisionTextDualEncoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_text_dual_encoder/modeling_flax_vision_text_dual_encoder.py#L219)

( config: VisionTextDualEncoderConfiginput\_shape: typing.Optional\[typing.Tuple\] = Noneseed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([VisionTextDualEncoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.VisionTextDualEncoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded via the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) method. The projection layers are automatically added to the model and should be fine-tuned on a downstream task, like contrastive image-text modeling.

In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment on new zero-shot vision tasks such as image classification or retrieval.

After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

#### \_\_call\_\_

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_text_dual_encoder/modeling_flax_vision_text_dual_encoder.py#L269)

( input\_idspixel\_valuesattention\_mask = Noneposition\_ids = Nonetoken\_type\_ids = Noneparams: dict = Nonedropout\_rng: PRNGKey = Nonetrain: bool = Falseoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.clip.modeling_flax_clip.FlaxCLIPOutput` or `tuple(torch.FloatTensor)`

The [FlaxVisionTextDualEncoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.FlaxVisionTextDualEncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> import jax
>>> from transformers import (
...     FlaxVisionTextDualEncoderModel,
...     VisionTextDualEncoderProcessor,
...     AutoImageProcessor,
...     AutoTokenizer,
... )

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> image_processor = AutoImageProcesor.from_pretrained("google/vit-base-patch16-224")
>>> processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
>>> model = FlaxVisionTextDualEncoderModel.from_vision_text_pretrained(
...     "google/vit-base-patch16-224", "bert-base-uncased"
... )

>>> 
>>> urls = [
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
... ]
>>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="np", padding=True
... )
>>> outputs = model(
...     input_ids=inputs.input_ids,
...     attention_mask=inputs.attention_mask,
...     pixel_values=inputs.pixel_values,
... )
>>> logits_per_image = outputs.logits_per_image  

>>> 
>>> model.save_pretrained("vit-bert")
>>> model = FlaxVisionTextDualEncoderModel.from_pretrained("vit-bert")

>>> 
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = jax.nn.softmax(logits_per_image, axis=1)  
```

## TFVisionTextDualEncoderModel

### class transformers.TFVisionTextDualEncoderModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_text_dual_encoder/modeling_tf_vision_text_dual_encoder.py#L176)

( \*args\*\*kwargs )

Parameters

-   **config** ([VisionEncoderDecoderConfig](/docs/transformers/v4.34.0/en/model_doc/vision-encoder-decoder#transformers.VisionEncoderDecoderConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded via the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.FlaxAutoModelForVision2Seq.from_pretrained) method. The projection layers are automatically added to the model and should be fine-tuned on a downstream task, like contrastive image-text modeling.

In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvment on new zero-shot vision tasks such as image classification or retrieval.

After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other models (see the examples for more information).

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular Keras Model and refer to the TF documentation for all matter related to general usage and behavior.

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vision_text_dual_encoder/modeling_tf_vision_text_dual_encoder.py#L341)

( input\_ids: tf.Tensor | None = Nonepixel\_values: tf.Tensor | None = Noneattention\_mask: tf.Tensor | None = Noneposition\_ids: tf.Tensor | None = Nonereturn\_loss: Optional\[bool\] = Nonetoken\_type\_ids: tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.models.clip.modeling_tf_clip.TFCLIPOutput` or `tuple(tf.Tensor)`

The [TFVisionTextDualEncoderModel](/docs/transformers/v4.34.0/en/model_doc/vision-text-dual-encoder#transformers.TFVisionTextDualEncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from PIL import Image
>>> import requests
>>> from transformers import (
...     TFVisionTextDualEncoderModel,
...     VisionTextDualEncoderProcessor,
...     AutoImageProcessor,
...     AutoTokenizer,
... )

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
>>> processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)
>>> model = TFVisionTextDualEncoderModel.from_vision_text_pretrained(
...     "google/vit-base-patch16-224", "bert-base-uncased"
... )

>>> 
>>> urls = [
...     "http://images.cocodataset.org/val2017/000000039769.jpg",
...     "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
... ]
>>> images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
>>> inputs = processor(
...     text=["a photo of a cat", "a photo of a dog"], images=images, return_tensors="np", padding=True
... )
>>> outputs = model(
...     input_ids=inputs.input_ids,
...     attention_mask=inputs.attention_mask,
...     pixel_values=inputs.pixel_values,
...     return_loss=True,
... )
>>> loss, logits_per_image = outputs.loss, outputs.logits_per_image  

>>> 
>>> model.save_pretrained("vit-bert")
>>> model = TFVisionTextDualEncoderModel.from_pretrained("vit-bert")

>>> 
>>> outputs = model(**inputs)
>>> logits_per_image = outputs.logits_per_image  
>>> probs = tf.nn.softmax(logits_per_image, axis=1)  
```