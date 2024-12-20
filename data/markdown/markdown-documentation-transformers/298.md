# ViTMAE

## Overview

The ViTMAE model was proposed in [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377v2) by Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick. The paper shows that, by pre-training a Vision Transformer (ViT) to reconstruct pixel values for masked patches, one can get results after fine-tuning that outperform supervised pre-training.

The abstract from the paper is the following:

_This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3x or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pre-training and shows promising scaling behavior._

Tips:

-   MAE (masked auto encoding) is a method for self-supervised pre-training of Vision Transformers (ViTs). The pre-training objective is relatively simple: by masking a large portion (75%) of the image patches, the model must reconstruct raw pixel values. One can use [ViTMAEForPreTraining](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining) for this purpose.
-   After pre-training, one “throws away” the decoder used to reconstruct pixels, and one uses the encoder for fine-tuning/linear probing. This means that after fine-tuning, one can directly plug in the weights into a [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification).
-   One can use [ViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor) to prepare images for the model. See the code examples for more info.
-   Note that the encoder of MAE is only used to encode the visual patches. The encoded patches are then concatenated with mask tokens, which the decoder (which also consists of Transformer blocks) takes as input. Each mask token is a shared, learned vector that indicates the presence of a missing patch to be predicted. Fixed sin/cos position embeddings are added both to the input of the encoder and the decoder.
-   For a visual understanding of how MAEs work you can check out this [post](https://keras.io/examples/vision/masked_image_modeling/).

![drawing](https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png) MAE architecture. Taken from the [original paper.](https://arxiv.org/abs/2111.06377)

This model was contributed by [nielsr](https://huggingface.co/nielsr). TensorFlow version of the model was contributed by [sayakpaul](https://github.com/sayakpaul) and [ariG23498](https://github.com/ariG23498) (equal contribution). The original code can be found [here](https://github.com/facebookresearch/mae).

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViTMAE.

-   [ViTMAEForPreTraining](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining), allowing you to pre-train the model from scratch/further pre-train the model on custom data.
-   A notebook that illustrates how to visualize reconstructed pixel values with [ViTMAEForPreTraining](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining) can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTMAE/ViT_MAE_visualization_demo.ipynb).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## ViTMAEConfig

### class transformers.ViTMAEConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_mae/configuration_vit_mae.py#L29)

( hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-12image\_size = 224patch\_size = 16num\_channels = 3qkv\_bias = Truedecoder\_num\_attention\_heads = 16decoder\_hidden\_size = 512decoder\_num\_hidden\_layers = 8decoder\_intermediate\_size = 2048mask\_ratio = 0.75norm\_pix\_loss = False\*\*kwargs )

This is the configuration class to store the configuration of a [ViTMAEModel](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEModel). It is used to instantiate an ViT MAE model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ViT [facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ViTMAEConfig, ViTMAEModel

>>> 
>>> configuration = ViTMAEConfig()

>>> 
>>> model = ViTMAEModel(configuration)

>>> 
>>> configuration = model.config
```

## ViTMAEModel

### class transformers.ViTMAEModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_mae/modeling_vit_mae.py#L637)

( config )

Parameters

-   **config** ([ViTMAEConfig](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare ViTMAE Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_mae/modeling_vit_mae.py#L661)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonenoise: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.vit_mae.modeling_vit_mae.ViTMAEModelOutput` or `tuple(torch.FloatTensor)`

The [ViTMAEModel](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, ViTMAEModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
>>> model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## ViTMAEForPreTraining

### class transformers.ViTMAEForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_mae/modeling_vit_mae.py#L849)

( config )

Parameters

-   **config** ([ViTMAEConfig](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The ViTMAE Model transformer with the decoder on top for self-supervised pre-training.

Note that we provide a script to pre-train this model on custom data in our [examples directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_mae/modeling_vit_mae.py#L962)

( pixel\_values: typing.Optional\[torch.FloatTensor\] = Nonenoise: typing.Optional\[torch.FloatTensor\] = Nonehead\_mask: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.vit_mae.modeling_vit_mae.ViTMAEForPreTrainingOutput` or `tuple(torch.FloatTensor)`

The [ViTMAEForPreTraining](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, ViTMAEForPreTraining
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
>>> model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> loss = outputs.loss
>>> mask = outputs.mask
>>> ids_restore = outputs.ids_restore
```

## TFViTMAEModel

### class transformers.TFViTMAEModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L778)

( \*args\*\*kwargs )

Parameters

-   **config** ([ViTMAEConfig](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare ViTMAE Model transformer outputting raw hidden-states without any specific head on top. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L787)

( pixel\_values: TFModelInputType | None = Nonenoise: tf.Tensor = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.models.vit_mae.modeling_tf_vit_mae.TFViTMAEModelOutput` or `tuple(tf.Tensor)`

The [TFViTMAEModel](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.TFViTMAEModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TFViTMAEModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
>>> model = TFViTMAEModel.from_pretrained("facebook/vit-mae-base")

>>> inputs = image_processor(images=image, return_tensors="tf")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## TFViTMAEForPreTraining

### class transformers.TFViTMAEForPreTraining

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L939)

( \*args\*\*kwargs )

Parameters

-   **config** ([ViTMAEConfig](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.ViTMAEConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The ViTMAE Model transformer with the decoder on top for self-supervised pre-training. This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and behavior.

TensorFlow models and layers in `transformers` accept two formats as input:

-   having all inputs as keyword arguments (like PyTorch models), or
-   having all inputs as a list, tuple or dict in the first positional argument.

The reason the second format is supported is that Keras methods prefer this format when passing inputs to models and layers. Because of this support, when using methods like `model.fit()` things should “just work” for you - just pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first positional argument:

-   a single Tensor with `pixel_values` only and nothing else: `model(pixel_values)`
-   a list of varying length with one or several input Tensors IN THE ORDER given in the docstring: `model([pixel_values, attention_mask])` or `model([pixel_values, attention_mask, token_type_ids])`
-   a dictionary with one or several input Tensors associated to the input names given in the docstring: `model({"pixel_values": pixel_values, "token_type_ids": token_type_ids})`

Note that when creating models and layers with [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don’t need to worry about any of this, as you can just pass inputs like you would to any other Python function!

#### call

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit_mae/modeling_tf_vit_mae.py#L1063)

( pixel\_values: TFModelInputType | None = Nonenoise: tf.Tensor = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → `transformers.models.vit_mae.modeling_tf_vit_mae.TFViTMAEForPreTrainingOutput` or `tuple(tf.Tensor)`

The [TFViTMAEForPreTraining](/docs/transformers/v4.34.0/en/model_doc/vit_mae#transformers.TFViTMAEForPreTraining) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TFViTMAEForPreTraining
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
>>> model = TFViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

>>> inputs = image_processor(images=image, return_tensors="pt")
>>> outputs = model(**inputs)
>>> loss = outputs.loss
>>> mask = outputs.mask
>>> ids_restore = outputs.ids_restore
```