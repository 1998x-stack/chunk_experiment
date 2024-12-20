# Vision Transformer (ViT)

## Overview

The Vision Transformer (ViT) model was proposed in [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. It’s the first paper that successfully trains a Transformer encoder on ImageNet, attaining very good results compared to familiar convolutional architectures.

The abstract from the paper is the following:

_While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train._

Tips:

-   Demo notebooks regarding inference as well as fine-tuning ViT on custom data can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer).
-   To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches, which are then linearly embedded. A \[CLS\] token is added to serve as representation of an entire image, which can be used for classification. The authors also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.
-   As the Vision Transformer expects each image to be of the same size (resolution), one can use [ViTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTImageProcessor) to resize (or rescale) and normalize images for the model.
-   Both the patch resolution and image resolution used during pre-training or fine-tuning are reflected in the name of each checkpoint. For example, `google/vit-base-patch16-224` refers to a base-sized architecture with patch resolution of 16x16 and fine-tuning resolution of 224x224. All checkpoints can be found on the [hub](https://huggingface.co/models?search=vit).
-   The available checkpoints are either (1) pre-trained on [ImageNet-21k](http://www.image-net.org/) (a collection of 14 million images and 21k classes) only, or (2) also fine-tuned on [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) (also referred to as ILSVRC 2012, a collection of 1.3 million images and 1,000 classes).
-   The Vision Transformer was pre-trained using a resolution of 224x224. During fine-tuning, it is often beneficial to use a higher resolution than pre-training [(Touvron et al., 2019)](https://arxiv.org/abs/1906.06423), [(Kolesnikov et al., 2020)](https://arxiv.org/abs/1912.11370). In order to fine-tune at higher resolution, the authors perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image.
-   The best results are obtained with supervised pre-training, which is not the case in NLP. The authors also performed an experiment with a self-supervised pre-training objective, namely masked patched prediction (inspired by masked language modeling). With this approach, the smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a significant improvement of 2% to training from scratch, but still 4% behind supervised pre-training.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vit_architecture.jpg) ViT architecture. Taken from the [original paper.](https://arxiv.org/abs/2010.11929)

Following the original Vision Transformer, some follow-up works have been made:

-   [DeiT](deit) (Data-efficient Image Transformers) by Facebook AI. DeiT models are distilled vision transformers. The authors of DeiT also released more efficiently trained ViT models, which you can directly plug into [ViTModel](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTModel) or [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification). There are 4 variants available (in 3 different sizes): _facebook/deit-tiny-patch16-224_, _facebook/deit-small-patch16-224_, _facebook/deit-base-patch16-224_ and _facebook/deit-base-patch16-384_. Note that one should use [DeiTImageProcessor](/docs/transformers/v4.34.0/en/model_doc/deit#transformers.DeiTImageProcessor) in order to prepare images for the model.
    
-   [BEiT](beit) (BERT pre-training of Image Transformers) by Microsoft Research. BEiT models outperform supervised pre-trained vision transformers using a self-supervised method inspired by BERT (masked image modeling) and based on a VQ-VAE.
    
-   DINO (a method for self-supervised training of Vision Transformers) by Facebook AI. Vision Transformers trained using the DINO method show very interesting properties not seen with convolutional models. They are capable of segmenting objects, without having ever been trained to do so. DINO checkpoints can be found on the [hub](https://huggingface.co/models?other=dino).
    
-   [MAE](vit_mae) (Masked Autoencoders) by Facebook AI. By pre-training Vision Transformers to reconstruct pixel values for a high portion (75%) of masked patches (using an asymmetric encoder-decoder architecture), the authors show that this simple method outperforms supervised pre-training after fine-tuning.
    

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code (written in JAX) can be found [here](https://github.com/google-research/vision_transformer).

Note that we converted the weights from Ross Wightman’s [timm library](https://github.com/rwightman/pytorch-image-models), who already converted the weights from JAX to PyTorch. Credits go to him!

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViT.

-   [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification) and [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb).
-   A blog on fine-tuning [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification) on a custom dataset can be found [here](https://huggingface.co/blog/fine-tune-vit).
-   More demo notebooks to fine-tune [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification) can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/VisionTransformer).
-   [Image classification task guide](../tasks/image_classification)

Besides that:

-   [ViTForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForMaskedImageModeling) is supported by this [example script](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViT. If you’re interested in submitting a resource to be included here, please feel free to open a Pull Request and we’ll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

`ViTForImageClassification` is supported by:

-   A blog post on how to [Fine-Tune ViT for Image Classification with Hugging Face Transformers](https://huggingface.co/blog/fine-tune-vit)
-   A blog post on [Image Classification with Hugging Face Transformers and `Keras`](https://www.philschmid.de/image-classification-huggingface-transformers-keras)
-   A notebook on [Fine-tuning for Image Classification with Hugging Face Transformers](https://github.com/huggingface/notebooks/blob/main/examples/image_classification.ipynb)
-   A notebook on how to [Fine-tune the Vision Transformer on CIFAR-10 with the Hugging Face Trainer](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb)
-   A notebook on how to [Fine-tune the Vision Transformer on CIFAR-10 with PyTorch Lightning](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_PyTorch_Lightning.ipynb)

⚗️ Optimization

-   A blog post on how to [Accelerate Vision Transformer (ViT) with Quantization using Optimum](https://www.philschmid.de/optimizing-vision-transformer)

⚡️ Inference

-   A notebook on [Quick demo: Vision Transformer (ViT) by Google Brain](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Quick_demo_of_HuggingFace_version_of_Vision_Transformer_inference.ipynb)

🚀 Deploy

-   A blog post on [Deploying Tensorflow Vision Models in Hugging Face with TF Serving](https://huggingface.co/blog/tf-serving-vision)
-   A blog post on [Deploying Hugging Face ViT on Vertex AI](https://huggingface.co/blog/deploy-vertex-ai)
-   A blog post on [Deploying Hugging Face ViT on Kubernetes with TF Serving](https://huggingface.co/blog/deploy-tfserving-kubernetes)

## ViTConfig

### class transformers.ViTConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/configuration_vit.py#L35)

( hidden\_size = 768num\_hidden\_layers = 12num\_attention\_heads = 12intermediate\_size = 3072hidden\_act = 'gelu'hidden\_dropout\_prob = 0.0attention\_probs\_dropout\_prob = 0.0initializer\_range = 0.02layer\_norm\_eps = 1e-12image\_size = 224patch\_size = 16num\_channels = 3qkv\_bias = Trueencoder\_stride = 16\*\*kwargs )

This is the configuration class to store the configuration of a [ViTModel](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTModel). It is used to instantiate an ViT model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the ViT [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Example:

```
>>> from transformers import ViTConfig, ViTModel

>>> 
>>> configuration = ViTConfig()

>>> 
>>> model = ViTModel(configuration)

>>> 
>>> configuration = model.config
```

## ViTFeatureExtractor

Preprocess an image or a batch of images.

## ViTImageProcessor

### class transformers.ViTImageProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/image_processing_vit.py#L41)

( do\_resize: bool = Truesize: typing.Union\[typing.Dict\[str, int\], NoneType\] = Noneresample: Resampling = <Resampling.BILINEAR: 2>do\_rescale: bool = Truerescale\_factor: typing.Union\[int, float\] = 0.00392156862745098do\_normalize: bool = Trueimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = None\*\*kwargs )

Constructs a ViT image processor.

#### preprocess

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/image_processing_vit.py#L146)

( images: typing.Union\[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), typing.List\[ForwardRef('PIL.Image.Image')\], typing.List\[numpy.ndarray\], typing.List\[ForwardRef('torch.Tensor')\]\]do\_resize: typing.Optional\[bool\] = Nonesize: typing.Dict\[str, int\] = Noneresample: Resampling = Nonedo\_rescale: typing.Optional\[bool\] = Nonerescale\_factor: typing.Optional\[float\] = Nonedo\_normalize: typing.Optional\[bool\] = Noneimage\_mean: typing.Union\[float, typing.List\[float\], NoneType\] = Noneimage\_std: typing.Union\[float, typing.List\[float\], NoneType\] = Nonereturn\_tensors: typing.Union\[str, transformers.utils.generic.TensorType, NoneType\] = Nonedata\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension\] = <ChannelDimension.FIRST: 'channels\_first'>input\_data\_format: typing.Union\[str, transformers.image\_utils.ChannelDimension, NoneType\] = None\*\*kwargs )

Preprocess an image or batch of images.

## ViTModel

### class transformers.ViTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_vit.py#L515)

( config: ViTConfigadd\_pooling\_layer: bool = Trueuse\_mask\_token: bool = False )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare ViT Model transformer outputting raw hidden-states without any specific head on top. This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_vit.py#L540)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneinterpolate\_pos\_encoding: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.BaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

The [ViTModel](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, ViTModel
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 197, 768]
```

## ViTForMaskedImageModeling

### class transformers.ViTForMaskedImageModeling

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_vit.py#L637)

( config: ViTConfig )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

Note that we provide a script to pre-train this model on custom data in our [examples directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_vit.py#L655)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonebool\_masked\_pos: typing.Optional\[torch.BoolTensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneinterpolate\_pos\_encoding: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.modeling_outputs.MaskedImageModelingOutput` or `tuple(torch.FloatTensor)`

The [ViTForMaskedImageModeling](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForMaskedImageModeling) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
>>> import torch
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

>>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
>>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
>>> 
>>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

>>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
>>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
>>> list(reconstructed_pixel_values.shape)
[1, 3, 224, 224]
```

## ViTForImageClassification

### class transformers.ViTForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_vit.py#L766)

( config: ViTConfig )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

Note that it’s possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained position embeddings to the higher resolution.

This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_vit.py#L779)

( pixel\_values: typing.Optional\[torch.Tensor\] = Nonehead\_mask: typing.Optional\[torch.Tensor\] = Nonelabels: typing.Optional\[torch.Tensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Noneinterpolate\_pos\_encoding: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → [transformers.modeling\_outputs.ImageClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_outputs.ImageClassifierOutput) or `tuple(torch.FloatTensor)`

The [ViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, ViTForImageClassification
>>> import torch
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
>>> model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

>>> inputs = image_processor(image, return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> 
>>> predicted_label = logits.argmax(-1).item()
>>> print(model.config.id2label[predicted_label])
Egyptian cat
```

## TFViTModel

### class transformers.TFViTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_tf_vit.py#L631)

( \*args\*\*kwargs )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

The bare ViT Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_tf_vit.py#L637)

( pixel\_values: TFModelInputType | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Noneinterpolate\_pos\_encoding: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonetraining: bool = False ) → [transformers.modeling\_tf\_outputs.TFBaseModelOutputWithPooling](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFBaseModelOutputWithPooling) or `tuple(tf.Tensor)`

The [TFViTModel](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.TFViTModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFViTModel
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

>>> inputs = image_processor(image, return_tensors="tf")
>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 197, 768]
```

## TFViTForImageClassification

### class transformers.TFViTForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_tf_vit.py#L704)

( \*args\*\*kwargs )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.from_pretrained) method to load the model weights.

ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

Note that it’s possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained position embeddings to the higher resolution.

This model inherits from [TFPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

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

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_tf_vit.py#L718)

( pixel\_values: TFModelInputType | None = Nonehead\_mask: np.ndarray | tf.Tensor | None = Noneoutput\_attentions: Optional\[bool\] = Noneoutput\_hidden\_states: Optional\[bool\] = Noneinterpolate\_pos\_encoding: Optional\[bool\] = Nonereturn\_dict: Optional\[bool\] = Nonelabels: np.ndarray | tf.Tensor | None = Nonetraining: Optional\[bool\] = False ) → [transformers.modeling\_tf\_outputs.TFSequenceClassifierOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.modeling_tf_outputs.TFSequenceClassifierOutput) or `tuple(tf.Tensor)`

The [TFViTForImageClassification](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.TFViTForImageClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, TFViTForImageClassification
>>> import tensorflow as tf
>>> from datasets import load_dataset

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
>>> model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

>>> inputs = image_processor(image, return_tensors="tf")
>>> logits = model(**inputs).logits

>>> 
>>> predicted_label = int(tf.math.argmax(logits, axis=-1))
>>> print(model.config.id2label[predicted_label])
Egyptian cat
```

## FlaxVitModel

### class transformers.FlaxViTModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_flax_vit.py#L555)

( config: ViTConfiginput\_shape = Noneseed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

The bare ViT Model transformer outputting raw hidden-states without any specific head on top.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

The `FlaxViTPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, FlaxViTModel
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> model = FlaxViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

>>> inputs = image_processor(images=image, return_tensors="np")
>>> outputs = model(**inputs)
>>> last_hidden_states = outputs.last_hidden_state
```

## FlaxViTForImageClassification

### class transformers.FlaxViTForImageClassification

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/vit/modeling_flax_vit.py#L638)

( config: ViTConfiginput\_shape = Noneseed: int = 0dtype: dtype = <class 'jax.numpy.float32'>\_do\_init: bool = True\*\*kwargs )

Parameters

-   **config** ([ViTConfig](/docs/transformers/v4.34.0/en/model_doc/vit#transformers.ViTConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.from_pretrained) method to load the model weights.
-   **dtype** (`jax.numpy.dtype`, _optional_, defaults to `jax.numpy.float32`) — The data type of the computation. Can be one of `jax.numpy.float32`, `jax.numpy.float16` (on GPUs) and `jax.numpy.bfloat16` (on TPUs).
    
    This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If specified all the computation will be performed with the given `dtype`.
    
    **Note that this only specifies the dtype of the computation and does not influence the dtype of model parameters.**
    
    If you wish to change the dtype of the model parameters, see [to\_fp16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_fp16) and [to\_bf16()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel.to_bf16).
    

ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the \[CLS\] token) e.g. for ImageNet.

This model inherits from [FlaxPreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading, saving and converting weights from PyTorch models)

This model is also a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module) subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to general usage and behavior.

Finally, this model supports inherent JAX features such as:

-   [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
-   [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
-   [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
-   [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)

The `FlaxViTPreTrainedModel` forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Example:

```
>>> from transformers import AutoImageProcessor, FlaxViTForImageClassification
>>> from PIL import Image
>>> import jax
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
>>> model = FlaxViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

>>> inputs = image_processor(images=image, return_tensors="np")
>>> outputs = model(**inputs)
>>> logits = outputs.logits

>>> 
>>> predicted_class_idx = jax.numpy.argmax(logits, axis=-1)
>>> print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
```