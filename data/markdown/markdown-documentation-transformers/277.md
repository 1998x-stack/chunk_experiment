# Table Transformer

## Overview

The Table Transformer model was proposed in [PubTables-1M: Towards comprehensive table extraction from unstructured documents](https://arxiv.org/abs/2110.00061) by Brandon Smock, Rohith Pesala, Robin Abraham. The authors introduce a new dataset, PubTables-1M, to benchmark progress in table extraction from unstructured documents, as well as table structure recognition and functional analysis. The authors train 2 [DETR](detr) models, one for table detection and one for table structure recognition, dubbed Table Transformers.

The abstract from the paper is the following:

_Recently, significant progress has been made applying machine learning to the problem of table structure inference and extraction from unstructured documents. However, one of the greatest challenges remains the creation of datasets with complete, unambiguous ground truth at scale. To address this, we develop a new, more comprehensive dataset for table extraction, called PubTables-1M. PubTables-1M contains nearly one million tables from scientific articles, supports multiple input modalities, and contains detailed header and location information for table structures, making it useful for a wide variety of modeling approaches. It also addresses a significant source of ground truth inconsistency observed in prior datasets called oversegmentation, using a novel canonicalization procedure. We demonstrate that these improvements lead to a significant increase in training performance and a more reliable estimate of model performance at evaluation for table structure recognition. Further, we show that transformer-based object detection models trained on PubTables-1M produce excellent results for all three tasks of detection, structure recognition, and functional analysis without the need for any special customization for these tasks._

Tips:

-   The authors released 2 models, one for [table detection](https://huggingface.co/microsoft/table-transformer-detection) in documents, one for [table structure recognition](https://huggingface.co/microsoft/table-transformer-structure-recognition) (the task of recognizing the individual rows, columns etc. in a table).
-   One can use the [AutoImageProcessor](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoImageProcessor) API to prepare images and optional targets for the model. This will load a [DetrImageProcessor](/docs/transformers/v4.34.0/en/model_doc/detr#transformers.DetrImageProcessor) behind the scenes.

![drawing](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/table_transformer_architecture.jpeg) Table detection and table structure recognition clarified. Taken from the [original paper](https://arxiv.org/abs/2110.00061).

This model was contributed by [nielsr](https://huggingface.co/nielsr). The original code can be found [here](https://github.com/microsoft/table-transformer).

## Resources

-   A demo notebook for the Table Transformer can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/Table%20Transformer).
-   It turns out padding of images is quite important for detection. An interesting Github thread with replies from the authors can be found [here](https://github.com/microsoft/table-transformer/issues/68).

## TableTransformerConfig

### class transformers.TableTransformerConfig

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/table_transformer/configuration_table_transformer.py#L36)

( use\_timm\_backbone = Truebackbone\_config = Nonenum\_channels = 3num\_queries = 100encoder\_layers = 6encoder\_ffn\_dim = 2048encoder\_attention\_heads = 8decoder\_layers = 6decoder\_ffn\_dim = 2048decoder\_attention\_heads = 8encoder\_layerdrop = 0.0decoder\_layerdrop = 0.0is\_encoder\_decoder = Trueactivation\_function = 'relu'd\_model = 256dropout = 0.1attention\_dropout = 0.0activation\_dropout = 0.0init\_std = 0.02init\_xavier\_std = 1.0auxiliary\_loss = Falseposition\_embedding\_type = 'sine'backbone = 'resnet50'use\_pretrained\_backbone = Truedilation = Falseclass\_cost = 1bbox\_cost = 5giou\_cost = 2mask\_loss\_coefficient = 1dice\_loss\_coefficient = 1bbox\_loss\_coefficient = 5giou\_loss\_coefficient = 2eos\_coefficient = 0.1\*\*kwargs )

This is the configuration class to store the configuration of a [TableTransformerModel](/docs/transformers/v4.34.0/en/model_doc/table-transformer#transformers.TableTransformerModel). It is used to instantiate a Table Transformer model according to the specified arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of the Table Transformer [microsoft/table-transformer-detection](https://huggingface.co/microsoft/table-transformer-detection) architecture.

Configuration objects inherit from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) and can be used to control the model outputs. Read the documentation from [PretrainedConfig](/docs/transformers/v4.34.0/en/main_classes/configuration#transformers.PretrainedConfig) for more information.

Examples:

```
>>> from transformers import TableTransformerModel, TableTransformerConfig

>>> 
>>> configuration = TableTransformerConfig()

>>> 
>>> model = TableTransformerModel(configuration)

>>> 
>>> configuration = model.config
```

## TableTransformerModel

### class transformers.TableTransformerModel

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/table_transformer/modeling_table_transformer.py#L1217)

( config: TableTransformerConfig )

Parameters

-   **config** ([TableTransformerConfig](/docs/transformers/v4.34.0/en/model_doc/table-transformer#transformers.TableTransformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/table_transformer/modeling_table_transformer.py#L1252)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.table_transformer.modeling_table_transformer.TableTransformerModelOutput` or `tuple(torch.FloatTensor)`

The [TableTransformerModel](/docs/transformers/v4.34.0/en/model_doc/table-transformer#transformers.TableTransformerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from transformers import AutoImageProcessor, TableTransformerModel
>>> from huggingface_hub import hf_hub_download
>>> from PIL import Image

>>> file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
>>> image = Image.open(file_path).convert("RGB")

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
>>> model = TableTransformerModel.from_pretrained("microsoft/table-transformer-detection")

>>> 
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> 
>>> outputs = model(**inputs)

>>> 
>>> 
>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 15, 256]
```

## TableTransformerForObjectDetection

### class transformers.TableTransformerForObjectDetection

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/table_transformer/modeling_table_transformer.py#L1386)

( config: TableTransformerConfig )

Parameters

-   **config** ([TableTransformerConfig](/docs/transformers/v4.34.0/en/model_doc/table-transformer#transformers.TableTransformerConfig)) — Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Table Transformer Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks such as COCO detection.

This model inherits from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

#### forward

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/table_transformer/modeling_table_transformer.py#L1413)

( pixel\_values: FloatTensorpixel\_mask: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_attention\_mask: typing.Optional\[torch.FloatTensor\] = Noneencoder\_outputs: typing.Optional\[torch.FloatTensor\] = Noneinputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonedecoder\_inputs\_embeds: typing.Optional\[torch.FloatTensor\] = Nonelabels: typing.Optional\[typing.List\[typing.Dict\]\] = Noneoutput\_attentions: typing.Optional\[bool\] = Noneoutput\_hidden\_states: typing.Optional\[bool\] = Nonereturn\_dict: typing.Optional\[bool\] = None ) → `transformers.models.table_transformer.modeling_table_transformer.TableTransformerObjectDetectionOutput` or `tuple(torch.FloatTensor)`

The [TableTransformerForObjectDetection](/docs/transformers/v4.34.0/en/model_doc/table-transformer#transformers.TableTransformerForObjectDetection) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module` instance afterwards instead of this since the former takes care of running the pre and post processing steps while the latter silently ignores them.

Examples:

```
>>> from huggingface_hub import hf_hub_download
>>> from transformers import AutoImageProcessor, TableTransformerForObjectDetection
>>> import torch
>>> from PIL import Image

>>> file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
>>> image = Image.open(file_path).convert("RGB")

>>> image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
>>> model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

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
Detected table with confidence 1.0 at location [202.1, 210.59, 1119.22, 385.09]
```