# Semantic segmentation

Semantic segmentation assigns a label or class to each individual pixel of an image. There are several types of segmentation, and in the case of semantic segmentation, no distinction is made between unique instances of the same object. Both objects are given the same label (for example, “car” instead of “car-1” and “car-2”). Common real-world applications of semantic segmentation include training self-driving cars to identify pedestrians and important traffic information, identifying cells and abnormalities in medical imagery, and monitoring environmental changes from satellite imagery.

This guide will show you how to:

1.  Finetune [SegFormer](https://huggingface.co/docs/transformers/main/en/model_doc/segformer#segformer) on the [SceneParse150](https://huggingface.co/datasets/scene_parse_150) dataset.
2.  Use your finetuned model for inference.

The task illustrated in this tutorial is supported by the following model architectures:

[BEiT](../model_doc/beit), [Data2VecVision](../model_doc/data2vec-vision), [DPT](../model_doc/dpt), [MobileNetV2](../model_doc/mobilenet_v2), [MobileViT](../model_doc/mobilevit), [MobileViTV2](../model_doc/mobilevitv2), [SegFormer](../model_doc/segformer), [UPerNet](../model_doc/upernet)

Before you begin, make sure you have all the necessary libraries installed:

```
pip install -q datasets transformers evaluate
```

We encourage you to log in to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to log in:

```
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load SceneParse150 dataset

Start by loading a smaller subset of the SceneParse150 dataset from the 🤗 Datasets library. This’ll give you a chance to experiment and make sure everything works before spending more time training on the full dataset.

```
>>> from datasets import load_dataset

>>> ds = load_dataset("scene_parse_150", split="train[:50]")
```

Split the dataset’s `train` split into a train and test set with the [train\_test\_split](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.train_test_split) method:

```
>>> ds = ds.train_test_split(test_size=0.2)
>>> train_ds = ds["train"]
>>> test_ds = ds["test"]
```

Then take a look at an example:

```
>>> train_ds[0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>,
 'annotation': <PIL.PngImagePlugin.PngImageFile image mode=L size=512x683 at 0x7F9B0C201DD0>,
 'scene_category': 368}
```

-   `image`: a PIL image of the scene.
-   `annotation`: a PIL image of the segmentation map, which is also the model’s target.
-   `scene_category`: a category id that describes the image scene like “kitchen” or “office”. In this guide, you’ll only need `image` and `annotation`, both of which are PIL images.

You’ll also want to create a dictionary that maps a label id to a label class which will be useful when you set up the model later. Download the mappings from the Hub and create the `id2label` and `label2id` dictionaries:

```
>>> import json
>>> from huggingface_hub import cached_download, hf_hub_url

>>> repo_id = "huggingface/label-files"
>>> filename = "ade20k-id2label.json"
>>> id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
>>> id2label = {int(k): v for k, v in id2label.items()}
>>> label2id = {v: k for k, v in id2label.items()}
>>> num_labels = len(id2label)
```

## Preprocess

The next step is to load a SegFormer image processor to prepare the images and annotations for the model. Some datasets, like this one, use the zero-index as the background class. However, the background class isn’t actually included in the 150 classes, so you’ll need to set `reduce_labels=True` to subtract one from all the labels. The zero-index is replaced by `255` so it’s ignored by SegFormer’s loss function:

```
>>> from transformers import AutoImageProcessor

>>> checkpoint = "nvidia/mit-b0"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)
```

It is common to apply some data augmentations to an image dataset to make a model more robust against overfitting. In this guide, you’ll use the [`ColorJitter`](https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html) function from [torchvision](https://pytorch.org/vision/stable/index.html) to randomly change the color properties of an image, but you can also use any image library you like.

```
>>> from torchvision.transforms import ColorJitter

>>> jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
```

Now create two preprocessing functions to prepare the images and annotations for the model. These functions convert the images into `pixel_values` and annotations to `labels`. For the training set, `jitter` is applied before providing the images to the image processor. For the test set, the image processor crops and normalizes the `images`, and only crops the `labels` because no data augmentation is applied during testing.

```
>>> def train_transforms(example_batch):
...     images = [jitter(x) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [x for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

To apply the `jitter` over the entire dataset, use the 🤗 Datasets [set\_transform](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.set_transform) function. The transform is applied on the fly which is faster and consumes less disk space:

```
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```

It is common to apply some data augmentations to an image dataset to make a model more robust against overfitting. In this guide, you’ll use [`tf.image`](https://www.tensorflow.org/api_docs/python/tf/image) to randomly change the color properties of an image, but you can also use any image library you like. Define two separate transformation functions:

-   training data transformations that include image augmentation
-   validation data transformations that only transpose the images, since computer vision models in 🤗 Transformers expect channels-first layout

```
>>> import tensorflow as tf


>>> def aug_transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.image.random_brightness(image, 0.25)
...     image = tf.image.random_contrast(image, 0.5, 2.0)
...     image = tf.image.random_saturation(image, 0.75, 1.25)
...     image = tf.image.random_hue(image, 0.1)
...     image = tf.transpose(image, (2, 0, 1))
...     return image


>>> def transforms(image):
...     image = tf.keras.utils.img_to_array(image)
...     image = tf.transpose(image, (2, 0, 1))
...     return image
```

Next, create two preprocessing functions to prepare batches of images and annotations for the model. These functions apply the image transformations and use the earlier loaded `image_processor` to convert the images into `pixel_values` and annotations to `labels`. `ImageProcessor` also takes care of resizing and normalizing the images.

```
>>> def train_transforms(example_batch):
...     images = [aug_transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [transforms(x.convert("RGB")) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

To apply the preprocessing transformations over the entire dataset, use the 🤗 Datasets [set\_transform](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.set_transform) function. The transform is applied on the fly which is faster and consumes less disk space:

```
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```

## Evaluate

Including a metric during training is often helpful for evaluating your model’s performance. You can quickly load an evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load the [mean Intersection over Union](https://huggingface.co/spaces/evaluate-metric/accuracy) (IoU) metric (see the 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about how to load and compute a metric):

```
>>> import evaluate

>>> metric = evaluate.load("mean_iou")
```

Then create a function to [compute](https://huggingface.co/docs/evaluate/v0.4.0/en/package_reference/main_classes#evaluate.EvaluationModule.compute) the metrics. Your predictions need to be converted to logits first, and then reshaped to match the size of the labels before you can call [compute](https://huggingface.co/docs/evaluate/v0.4.0/en/package_reference/main_classes#evaluate.EvaluationModule.compute):

```
>>> import numpy as np
>>> import torch
>>> from torch import nn

>>> def compute_metrics(eval_pred):
...     with torch.no_grad():
...         logits, labels = eval_pred
...         logits_tensor = torch.from_numpy(logits)
...         logits_tensor = nn.functional.interpolate(
...             logits_tensor,
...             size=labels.shape[-2:],
...             mode="bilinear",
...             align_corners=False,
...         ).argmax(dim=1)

...         pred_labels = logits_tensor.detach().cpu().numpy()
...         metrics = metric.compute(
...             predictions=pred_labels,
...             references=labels,
...             num_labels=num_labels,
...             ignore_index=255,
...             reduce_labels=False,
...         )
...         for key, value in metrics.items():
...             if type(value) is np.ndarray:
...                 metrics[key] = value.tolist()
...         return metrics
```

```
>>> def compute_metrics(eval_pred):
...     logits, labels = eval_pred
...     logits = tf.transpose(logits, perm=[0, 2, 3, 1])
...     logits_resized = tf.image.resize(
...         logits,
...         size=tf.shape(labels)[1:],
...         method="bilinear",
...     )

...     pred_labels = tf.argmax(logits_resized, axis=-1)
...     metrics = metric.compute(
...         predictions=pred_labels,
...         references=labels,
...         num_labels=num_labels,
...         ignore_index=-1,
...         reduce_labels=image_processor.do_reduce_labels,
...     )

...     per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
...     per_category_iou = metrics.pop("per_category_iou").tolist()

...     metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
...     metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
...     return {"val_" + k: v for k, v in metrics.items()}
```

Your `compute_metrics` function is ready to go now, and you’ll return to it when you setup your training.

## Train

If you aren’t familiar with finetuning a model with the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer), take a look at the basic tutorial [here](../training#finetune-with-trainer)!

You’re ready to start training your model now! Load SegFormer with [AutoModelForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModelForSemanticSegmentation), and pass the model the mapping between label ids and label classes:

```
>>> from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

>>> model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
```

At this point, only three steps remain:

1.  Define your training hyperparameters in [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments). It is important you don’t remove unused columns because this’ll drop the `image` column. Without the `image` column, you can’t create `pixel_values`. Set `remove_unused_columns=False` to prevent this behavior! The only other required parameter is `output_dir` which specifies where to save your model. You’ll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) will evaluate the IoU metric and save the training checkpoint.
2.  Pass the training arguments to [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) along with the model, dataset, tokenizer, data collator, and `compute_metrics` function.
3.  Call [train()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.train) to finetune your model.

```
>>> training_args = TrainingArguments(
...     output_dir="segformer-b0-scene-parse-150",
...     learning_rate=6e-5,
...     num_train_epochs=50,
...     per_device_train_batch_size=2,
...     per_device_eval_batch_size=2,
...     save_total_limit=3,
...     evaluation_strategy="steps",
...     save_strategy="steps",
...     save_steps=20,
...     eval_steps=20,
...     logging_steps=1,
...     eval_accumulation_steps=5,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=test_ds,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Once training is completed, share your model to the Hub with the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.push_to_hub) method so everyone can use your model:

```
>>> trainer.push_to_hub()
```

If you are unfamiliar with fine-tuning a model with Keras, check out the [basic tutorial](./training#train-a-tensorflow-model-with-keras) first!

To fine-tune a model in TensorFlow, follow these steps:

1.  Define the training hyperparameters, and set up an optimizer and a learning rate schedule.
2.  Instantiate a pretrained model.
3.  Convert a 🤗 Dataset to a `tf.data.Dataset`.
4.  Compile your model.
5.  Add callbacks to calculate metrics and upload your model to 🤗 Hub
6.  Use the `fit()` method to run the training.

Start by defining the hyperparameters, optimizer and learning rate schedule:

```
>>> from transformers import create_optimizer

>>> batch_size = 2
>>> num_epochs = 50
>>> num_train_steps = len(train_ds) * num_epochs
>>> learning_rate = 6e-5
>>> weight_decay_rate = 0.01

>>> optimizer, lr_schedule = create_optimizer(
...     init_lr=learning_rate,
...     num_train_steps=num_train_steps,
...     weight_decay_rate=weight_decay_rate,
...     num_warmup_steps=0,
... )
```

Then, load SegFormer with [TFAutoModelForSemanticSegmentation](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.TFAutoModelForSemanticSegmentation) along with the label mappings, and compile it with the optimizer. Note that Transformers models all have a default task-relevant loss function, so you don’t need to specify one unless you want to:

```
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained(
...     checkpoint,
...     id2label=id2label,
...     label2id=label2id,
... )
>>> model.compile(optimizer=optimizer)  
```

Convert your datasets to the `tf.data.Dataset` format using the [to\_tf\_dataset](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset) and the [DefaultDataCollator](/docs/transformers/v4.34.0/en/main_classes/data_collator#transformers.DefaultDataCollator):

```
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator(return_tensors="tf")

>>> tf_train_dataset = train_ds.to_tf_dataset(
...     columns=["pixel_values", "label"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )

>>> tf_eval_dataset = test_ds.to_tf_dataset(
...     columns=["pixel_values", "label"],
...     shuffle=True,
...     batch_size=batch_size,
...     collate_fn=data_collator,
... )
```

To compute the accuracy from the predictions and push your model to the 🤗 Hub, use [Keras callbacks](../main_classes/keras_callbacks). Pass your `compute_metrics` function to [KerasMetricCallback](/docs/transformers/v4.34.0/en/main_classes/keras_callbacks#transformers.KerasMetricCallback), and use the [PushToHubCallback](/docs/transformers/v4.34.0/en/main_classes/keras_callbacks#transformers.PushToHubCallback) to upload the model:

```
>>> from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

>>> metric_callback = KerasMetricCallback(
...     metric_fn=compute_metrics, eval_dataset=tf_eval_dataset, batch_size=batch_size, label_cols=["labels"]
... )

>>> push_to_hub_callback = PushToHubCallback(output_dir="scene_segmentation", tokenizer=image_processor)

>>> callbacks = [metric_callback, push_to_hub_callback]
```

Finally, you are ready to train your model! Call `fit()` with your training and validation datasets, the number of epochs, and your callbacks to fine-tune the model:

```
>>> model.fit(
...     tf_train_dataset,
...     validation_data=tf_eval_dataset,
...     callbacks=callbacks,
...     epochs=num_epochs,
... )
```

Congratulations! You have fine-tuned your model and shared it on the 🤗 Hub. You can now use it for inference!

## Inference

Great, now that you’ve finetuned a model, you can use it for inference!

Load an image for inference:

```
>>> image = ds[0]["image"]
>>> image
```

![Image of bedroom](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-image.png)

The simplest way to try out your finetuned model for inference is to use it in a [pipeline()](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.pipeline). Instantiate a `pipeline` for image segmentation with your model, and pass your image to it:

```
>>> from transformers import pipeline

>>> segmenter = pipeline("image-segmentation", model="my_awesome_seg_model")
>>> segmenter(image)
[{'score': None,
  'label': 'wall',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062690>},
 {'score': None,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A50>},
 {'score': None,
  'label': 'floor',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062B50>},
 {'score': None,
  'label': 'ceiling',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062A10>},
 {'score': None,
  'label': 'bed ',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E90>},
 {'score': None,
  'label': 'windowpane',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062390>},
 {'score': None,
  'label': 'cabinet',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062550>},
 {'score': None,
  'label': 'chair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062D90>},
 {'score': None,
  'label': 'armchair',
  'mask': <PIL.Image.Image image mode=L size=640x427 at 0x7FD5B2062E10>}]
```

You can also manually replicate the results of the `pipeline` if you’d like. Process the image with an image processor and place the `pixel_values` on a GPU:

```
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
>>> encoding = image_processor(image, return_tensors="pt")
>>> pixel_values = encoding.pixel_values.to(device)
```

Pass your input to the model and return the `logits`:

```
>>> outputs = model(pixel_values=pixel_values)
>>> logits = outputs.logits.cpu()
```

Next, rescale the logits to the original image size:

```
>>> upsampled_logits = nn.functional.interpolate(
...     logits,
...     size=image.size[::-1],
...     mode="bilinear",
...     align_corners=False,
... )

>>> pred_seg = upsampled_logits.argmax(dim=1)[0]
```

Load an image processor to preprocess the image and return the input as TensorFlow tensors:

```
>>> from transformers import AutoImageProcessor

>>> image_processor = AutoImageProcessor.from_pretrained("MariaK/scene_segmentation")
>>> inputs = image_processor(image, return_tensors="tf")
```

Pass your input to the model and return the `logits`:

```
>>> from transformers import TFAutoModelForSemanticSegmentation

>>> model = TFAutoModelForSemanticSegmentation.from_pretrained("MariaK/scene_segmentation")
>>> logits = model(**inputs).logits
```

Next, rescale the logits to the original image size and apply argmax on the class dimension:

```
>>> logits = tf.transpose(logits, [0, 2, 3, 1])

>>> upsampled_logits = tf.image.resize(
...     logits,
...     
...     image.size[::-1],
... )

>>> pred_seg = tf.math.argmax(upsampled_logits, axis=-1)[0]
```

To visualize the results, load the [dataset color palette](https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51) as `ade_palette()` that maps each class to their RGB values. Then you can combine and plot your image and the predicted segmentation map:

```
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
>>> palette = np.array(ade_palette())
>>> for label, color in enumerate(palette):
...     color_seg[pred_seg == label, :] = color
>>> color_seg = color_seg[..., ::-1]  

>>> img = np.array(image) * 0.5 + color_seg * 0.5  
>>> img = img.astype(np.uint8)

>>> plt.figure(figsize=(15, 10))
>>> plt.imshow(img)
>>> plt.show()
```

![Image of bedroom overlaid with segmentation map](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-preds.png)