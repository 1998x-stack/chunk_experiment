# Fine-tune a pretrained model

## Train a TensorFlow model with Keras

You can also train 🤗 Transformers models in TensorFlow with the Keras API!

### Loading data for Keras

When you want to train a 🤗 Transformers model with the Keras API, you need to convert your dataset to a format that Keras understands. If your dataset is small, you can just convert the whole thing to NumPy arrays and pass it to Keras. Let’s try that first before we do anything more complicated.

First, load a dataset. We’ll use the CoLA dataset from the [GLUE benchmark](https://huggingface.co/datasets/glue), since it’s a simple binary text classification task, and just take the training split for now.

```
from datasets import load_dataset

dataset = load_dataset("glue", "cola")
dataset = dataset["train"]  
```

Next, load a tokenizer and tokenize the data as NumPy arrays. Note that the labels are already a list of 0 and 1s, so we can just convert that directly to a NumPy array without tokenization!

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_data = tokenizer(dataset["sentence"], return_tensors="np", padding=True)

tokenized_data = dict(tokenized_data)

labels = np.array(dataset["label"])  
```

Finally, load, [`compile`](https://keras.io/api/models/model_training_apis/#compile-method), and [`fit`](https://keras.io/api/models/model_training_apis/#fit-method) the model. Note that Transformers models all have a default task-relevant loss function, so you don’t need to specify one unless you want to:

```
from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam


model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased")

model.compile(optimizer=Adam(3e-5))  

model.fit(tokenized_data, labels)
```

You don’t have to pass a loss argument to your models when you `compile()` them! Hugging Face models automatically choose a loss that is appropriate for their task and model architecture if this argument is left blank. You can always override this by specifying a loss yourself if you want to!

This approach works great for smaller datasets, but for larger datasets, you might find it starts to become a problem. Why? Because the tokenized array and labels would have to be fully loaded into memory, and because NumPy doesn’t handle “jagged” arrays, so every tokenized sample would have to be padded to the length of the longest sample in the whole dataset. That’s going to make your array even bigger, and all those padding tokens will slow down training too!

### Loading data as a tf.data.Dataset

If you want to avoid slowing down training, you can load your data as a `tf.data.Dataset` instead. Although you can write your own `tf.data` pipeline if you want, we have two convenience methods for doing this:

-   [prepare\_tf\_dataset()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset): This is the method we recommend in most cases. Because it is a method on your model, it can inspect the model to automatically figure out which columns are usable as model inputs, and discard the others to make a simpler, more performant dataset.
-   [to\_tf\_dataset](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.to_tf_dataset): This method is more low-level, and is useful when you want to exactly control how your dataset is created, by specifying exactly which `columns` and `label_cols` to include.

Before you can use [prepare\_tf\_dataset()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel.prepare_tf_dataset), you will need to add the tokenizer outputs to your dataset as columns, as shown in the following code sample:

```
def tokenize_dataset(data):
    
    return tokenizer(data["text"])


dataset = dataset.map(tokenize_dataset)
```

Remember that Hugging Face datasets are stored on disk by default, so this will not inflate your memory usage! Once the columns have been added, you can stream batches from the dataset and add padding to each batch, which greatly reduces the number of padding tokens compared to padding the entire dataset.

```
>>> tf_dataset = model.prepare_tf_dataset(dataset["train"], batch_size=16, shuffle=True, tokenizer=tokenizer)
```

Note that in the code sample above, you need to pass the tokenizer to `prepare_tf_dataset` so it can correctly pad batches as they’re loaded. If all the samples in your dataset are the same length and no padding is necessary, you can skip this argument. If you need to do something more complex than just padding samples (e.g. corrupting tokens for masked language modelling), you can use the `collate_fn` argument instead to pass a function that will be called to transform the list of samples into a batch and apply any preprocessing you want. See our [examples](https://github.com/huggingface/transformers/tree/main/examples) or [notebooks](https://huggingface.co/docs/transformers/notebooks) to see this approach in action.

Once you’ve created a `tf.data.Dataset`, you can compile and fit the model as before:

```
model.compile(optimizer=Adam(3e-5))  

model.fit(tf_dataset)
```