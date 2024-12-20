# Processors

Processors can mean two different things in the Transformers library:

-   the objects that pre-process inputs for multi-modal models such as [Wav2Vec2](../model_doc/wav2vec2) (speech and text) or [CLIP](../model_doc/clip) (text and vision)
-   deprecated objects that were used in older versions of the library to preprocess data for GLUE or SQUAD.

## Multi-modal processors

Any multi-modal model will require an object to encode or decode the data that groups several modalities (among text, vision and audio). This is handled by objects called processors, which group together two or more processing objects such as tokenizers (for the text modality), image processors (for vision) and feature extractors (for audio).

Those processors inherit from the following base class that implements the saving and loading functionality:

### class transformers.ProcessorMixin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/processing_utils.py#L42)

( \*args \*\*kwargs )

This is a mixin used to provide saving/loading functionality for all processor classes.

#### from\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/processing_utils.py#L167)

( pretrained\_model\_name\_or\_path: typing.Union\[str, os.PathLike\] cache\_dir: typing.Union\[str, os.PathLike, NoneType\] = None force\_download: bool = False local\_files\_only: bool = False token: typing.Union\[bool, str, NoneType\] = None revision: str = 'main' \*\*kwargs )

Parameters

-   **pretrained\_model\_name\_or\_path** (`str` or `os.PathLike`) — This can be either:
    
    -   a string, the _model id_ of a pretrained feature\_extractor hosted inside a model repo on huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
    -   a path to a _directory_ containing a feature extractor file saved using the [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) method, e.g., `./my_model_directory/`.
    -   a path or url to a saved feature extractor JSON _file_, e.g., `./my_model_directory/preprocessor_config.json`. \*\*kwargs — Additional keyword arguments passed along to both [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained) and `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`.
    

Instantiate a processor associated with a pretrained model.

This class method is simply calling the feature extractor [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.from_pretrained), image processor [ImageProcessingMixin](/docs/transformers/v4.34.0/en/main_classes/image_processor#transformers.ImageProcessingMixin) and the tokenizer `~tokenization_utils_base.PreTrainedTokenizer.from_pretrained` methods. Please refer to the docstrings of the methods above for more information.

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/hub.py#L786)

( repo\_id: str use\_temp\_dir: typing.Optional\[bool\] = None commit\_message: typing.Optional\[str\] = None private: typing.Optional\[bool\] = None token: typing.Union\[bool, str, NoneType\] = None max\_shard\_size: typing.Union\[int, str, NoneType\] = '10GB' create\_pr: bool = False safe\_serialization: bool = False revision: str = None \*\*deprecated\_kwargs )

Parameters

-   **repo\_id** (`str`) — The name of the repository you want to push your processor to. It should contain your organization name when pushing to a given organization.
-   **use\_temp\_dir** (`bool`, _optional_) — Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub. Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
-   **commit\_message** (`str`, _optional_) — Message to commit while pushing. Will default to `"Upload processor"`.
-   **private** (`bool`, _optional_) — Whether or not the repository created should be private.
-   **token** (`bool` or `str`, _optional_) — The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`). Will default to `True` if `repo_url` is not specified.
-   **max\_shard\_size** (`int` or `str`, _optional_, defaults to `"10GB"`) — Only applicable for models. The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
-   **create\_pr** (`bool`, _optional_, defaults to `False`) — Whether or not to create a PR with the uploaded files or directly commit.
-   **safe\_serialization** (`bool`, _optional_, defaults to `False`) — Whether or not to convert the model weights in safetensors format for safer serialization.
-   **revision** (`str`, _optional_) — Branch to push the uploaded files to.

Upload the processor files to the 🤗 Model Hub.

Examples:

```
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("bert-base-cased")


processor.push_to_hub("my-finetuned-bert")


processor.push_to_hub("huggingface/my-finetuned-bert")
```

#### register\_for\_auto\_class

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/processing_utils.py#L229)

( auto\_class = 'AutoProcessor' )

Parameters

-   **auto\_class** (`str` or `type`, _optional_, defaults to `"AutoProcessor"`) — The auto class to register this new feature extractor with.

Register this class with a given auto class. This should only be used for custom feature extractors as the ones in the library are already mapped with `AutoProcessor`.

This API is experimental and may have some slight breaking changes in the next releases.

#### save\_pretrained

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/processing_utils.py#L93)

( save\_directory push\_to\_hub: bool = False \*\*kwargs )

Parameters

-   **save\_directory** (`str` or `os.PathLike`) — Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will be created if it does not exist).
-   **push\_to\_hub** (`bool`, _optional_, defaults to `False`) — Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the repository you want to push to with `repo_id` (will default to the name of `save_directory` in your namespace).
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional key word arguments passed along to the [push\_to\_hub()](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.ProcessorMixin.push_to_hub) method.

Saves the attributes of this processor (feature extractor, tokenizer…) in the specified directory so that it can be reloaded using the [from\_pretrained()](/docs/transformers/v4.34.0/en/model_doc/nougat#transformers.NougatProcessor.from_pretrained) method.

This class method is simply calling [save\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin.save_pretrained) and [save\_pretrained()](/docs/transformers/v4.34.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.save_pretrained). Please refer to the docstrings of the methods above for more information.

## Deprecated processors

All processors follow the same architecture which is that of the [DataProcessor](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.DataProcessor). The processor returns a list of [InputExample](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.InputExample). These [InputExample](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.InputExample) can be converted to [InputFeatures](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.InputFeatures) in order to be fed to the model.

Base class for data converters for sequence classification data sets.

#### get\_example\_from\_tensor\_dict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/processors/utils.py#L83)

( tensor\_dict )

Gets an example from a dict with tensorflow tensors.

Gets the list of labels for this data set.

Some tensorflow\_datasets datasets are not formatted the same way the GLUE datasets are. This method converts examples to the correct format.

### class transformers.InputExample

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/processors/utils.py#L30)

( guid: str text\_a: str text\_b: typing.Optional\[str\] = None label: typing.Optional\[str\] = None )

A single training/test example for simple sequence classification.

Serializes this instance to a JSON string.

### class transformers.InputFeatures

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/processors/utils.py#L55)

( input\_ids: typing.List\[int\] attention\_mask: typing.Optional\[typing.List\[int\]\] = None token\_type\_ids: typing.Optional\[typing.List\[int\]\] = None label: typing.Union\[int, float, NoneType\] = None )

A single set of features of data. Property names are the same names as the corresponding inputs to a model.

Serializes this instance to a JSON string.

## GLUE

[General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) is a benchmark that evaluates the performance of models across a diverse set of existing NLU tasks. It was released together with the paper [GLUE: A multi-task benchmark and analysis platform for natural language understanding](https://openreview.net/pdf?id=rJ4km2R5t7)

This library hosts a total of 10 processors for the following tasks: MRPC, MNLI, MNLI (mismatched), CoLA, SST2, STSB, QQP, QNLI, RTE and WNLI.

Those processors are:

-   `~data.processors.utils.MrpcProcessor`
-   `~data.processors.utils.MnliProcessor`
-   `~data.processors.utils.MnliMismatchedProcessor`
-   `~data.processors.utils.Sst2Processor`
-   `~data.processors.utils.StsbProcessor`
-   `~data.processors.utils.QqpProcessor`
-   `~data.processors.utils.QnliProcessor`
-   `~data.processors.utils.RteProcessor`
-   `~data.processors.utils.WnliProcessor`

Additionally, the following method can be used to load values from a data file and convert them to a list of [InputExample](/docs/transformers/v4.34.0/en/main_classes/processors#transformers.InputExample).

#### transformers.glue\_convert\_examples\_to\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/processors/glue.py#L41)

( examples: typing.Union\[typing.List\[transformers.data.processors.utils.InputExample\], ForwardRef('tf.data.Dataset')\] tokenizer: PreTrainedTokenizer max\_length: typing.Optional\[int\] = None task = None label\_list = None output\_mode = None )

Loads a data file into a list of `InputFeatures`

## XNLI

[The Cross-Lingual NLI Corpus (XNLI)](https://www.nyu.edu/projects/bowman/xnli/) is a benchmark that evaluates the quality of cross-lingual text representations. XNLI is crowd-sourced dataset based on [_MultiNLI_](http://www.nyu.edu/projects/bowman/multinli/): pairs of text are labeled with textual entailment annotations for 15 different languages (including both high-resource language such as English and low-resource languages such as Swahili).

It was released together with the paper [XNLI: Evaluating Cross-lingual Sentence Representations](https://arxiv.org/abs/1809.05053)

This library hosts the processor to load the XNLI data:

-   `~data.processors.utils.XnliProcessor`

Please note that since the gold labels are available on the test set, evaluation is performed on the test set.

An example using these processors is given in the [run\_xnli.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/text-classification/run_xnli.py) script.

## SQuAD

[The Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer//) is a benchmark that evaluates the performance of models on question answering. Two versions are available, v1.1 and v2.0. The first version (v1.1) was released together with the paper [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250). The second version (v2.0) was released alongside the paper [Know What You Don’t Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822).

This library hosts a processor for each of the two versions:

### Processors

Those processors are:

-   `~data.processors.utils.SquadV1Processor`
-   `~data.processors.utils.SquadV2Processor`

They both inherit from the abstract class `~data.processors.utils.SquadProcessor`

### class transformers.data.processors.squad.SquadProcessor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/processors/squad.py#L541)

( )

Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.

#### get\_dev\_examples

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/processors/squad.py#L629)

( data\_dir filename = None )

Returns the evaluation example from the data directory.

#### get\_examples\_from\_dataset

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/processors/squad.py#L574)

( dataset evaluate = False )

Creates a list of `SquadExample` using a TFDS dataset.

Examples:

```
>>> import tensorflow_datasets as tfds

>>> dataset = tfds.load("squad")

>>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
>>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
```

#### get\_train\_examples

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/processors/squad.py#L607)

( data\_dir filename = None )

Returns the training examples from the data directory.

Additionally, the following method can be used to convert SQuAD examples into `~data.processors.utils.SquadFeatures` that can be used as model inputs.

#### transformers.squad\_convert\_examples\_to\_features

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/data/processors/squad.py#L316)

( examples tokenizer max\_seq\_length doc\_stride max\_query\_length is\_training padding\_strategy = 'max\_length' return\_dataset = False threads = 1 tqdm\_enabled = True )

Converts a list of examples into a list of features that can be directly given as input to a model. It is model-dependant and takes advantage of many of the tokenizer’s features to create the model’s inputs.

Example:

```
processor = SquadV2Processor()
examples = processor.get_dev_examples(data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=args.max_query_length,
    is_training=not evaluate,
)
```

These processors as well as the aforementioned method can be used with files containing the data as well as with the _tensorflow\_datasets_ package. Examples are given below.

### Example usage

Here is an example using the processors as well as the conversion method using data files:

```
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)


processor = SquadV1Processor()
examples = processor.get_dev_examples(squad_v1_data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

Using _tensorflow\_datasets_ is as easy as using a data file:

```
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

Another example using these processors is given in the [run\_squad.py](https://github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py) script.