# Utilities for pipelines

Transformers documentation

Natural Language Processing

Performance and scalability

Reinforcement learning models

This page lists all the utility functions the library provides for pipelines.

Most of those are only useful if you are studying the code of the models in the library.

## Argument handling

### class transformers.pipelines.ArgumentHandler

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L425)

( )

Base interface for handling arguments for each [Pipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.Pipeline).

### class transformers.pipelines.ZeroShotClassificationArgumentHandler

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/zero_shot_classification.py#L14)

( )

Handles arguments for zero-shot for text classification by turning each possible label into an NLI premise/hypothesis pair.

### class transformers.pipelines.QuestionAnsweringArgumentHandler

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/question_answering.py#L150)

( )

QuestionAnsweringPipeline requires the user to provide multiple arguments (i.e. question & context) to be mapped to internal `SquadExample`.

QuestionAnsweringArgumentHandler manages all the possible to create a `SquadExample` from the command-line supplied arguments.

## Data format

### class transformers.PipelineDataFormat

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L435)

( output\_path: typing.Optional\[str\] input\_path: typing.Optional\[str\] column: typing.Optional\[str\] overwrite: bool = False )

Parameters

-   **output\_path** (`str`, _optional_) — Where to save the outgoing data.
-   **input\_path** (`str`, _optional_) — Where to look for the input data.
-   **column** (`str`, _optional_) — The column to read.
-   **overwrite** (`bool`, _optional_, defaults to `False`) — Whether or not to overwrite the `output_path`.

Base class for all the pipeline supported data format both for reading and writing. Supported data formats currently includes:

-   JSON
-   CSV
-   stdin/stdout (pipe)

`PipelineDataFormat` also includes some utilities to work with multi-columns like mapping from datasets columns to pipelines keyword arguments through the `dataset_kwarg_1=dataset_column_1` format.

#### from\_str

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L512)

( format: str output\_path: typing.Optional\[str\] input\_path: typing.Optional\[str\] column: typing.Optional\[str\] overwrite = False ) → [PipelineDataFormat](/docs/transformers/v4.34.0/en/internal/pipelines_utils#transformers.PipelineDataFormat)

Parameters

-   **format** (`str`) — The format of the desired pipeline. Acceptable values are `"json"`, `"csv"` or `"pipe"`.
-   **output\_path** (`str`, _optional_) — Where to save the outgoing data.
-   **input\_path** (`str`, _optional_) — Where to look for the input data.
-   **column** (`str`, _optional_) — The column to read.
-   **overwrite** (`bool`, _optional_, defaults to `False`) — Whether or not to overwrite the `output_path`.

The proper data format.

Creates an instance of the right subclass of [PipelineDataFormat](/docs/transformers/v4.34.0/en/internal/pipelines_utils#transformers.PipelineDataFormat) depending on `format`.

#### save

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L484)

( data: typing.Union\[dict, typing.List\[dict\]\] )

Parameters

-   **data** (`dict` or list of `dict`) — The data to store.

Save the provided data object with the representation for the current [PipelineDataFormat](/docs/transformers/v4.34.0/en/internal/pipelines_utils#transformers.PipelineDataFormat).

#### save\_binary

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L494)

( data: typing.Union\[dict, typing.List\[dict\]\] ) → `str`

Parameters

-   **data** (`dict` or list of `dict`) — The data to store.

Path where the data has been saved.

Save the provided data object as a pickle-formatted binary data on the disk.

### class transformers.CsvPipelineDataFormat

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L548)

( output\_path: typing.Optional\[str\] input\_path: typing.Optional\[str\] column: typing.Optional\[str\] overwrite = False )

Parameters

-   **output\_path** (`str`, _optional_) — Where to save the outgoing data.
-   **input\_path** (`str`, _optional_) — Where to look for the input data.
-   **column** (`str`, _optional_) — The column to read.
-   **overwrite** (`bool`, _optional_, defaults to `False`) — Whether or not to overwrite the `output_path`.

Support for pipelines using CSV data format.

#### save

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L578)

( data: typing.List\[dict\] )

Parameters

-   **data** (`List[dict]`) — The data to store.

Save the provided data object with the representation for the current [PipelineDataFormat](/docs/transformers/v4.34.0/en/internal/pipelines_utils#transformers.PipelineDataFormat).

### class transformers.JsonPipelineDataFormat

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L592)

( output\_path: typing.Optional\[str\] input\_path: typing.Optional\[str\] column: typing.Optional\[str\] overwrite = False )

Parameters

-   **output\_path** (`str`, _optional_) — Where to save the outgoing data.
-   **input\_path** (`str`, _optional_) — Where to look for the input data.
-   **column** (`str`, _optional_) — The column to read.
-   **overwrite** (`bool`, _optional_, defaults to `False`) — Whether or not to overwrite the `output_path`.

Support for pipelines using JSON file format.

#### save

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L623)

( data: dict )

Parameters

-   **data** (`dict`) — The data to store.

Save the provided data object in a json file.

### class transformers.PipedPipelineDataFormat

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L634)

( output\_path: typing.Optional\[str\] input\_path: typing.Optional\[str\] column: typing.Optional\[str\] overwrite: bool = False )

Parameters

-   **output\_path** (`str`, _optional_) — Where to save the outgoing data.
-   **input\_path** (`str`, _optional_) — Where to look for the input data.
-   **column** (`str`, _optional_) — The column to read.
-   **overwrite** (`bool`, _optional_, defaults to `False`) — Whether or not to overwrite the `output_path`.

Read data from piped input to the python process. For multi columns data, columns should separated by

If columns are provided, then the output will be a dictionary with {column\_x: value\_x}

#### save

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L663)

( data: dict )

Parameters

-   **data** (`dict`) — The data to store.

Print the data.

## Utilities

### class transformers.pipelines.PipelineException

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/pipelines/base.py#L408)

( task: str model: str reason: str )

Parameters

-   **task** (`str`) — The task of the pipeline.
-   **model** (`str`) — The model used by the pipeline.
-   **reason** (`str`) — The error message to display.

Raised by a [Pipeline](/docs/transformers/v4.34.0/en/main_classes/pipelines#transformers.Pipeline) when handling **call**.