Callbacks are objects that can customize the behavior of the training loop in the PyTorch [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) (this feature is not yet implemented in TensorFlow) that can inspect the training loop state (for progress reporting, logging on TensorBoard or other ML platforms…) and take decisions (like early stopping).

Callbacks are “read only” pieces of code, apart from the [TrainerControl](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerControl) object they return, they cannot change anything in the training loop. For customizations that require changes in the training loop, you should subclass [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) and override the methods you need (see [trainer](trainer) for examples).

By default a [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) will use the following callbacks:

-   [DefaultFlowCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.DefaultFlowCallback) which handles the default behavior for logging, saving and evaluation.
-   [PrinterCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.PrinterCallback) or [ProgressCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.ProgressCallback) to display progress and print the logs (the first one is used if you deactivate tqdm through the [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments), otherwise it’s the second one).
-   [TensorBoardCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.TensorBoardCallback) if tensorboard is accessible (either through PyTorch >= 1.4 or tensorboardX).
-   [WandbCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.WandbCallback) if [wandb](https://www.wandb.com/) is installed.
-   [CometCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.CometCallback) if [comet\_ml](https://www.comet.ml/site/) is installed.
-   [MLflowCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.MLflowCallback) if [mlflow](https://www.mlflow.org/) is installed.
-   [NeptuneCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.NeptuneCallback) if [neptune](https://neptune.ai/) is installed.
-   [AzureMLCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.AzureMLCallback) if [azureml-sdk](https://pypi.org/project/azureml-sdk/) is installed.
-   [CodeCarbonCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.CodeCarbonCallback) if [codecarbon](https://pypi.org/project/codecarbon/) is installed.
-   [ClearMLCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.ClearMLCallback) if [clearml](https://github.com/allegroai/clearml) is installed.
-   [DagsHubCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.DagsHubCallback) if [dagshub](https://dagshub.com/) is installed.
-   [FlyteCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.integrations.FlyteCallback) if [flyte](https://flyte.org/) is installed.

The main class that implements callbacks is [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback). It gets the [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments) used to instantiate the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer), can access that Trainer’s internal state via [TrainerState](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerState), and can take some actions on the training loop via [TrainerControl](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerControl).

# Available Callbacks

Here is the list of the available [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) in the library:

### class transformers.integrations.CometCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/integrations/integration_utils.py#L832)

( )

A [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [Comet ML](https://www.comet.ml/site/).

Setup the optional Comet.ml integration.

Environment:

-   **COMET\_MODE** (`str`, _optional_, defaults to `ONLINE`): Whether to create an online, offline experiment or disable Comet logging. Can be `OFFLINE`, `ONLINE`, or `DISABLED`.
-   **COMET\_PROJECT\_NAME** (`str`, _optional_): Comet project name for experiments.
-   **COMET\_OFFLINE\_DIRECTORY** (`str`, _optional_): Folder to use for saving offline experiments when `COMET_MODE` is `OFFLINE`.
-   **COMET\_LOG\_ASSETS** (`str`, _optional_, defaults to `TRUE`): Whether or not to log training assets (tf event logs, checkpoints, etc), to Comet. Can be `TRUE`, or `FALSE`.

For a number of configurable items in the environment, see [here](https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables).

### class transformers.DefaultFlowCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L424)

( )

A [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) that handles the default flow of the training loop for logs, evaluation and checkpoints.

### class transformers.EarlyStoppingCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L535)

( early\_stopping\_patience: int = 1 early\_stopping\_threshold: typing.Optional\[float\] = 0.0 )

Parameters

-   **early\_stopping\_patience** (`int`) — Use with `metric_for_best_model` to stop training when the specified metric worsens for `early_stopping_patience` evaluation calls.
-   **early\_stopping\_threshold(`float`,** _optional_) — Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the specified metric must improve to satisfy early stopping conditions. \`

A [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) that handles early stopping.

This callback depends on [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments) argument _load\_best\_model\_at\_end_ functionality to set best\_metric in [TrainerState](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerState). Note that if the [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments) argument _save\_steps_ differs from _eval\_steps_, the early stopping will not occur until the next save step.

### class transformers.integrations.TensorBoardCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/integrations/integration_utils.py#L579)

( tb\_writer = None )

Parameters

-   **tb\_writer** (`SummaryWriter`, _optional_) — The writer to use. Will instantiate one if not set.

A [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).

### class transformers.integrations.WandbCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/integrations/integration_utils.py#L665)

( )

A [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).

Setup the optional Weights & Biases (_wandb_) integration.

One can subclass and override this method to customize the setup if needed. Find more information [here](https://docs.wandb.ai/guides/integrations/huggingface). You can also override the following environment variables:

Environment:

-   **WANDB\_LOG\_MODEL** (`str`, _optional_, defaults to `"false"`): Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"` or `"false"`. If set to `"end"`, the model will be uploaded at the end of training. If set to `"checkpoint"`, the checkpoint will be uploaded every `args.save_steps` . If set to `"false"`, the model will not be uploaded. Use along with `load_best_model_at_end()` to upload best model.
    
    Deprecated in 5.0
    
    Setting `WANDB_LOG_MODEL` as `bool` will be deprecated in version 5 of 🤗 Transformers.
    
-   **WANDB\_WATCH** (`str`, _optional_ defaults to `"false"`): Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`. Set to `"all"` to log gradients and parameters.
    
-   **WANDB\_PROJECT** (`str`, _optional_, defaults to `"huggingface"`): Set this to a custom string to store results in a different project.
    
-   **WANDB\_DISABLED** (`bool`, _optional_, defaults to `False`): Whether to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
    

### class transformers.integrations.MLflowCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/integrations/integration_utils.py#L932)

( )

A [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.

Setup the optional MLflow integration.

Environment:

-   **HF\_MLFLOW\_LOG\_ARTIFACTS** (`str`, _optional_): Whether to use MLflow `.log_artifact()` facility to log artifacts. This only makes sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or _1_, will copy each saved checkpoint on each save in [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments)’s `output_dir` to the local or remote artifact storage. Using it without a remote storage will just copy the files to your artifact location.
-   **MLFLOW\_EXPERIMENT\_NAME** (`str`, _optional_, defaults to `None`): Whether to use an MLflow experiment\_name under which to launch the run. Default to `None` which will point to the `Default` experiment in MLflow. Otherwise, it is a case sensitive name of the experiment to be activated. If an experiment with this name does not exist, a new experiment with this name is created.
-   **MLFLOW\_TAGS** (`str`, _optional_): A string dump of a dictionary of key/value pair to be added to the MLflow run as tags. Example: `os.environ['MLFLOW_TAGS']='{"release.candidate": "RC1", "release.version": "2.2.0"}'`.
-   **MLFLOW\_NESTED\_RUN** (`str`, _optional_): Whether to use MLflow nested runs. If set to `True` or _1_, will create a nested run inside the current run.
-   **MLFLOW\_RUN\_ID** (`str`, _optional_): Allow to reattach to an existing run which can be usefull when resuming training from a checkpoint. When `MLFLOW_RUN_ID` environment variable is set, `start_run` attempts to resume a run with the specified run ID and other parameters are ignored.
-   **MLFLOW\_FLATTEN\_PARAMS** (`str`, _optional_, defaults to `False`): Whether to flatten the parameters dictionary before logging.

### class transformers.integrations.NeptuneCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/integrations/integration_utils.py#L1127)

( api\_token: typing.Optional\[str\] = None project: typing.Optional\[str\] = None name: typing.Optional\[str\] = None base\_namespace: str = 'finetuning' run = None log\_parameters: bool = True log\_checkpoints: typing.Optional\[str\] = None \*\*neptune\_run\_kwargs )

Parameters

-   **api\_token** (`str`, _optional_) — Neptune API token obtained upon registration. You can leave this argument out if you have saved your token to the `NEPTUNE_API_TOKEN` environment variable (strongly recommended). See full setup instructions in the [docs](https://docs.neptune.ai/setup/installation).
-   **project** (`str`, _optional_) — Name of an existing Neptune project, in the form “workspace-name/project-name”. You can find and copy the name in Neptune from the project settings -> Properties. If None (default), the value of the `NEPTUNE_PROJECT` environment variable is used.
-   **name** (`str`, _optional_) — Custom name for the run.
-   **base\_namespace** (`str`, optional, defaults to “finetuning”) — In the Neptune run, the root namespace that will contain all of the metadata logged by the callback.
-   **log\_parameters** (`bool`, _optional_, defaults to `True`) — If True, logs all Trainer arguments and model parameters provided by the Trainer.
-   **log\_checkpoints** (`str`, _optional_) — If “same”, uploads checkpoints whenever they are saved by the Trainer. If “last”, uploads only the most recently saved checkpoint. If “best”, uploads the best checkpoint (among the ones saved by the Trainer). If `None`, does not upload checkpoints.
-   **run** (`Run`, _optional_) — Pass a Neptune run object if you want to continue logging to an existing run. Read more about resuming runs in the [docs](https://docs.neptune.ai/logging/to_existing_object).
-   \***\*neptune\_run\_kwargs** (_optional_) — Additional keyword arguments to be passed directly to the [`neptune.init_run()`](https://docs.neptune.ai/api/neptune#init_run) function when a new run is created.

TrainerCallback that sends the logs to [Neptune](https://app.neptune.ai/).

For instructions and examples, see the [Transformers integration guide](https://docs.neptune.ai/integrations/transformers) in the Neptune documentation.

### class transformers.integrations.ClearMLCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/integrations/integration_utils.py#L1427)

( )

A [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [ClearML](https://clear.ml/).

Environment:

-   **CLEARML\_PROJECT** (`str`, _optional_, defaults to `HuggingFace Transformers`): ClearML project name.
-   **CLEARML\_TASK** (`str`, _optional_, defaults to `Trainer`): ClearML task name.
-   **CLEARML\_LOG\_MODEL** (`bool`, _optional_, defaults to `False`): Whether to log models as artifacts during training.

### class transformers.integrations.DagsHubCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/integrations/integration_utils.py#L1067)

( )

A [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) that logs to [DagsHub](https://dagshub.com/). Extends `MLflowCallback`

Setup the DagsHub’s Logging integration.

Environment:

-   **HF\_DAGSHUB\_LOG\_ARTIFACTS** (`str`, _optional_): Whether to save the data and model artifacts for the experiment. Default to `False`.

### class transformers.integrations.FlyteCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/integrations/integration_utils.py#L1546)

( save\_log\_history: bool = True sync\_checkpoints: bool = True )

Parameters

-   **save\_log\_history** (`bool`, _optional_, defaults to `True`) — When set to True, the training logs are saved as a Flyte Deck.
-   **sync\_checkpoints** (`bool`, _optional_, defaults to `True`) — When set to True, checkpoints are synced with Flyte and can be used to resume training in the case of an interruption.

A [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) that sends the logs to [Flyte](https://flyte.org/). NOTE: This callback only works within a Flyte task.

Example:

```
from flytekit import current_context, task


@task
def train_hf_transformer():
    cp = current_context().checkpoint
    trainer = Trainer(..., callbacks=[FlyteCallback()])
    output = trainer.train(resume_from_checkpoint=cp.restore())
```

## TrainerCallback

### class transformers.TrainerCallback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L168)

( )

Parameters

-   **args** ([TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments)) — The training arguments used to instantiate the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer).
-   **state** ([TrainerState](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerState)) — The current state of the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer).
-   **control** ([TrainerControl](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerControl)) — The object that is returned to the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) and can be used to make some decisions.
-   **model** ([PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel) or `torch.nn.Module`) — The model being trained.
-   **tokenizer** ([PreTrainedTokenizer](/docs/transformers/v4.34.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer)) — The tokenizer used for encoding the data.
-   **optimizer** (`torch.optim.Optimizer`) — The optimizer used for the training steps.
-   **lr\_scheduler** (`torch.optim.lr_scheduler.LambdaLR`) — The scheduler used for setting the learning rate.
-   **train\_dataloader** (`torch.utils.data.DataLoader`, _optional_) — The current dataloader used for training.
-   **eval\_dataloader** (`torch.utils.data.DataLoader`, _optional_) — The current dataloader used for training.
-   **metrics** (`Dict[str, float]`) — The metrics computed by the last evaluation phase.
    
    Those are only accessible in the event `on_evaluate`.
    
-   **logs** (`Dict[str, float]`) — The values to log.
    
    Those are only accessible in the event `on_log`.
    

A class for objects that will inspect the state of the training loop at some events and take some decisions. At each of those events the following arguments are available:

The `control` object is the only one that can be changed by the callback, in which case the event that changes it should return the modified version.

The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`. You can unpack the ones you need in the signature of the event using them. As an example, see the code of the simple `~transformer.PrinterCallback`.

Example:

```
class PrinterCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)
```

#### on\_epoch\_begin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L236)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called at the beginning of an epoch.

#### on\_epoch\_end

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L242)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called at the end of an epoch.

#### on\_evaluate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L268)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called after an evaluation phase.

#### on\_init\_end

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L218)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called at the end of the initialization of the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer).

#### on\_log

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L286)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called after logging the last logs.

#### on\_predict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L274)

( args: TrainingArguments state: TrainerState control: TrainerControl metrics \*\*kwargs )

Event called after a successful prediction.

#### on\_prediction\_step

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L292)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called after a prediction step.

#### on\_save

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L280)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called after a checkpoint save.

#### on\_step\_begin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L248)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called at the beginning of a training step. If using gradient accumulation, one training step might take several inputs.

#### on\_step\_end

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L261)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called at the end of a training step. If using gradient accumulation, one training step might take several inputs.

#### on\_substep\_end

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L255)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called at the end of an substep during gradient accumulation.

#### on\_train\_begin

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L224)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called at the beginning of training.

#### on\_train\_end

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L230)

( args: TrainingArguments state: TrainerState control: TrainerControl \*\*kwargs )

Event called at the end of training.

Here is an example of how to register a custom callback with the PyTorch [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer):

```
class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MyCallback],  
)
```

Another way to register a callback is to call `trainer.add_callback()` as follows:

```
trainer = Trainer(...)
trainer.add_callback(MyCallback)

trainer.add_callback(MyCallback())
```

## TrainerState

### class transformers.TrainerState

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L35)

( epoch: typing.Optional\[float\] = None global\_step: int = 0 max\_steps: int = 0 logging\_steps: int = 500 eval\_steps: int = 500 save\_steps: int = 500 num\_train\_epochs: int = 0 total\_flos: float = 0 log\_history: typing.List\[typing.Dict\[str, float\]\] = None best\_metric: typing.Optional\[float\] = None best\_model\_checkpoint: typing.Optional\[str\] = None is\_local\_process\_zero: bool = True is\_world\_process\_zero: bool = True is\_hyper\_param\_search: bool = False trial\_name: str = None trial\_params: typing.Dict\[str, typing.Union\[str, float, int, bool\]\] = None )

Parameters

-   **epoch** (`float`, _optional_) — Only set during training, will represent the epoch the training is at (the decimal part being the percentage of the current epoch completed).
-   **global\_step** (`int`, _optional_, defaults to 0) — During training, represents the number of update steps completed.
-   **max\_steps** (`int`, _optional_, defaults to 0) — The number of update steps to do during the current training.
-   **logging\_steps** (`int`, _optional_, defaults to 500) — Log every X updates steps
-   **eval\_steps** (`int`, _optional_) — Run an evaluation every X steps.
-   **save\_steps** (`int`, _optional_, defaults to 500) — Save checkpoint every X updates steps.
-   **total\_flos** (`float`, _optional_, defaults to 0) — The total number of floating operations done by the model since the beginning of training (stored as floats to avoid overflow).
-   **log\_history** (`List[Dict[str, float]]`, _optional_) — The list of logs done since the beginning of training.
-   **best\_metric** (`float`, _optional_) — When tracking the best model, the value of the best metric encountered so far.
-   **best\_model\_checkpoint** (`str`, _optional_) — When tracking the best model, the value of the name of the checkpoint for the best model encountered so far.
-   **is\_local\_process\_zero** (`bool`, _optional_, defaults to `True`) — Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several machines) main process.
-   **is\_world\_process\_zero** (`bool`, _optional_, defaults to `True`) — Whether or not this process is the global main process (when training in a distributed fashion on several machines, this is only going to be `True` for one process).
-   **is\_hyper\_param\_search** (`bool`, _optional_, defaults to `False`) — Whether we are in the process of a hyper parameter search using Trainer.hyperparameter\_search. This will impact the way data will be logged in TensorBoard.

A class containing the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) inner state that will be saved along the model and optimizer when checkpointing and passed to the [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback).

In all this class, one step is to be understood as one update step. When using gradient accumulation, one update step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update step requires going through _n_ batches.

Create an instance from the content of `json_path`.

Save the content of this instance in JSON format inside `json_path`.

## TrainerControl

### class transformers.TrainerControl

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_callback.py#L119)

( should\_training\_stop: bool = False should\_epoch\_stop: bool = False should\_save: bool = False should\_evaluate: bool = False should\_log: bool = False )

Parameters

-   **should\_training\_stop** (`bool`, _optional_, defaults to `False`) — Whether or not the training should be interrupted.
    
    If `True`, this variable will not be set back to `False`. The training will just stop.
    
-   **should\_epoch\_stop** (`bool`, _optional_, defaults to `False`) — Whether or not the current epoch should be interrupted.
    
    If `True`, this variable will be set back to `False` at the beginning of the next epoch.
    
-   **should\_save** (`bool`, _optional_, defaults to `False`) — Whether or not the model should be saved at this step.
    
    If `True`, this variable will be set back to `False` at the beginning of the next step.
    
-   **should\_evaluate** (`bool`, _optional_, defaults to `False`) — Whether or not the model should be evaluated at this step.
    
    If `True`, this variable will be set back to `False` at the beginning of the next step.
    
-   **should\_log** (`bool`, _optional_, defaults to `False`) — Whether or not the logs should be reported at this step.
    
    If `True`, this variable will be set back to `False` at the beginning of the next step.
    

A class that handles the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) control flow. This class is used by the [TrainerCallback](/docs/transformers/v4.34.0/en/main_classes/callback#transformers.TrainerCallback) to activate some switches in the training loop.