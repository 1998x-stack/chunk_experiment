The [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) class provides an API for feature-complete training in PyTorch for most standard use cases. It’s used in most of the [example scripts](https://github.com/huggingface/transformers/tree/main/examples).

Before instantiating your [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer), create a [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments) to access all the points of customization during training.

The API supports distributed training on multiple GPUs/TPUs, mixed precision through [NVIDIA Apex](https://github.com/NVIDIA/apex) and Native AMP for PyTorch.

The [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) contains the basic training loop which supports the above features. To inject custom behavior you can subclass them and override the following methods:

-   **get\_train\_dataloader** — Creates the training DataLoader.
-   **get\_eval\_dataloader** — Creates the evaluation DataLoader.
-   **get\_test\_dataloader** — Creates the test DataLoader.
-   **log** — Logs information on the various objects watching training.
-   **create\_optimizer\_and\_scheduler** — Sets up the optimizer and learning rate scheduler if they were not passed at init. Note, that you can also subclass or override the `create_optimizer` and `create_scheduler` methods separately.
-   **create\_optimizer** — Sets up the optimizer if it wasn’t passed at init.
-   **create\_scheduler** — Sets up the learning rate scheduler if it wasn’t passed at init.
-   **compute\_loss** - Computes the loss on a batch of training inputs.
-   **training\_step** — Performs a training step.
-   **prediction\_step** — Performs an evaluation/test step.
-   **evaluate** — Runs an evaluation loop and returns metrics.
-   **predict** — Returns predictions (with metrics if labels are available) on a test set.

The [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) class is optimized for 🤗 Transformers models and can have surprising behaviors when you use it on other models. When using it on your own model, make sure:

-   your model always return tuples or subclasses of [ModelOutput](/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput).
-   your model can compute the loss if a `labels` argument is provided and that loss is returned as the first element of the tuple (if your model returns tuples)
-   your model can accept multiple label arguments (use the `label_names` in your [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments) to indicate their name to the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer)) but none of them should be named `"label"`.

Here is an example of how to customize [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) to use a weighted loss (useful when you have an unbalanced training set):

```
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
```

Another way to customize the training loop behavior for the PyTorch [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) is to use [callbacks](callback) that can inspect the training loop state (for progress reporting, logging on TensorBoard or other ML platforms…) and take decisions (like early stopping).

# Trainer

### class transformers.Trainer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L236)

( model: typing.Union\[transformers.modeling\_utils.PreTrainedModel, torch.nn.modules.module.Module\] = Noneargs: TrainingArguments = Nonedata\_collator: typing.Optional\[DataCollator\] = Nonetrain\_dataset: typing.Optional\[torch.utils.data.dataset.Dataset\] = Noneeval\_dataset: typing.Union\[torch.utils.data.dataset.Dataset, typing.Dict\[str, torch.utils.data.dataset.Dataset\], NoneType\] = Nonetokenizer: typing.Optional\[transformers.tokenization\_utils\_base.PreTrainedTokenizerBase\] = Nonemodel\_init: typing.Union\[typing.Callable\[\[\], transformers.modeling\_utils.PreTrainedModel\], NoneType\] = Nonecompute\_metrics: typing.Union\[typing.Callable\[\[transformers.trainer\_utils.EvalPrediction\], typing.Dict\], NoneType\] = Nonecallbacks: typing.Optional\[typing.List\[transformers.trainer\_callback.TrainerCallback\]\] = Noneoptimizers: typing.Tuple\[torch.optim.optimizer.Optimizer, torch.optim.lr\_scheduler.LambdaLR\] = (None, None)preprocess\_logits\_for\_metrics: typing.Union\[typing.Callable\[\[torch.Tensor, torch.Tensor\], torch.Tensor\], NoneType\] = None )

Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

Important attributes:

-   **model** — Always points to the core model. If using a transformers model, it will be a [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel) subclass.
-   **model\_wrapped** — Always points to the most external model in case one or more other modules wrap the original model. This is the model that should be used for the forward pass. For example, under `DeepSpeed`, the inner model is wrapped in `DeepSpeed` and then again in `torch.nn.DistributedDataParallel`. If the inner model hasn’t been wrapped, then `self.model_wrapped` is the same as `self.model`.
-   **is\_model\_parallel** — Whether or not a model has been switched to a model parallel mode (different from data parallelism, this means some of the model layers are split on different GPUs).
-   **place\_model\_on\_device** — Whether or not to automatically place the model on the device - it will be set to `False` if model parallel or deepspeed is used, or if the default `TrainingArguments.place_model_on_device` is overridden to return `False` .
-   **is\_in\_train** — Whether or not a model is currently running `train` (e.g. when `evaluate` is called while in `train`)

#### add\_callback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L700)

( callback )

Parameters

-   **callback** (`type` or `~transformer.TrainerCallback`) — A `~transformer.TrainerCallback` class or an instance of a `~transformer.TrainerCallback`. In the first case, will instantiate a member of that class.

Add a callback to the current list of `~transformer.TrainerCallback`.

#### autocast\_smart\_context\_manager

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L2734)

( cache\_enabled: typing.Optional\[bool\] = True )

A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired arguments, depending on the situation.

#### compute\_loss

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L2791)

( modelinputsreturn\_outputs = False )

How the loss is computed by Trainer. By default, all models return the loss in the first element.

Subclass and override for custom behavior.

A helper wrapper to group together context managers.

#### create\_model\_card

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L3594)

( language: typing.Optional\[str\] = Nonelicense: typing.Optional\[str\] = Nonetags: typing.Union\[str, typing.List\[str\], NoneType\] = Nonemodel\_name: typing.Optional\[str\] = Nonefinetuned\_from: typing.Optional\[str\] = Nonetasks: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedataset\_tags: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedataset: typing.Union\[str, typing.List\[str\], NoneType\] = Nonedataset\_args: typing.Union\[str, typing.List\[str\], NoneType\] = None )

Creates a draft of a model card using the information available to the `Trainer`.

Setup the optimizer.

We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the Trainer’s init through `optimizers`, or subclass and override this method in a subclass.

#### create\_optimizer\_and\_scheduler

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L942)

( num\_training\_steps: int )

Setup the optimizer and the learning rate scheduler.

We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the Trainer’s init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or `create_scheduler`) in a subclass.

#### create\_scheduler

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L1148)

( num\_training\_steps: intoptimizer: Optimizer = None )

Parameters

-   **num\_training\_steps** (int) — The number of training steps to do.

Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or passed as an argument.

#### evaluate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L3029)

( eval\_dataset: typing.Optional\[torch.utils.data.dataset.Dataset\] = Noneignore\_keys: typing.Optional\[typing.List\[str\]\] = Nonemetric\_key\_prefix: str = 'eval' )

Parameters

-   **eval\_dataset** (`Dataset`, _optional_) — Pass a dataset if you wish to override `self.eval_dataset`. If it is a [Dataset](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__` method.
-   **ignore\_keys** (`List[str]`, _optional_) — A list of keys in the output of your model (if it is a dictionary) that should be ignored when gathering predictions.
-   **metric\_key\_prefix** (`str`, _optional_, defaults to `"eval"`) — An optional prefix to be used as the metrics key prefix. For example the metrics “bleu” will be named “eval\_bleu” if the prefix is “eval” (default)

Run evaluation and returns metrics.

The calling script will be responsible for providing a method to compute metrics, as they are task-dependent (pass it to the init `compute_metrics` argument).

You can also subclass and override this method to inject custom behavior.

#### evaluation\_loop

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L3162)

( dataloader: DataLoaderdescription: strprediction\_loss\_only: typing.Optional\[bool\] = Noneignore\_keys: typing.Optional\[typing.List\[str\]\] = Nonemetric\_key\_prefix: str = 'eval' )

Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

Works both with or without labels.

#### floating\_point\_ops

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L3502)

( inputs: typing.Dict\[str, typing.Union\[torch.Tensor, typing.Any\]\] ) → `int`

Parameters

-   **inputs** (`Dict[str, Union[torch.Tensor, Any]]`) — The inputs and targets of the model.

The number of floating-point operations.

For models that inherit from [PreTrainedModel](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel), uses that method to compute the number of floating point operations for every backward + forward pass. If using another model, either implement such a method in the model or subclass and override this method.

Get all parameter names that weight decay will be applied to

Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still apply to those modules since this function only filter out instance of nn.LayerNorm

#### get\_eval\_dataloader

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L876)

( eval\_dataset: typing.Optional\[torch.utils.data.dataset.Dataset\] = None )

Parameters

-   **eval\_dataset** (`torch.utils.data.Dataset`, _optional_) — If provided, will override `self.eval_dataset`. If it is a [Dataset](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the `model.forward()` method are automatically removed. It must implement `__len__`.

Returns the evaluation `~torch.utils.data.DataLoader`.

Subclass and override this method if you want to inject some custom behavior.

#### get\_optimizer\_cls\_and\_kwargs

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L1024)

( args: TrainingArguments )

Parameters

-   **args** (`transformers.training_args.TrainingArguments`) — The training arguments for the training session.

Returns the optimizer class and optimizer parameters based on the training arguments.

#### get\_test\_dataloader

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L910)

( test\_dataset: Dataset )

Parameters

-   **test\_dataset** (`torch.utils.data.Dataset`, _optional_) — The test dataset to use. If it is a [Dataset](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the `model.forward()` method are automatically removed. It must implement `__len__`.

Returns the test `~torch.utils.data.DataLoader`.

Subclass and override this method if you want to inject some custom behavior.

Returns the training `~torch.utils.data.DataLoader`.

Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed training if necessary) otherwise.

Subclass and override this method if you want to inject some custom behavior.

#### hyperparameter\_search

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L2598)

( hp\_space: typing.Union\[typing.Callable\[\[ForwardRef('optuna.Trial')\], typing.Dict\[str, float\]\], NoneType\] = Nonecompute\_objective: typing.Union\[typing.Callable\[\[typing.Dict\[str, float\]\], float\], NoneType\] = Nonen\_trials: int = 20direction: typing.Union\[str, typing.List\[str\]\] = 'minimize'backend: typing.Union\[ForwardRef('str'), transformers.trainer\_utils.HPSearchBackend, NoneType\] = Nonehp\_name: typing.Union\[typing.Callable\[\[ForwardRef('optuna.Trial')\], str\], NoneType\] = None\*\*kwargs ) → \[`trainer_utils.BestRun` or `List[trainer_utils.BestRun]`\]

Launch an hyperparameter search using `optuna` or `Ray Tune` or `SigOpt`. The optimized quantity is determined by `compute_objective`, which defaults to a function returning the evaluation loss when no metric is provided, the sum of all metrics otherwise.

To use this method, you need to have provided a `model_init` when initializing your [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer): we need to reinitialize the model at each new run. This is incompatible with the `optimizers` argument, so you need to subclass [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) and override the method [create\_optimizer\_and\_scheduler()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.create_optimizer_and_scheduler) for custom optimizer/scheduler.

#### init\_git\_repo

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L3537)

( at\_init: bool = False )

Parameters

-   **at\_init** (`bool`, _optional_, defaults to `False`) — Whether this function is called before any training or not. If `self.args.overwrite_output_dir` is `True` and `at_init` is `True`, the path to the repo (which is `self.args.output_dir`) might be wiped out.

Initializes a git repo in `self.args.hub_model_id`.

This function is deprecated and will be removed in v4.34.0 of Transformers.

Initializes a git repo in `self.args.hub_model_id`.

Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several machines) main process.

Whether or not this process is the global main process (when training in a distributed fashion on several machines, this is only going to be `True` for one process).

#### log

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L2677)

( logs: typing.Dict\[str, float\] )

Parameters

-   **logs** (`Dict[str, float]`) — The values to log.

Log `logs` on the various objects watching training.

Subclass and override this method to inject custom behavior.

#### log\_metrics

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_pt_utils.py#L905)

( splitmetrics )

Parameters

-   **split** (`str`) — Mode/split name: one of `train`, `eval`, `test`
-   **metrics** (`Dict[str, float]`) — The metrics returned from train/evaluate/predictmetrics: metrics dict

Log metrics in a specially formatted way

Under distributed environment this is done only for a process with rank 0.

Notes on memory reports:

In order to get memory usage report you need to install `psutil`. You can do that with `pip install psutil`.

Now when this method is run, you will see a report that will include: :

```
init_mem_cpu_alloc_delta   =     1301MB
init_mem_cpu_peaked_delta  =      154MB
init_mem_gpu_alloc_delta   =      230MB
init_mem_gpu_peaked_delta  =        0MB
train_mem_cpu_alloc_delta  =     1345MB
train_mem_cpu_peaked_delta =        0MB
train_mem_gpu_alloc_delta  =      693MB
train_mem_gpu_peaked_delta =        7MB
```

**Understanding the reports:**

-   the first segment, e.g., `train__`, tells you which stage the metrics are for. Reports starting with `init_` will be added to the first stage that gets run. So that if only evaluation is run, the memory usage for the `__init__` will be reported along with the `eval_` metrics.
-   the third segment, is either `cpu` or `gpu`, tells you whether it’s the general RAM or the gpu0 memory metric.
-   `*_alloc_delta` - is the difference in the used/allocated memory counter between the end and the start of the stage - it can be negative if a function released more memory than it allocated.
-   `*_peaked_delta` - is any extra memory that was consumed and then freed - relative to the current allocated memory counter - it is never negative. When you look at the metrics of any stage you add up `alloc_delta` + `peaked_delta` and you know how much memory was needed to complete that stage.

The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more memory than the rest since it stores the gradient and optimizer states for all participating GPUS. Perhaps in the future these reports will evolve to measure those too.

The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the memory shared with other processes. It is important to note that it does not include swapped out memory, so the reports could be imprecise.

The CPU peak memory is measured using a sampling thread. Due to python’s GIL it may miss some of the peak memory if that thread didn’t get a chance to run when the highest memory was used. Therefore this report can be less than reality. Using `tracemalloc` would have reported the exact peak memory, but it doesn’t report memory allocations outside of python. So if some C++ CUDA extension allocated its own memory it won’t be reported. And therefore it was dropped in favor of the memory sampling approach, which reads the current process memory usage.

The GPU allocated and peak memory reporting is done with `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()`. This metric reports only “deltas” for pytorch-specific allocations, as `torch.cuda` memory management system doesn’t track any memory allocated outside of pytorch. For example, the very first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.

Note that this tracker doesn’t account for memory allocations outside of [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer)’s `__init__`, `train`, `evaluate` and `predict` calls.

Because `evaluation` calls may happen during `train`, we can’t handle nested invocations because `torch.cuda.max_memory_allocated` is a single counter, so if it gets reset by a nested eval call, `train`’s tracker will report incorrect info. If this [pytorch issue](https://github.com/pytorch/pytorch/issues/16266) gets resolved it will be possible to change this class to be re-entrant. Until then we will only track the outer level of `train`, `evaluate` and `predict` methods. Which means that if `eval` is called during `train`, it’s the latter that will account for its memory usage and that of the former.

This also means that if any other tool that is used along the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) calls `torch.cuda.reset_peak_memory_stats`, the gpu peak memory stats could be invalid. And the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) will disrupt the normal behavior of any such tools that rely on calling `torch.cuda.reset_peak_memory_stats` themselves.

For best performance you may want to consider turning the memory profiling off for production runs.

#### metrics\_format

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_pt_utils.py#L879)

( metrics: typing.Dict\[str, float\] ) → metrics (`Dict[str, float]`)

Parameters

-   **metrics** (`Dict[str, float]`) — The metrics returned from train/evaluate/predict

Returns

metrics (`Dict[str, float]`)

The reformatted metrics

Reformat Trainer metrics values to a human-readable format

Helper to get number of samples in a `~torch.utils.data.DataLoader` by accessing its dataset. When dataloader.dataset does not exist or has no length, estimates as best it can

#### num\_tokens

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L1180)

( train\_dl: DataLoadermax\_steps: typing.Optional\[int\] = None )

Helper to get number of tokens in a `~torch.utils.data.DataLoader` by enumerating dataloader.

#### pop\_callback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L711)

( callback ) → `~transformer.TrainerCallback`

Parameters

-   **callback** (`type` or `~transformer.TrainerCallback`) — A `~transformer.TrainerCallback` class or an instance of a `~transformer.TrainerCallback`. In the first case, will pop the first member of that class found in the list of callbacks.

Returns

`~transformer.TrainerCallback`

The callback removed, if found.

Remove a callback from the current list of `~transformer.TrainerCallback` and returns it.

If the callback is not found, returns `None` (and no error is raised).

#### predict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L3100)

( test\_dataset: Datasetignore\_keys: typing.Optional\[typing.List\[str\]\] = Nonemetric\_key\_prefix: str = 'test' )

Parameters

-   **test\_dataset** (`Dataset`) — Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the `model.forward()` method are automatically removed. Has to implement the method `__len__`
-   **ignore\_keys** (`List[str]`, _optional_) — A list of keys in the output of your model (if it is a dictionary) that should be ignored when gathering predictions.
-   **metric\_key\_prefix** (`str`, _optional_, defaults to `"test"`) — An optional prefix to be used as the metrics key prefix. For example the metrics “bleu” will be named “test\_bleu” if the prefix is “test” (default)

Run prediction and returns predictions and potential metrics.

Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method will also return metrics, like in `evaluate()`.

If your predictions or labels have different sequence length (for instance because you’re doing dynamic padding in a token classification task) the predictions will be padded (on the right) to allow for concatenation into one array. The padding index is -100.

Returns: _NamedTuple_ A namedtuple with the following keys:

-   predictions (`np.ndarray`): The predictions on `test_dataset`.
-   label\_ids (`np.ndarray`, _optional_): The labels (if the dataset contained some).
-   metrics (`Dict[str, float]`, _optional_): The potential dictionary of metrics (if the dataset contained labels).

#### prediction\_loop

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L3767)

( dataloader: DataLoaderdescription: strprediction\_loss\_only: typing.Optional\[bool\] = Noneignore\_keys: typing.Optional\[typing.List\[str\]\] = Nonemetric\_key\_prefix: str = 'eval' )

Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

Works both with or without labels.

#### prediction\_step

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L3397)

( model: Moduleinputs: typing.Dict\[str, typing.Union\[torch.Tensor, typing.Any\]\]prediction\_loss\_only: boolignore\_keys: typing.Optional\[typing.List\[str\]\] = None ) → Tuple\[Optional\[torch.Tensor\], Optional\[torch.Tensor\], Optional\[torch.Tensor\]\]

Parameters

-   **model** (`nn.Module`) — The model to evaluate.
-   **inputs** (`Dict[str, Union[torch.Tensor, Any]]`) — The inputs and targets of the model.
    
    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the argument `labels`. Check your model’s documentation for all accepted arguments.
    
-   **prediction\_loss\_only** (`bool`) — Whether or not to return the loss only.
-   **ignore\_keys** (`List[str]`, _optional_) — A list of keys in the output of your model (if it is a dictionary) that should be ignored when gathering predictions.

Returns

Tuple\[Optional\[torch.Tensor\], Optional\[torch.Tensor\], Optional\[torch.Tensor\]\]

A tuple with the loss, logits and labels (each being optional).

Perform an evaluation step on `model` using `inputs`.

Subclass and override to inject custom behavior.

#### push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L3714)

( commit\_message: typing.Optional\[str\] = 'End of training'blocking: bool = True\*\*kwargs )

Parameters

-   **commit\_message** (`str`, _optional_, defaults to `"End of training"`) — Message to commit while pushing.
-   **blocking** (`bool`, _optional_, defaults to `True`) — Whether the function should return only when the `git push` has finished.
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional keyword arguments passed along to [create\_model\_card()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.create_model_card).

Upload `self.model` and `self.tokenizer` to the 🤗 model hub on the repo `self.args.hub_model_id`.

#### remove\_callback

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L727)

( callback )

Parameters

-   **callback** (`type` or `~transformer.TrainerCallback`) — A `~transformer.TrainerCallback` class or an instance of a `~transformer.TrainerCallback`. In the first case, will remove the first member of that class found in the list of callbacks.

Remove a callback from the current list of `~transformer.TrainerCallback`.

#### save\_metrics

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_pt_utils.py#L995)

( splitmetricscombined = True )

Parameters

-   **split** (`str`) — Mode/split name: one of `train`, `eval`, `test`, `all`
-   **metrics** (`Dict[str, float]`) — The metrics returned from train/evaluate/predict
-   **combined** (`bool`, _optional_, defaults to `True`) — Creates combined metrics by updating `all_results.json` with metrics of this call

Save metrics into a json file for that split, e.g. `train_results.json`.

Under distributed environment this is done only for a process with rank 0.

To understand the metrics please read the docstring of [log\_metrics()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.log_metrics). The only difference is that raw unformatted numbers are saved in the current method.

#### save\_model

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L2846)

( output\_dir: typing.Optional\[str\] = None\_internal\_call: bool = False )

Will save the model, so you can reload it using `from_pretrained()`.

Will only save from the main process.

Saves the Trainer state, since Trainer.save\_model saves only the tokenizer with the model

Under distributed environment this is done only for a process with rank 0.

#### train

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L1494)

( resume\_from\_checkpoint: typing.Union\[bool, str, NoneType\] = Nonetrial: typing.Union\[ForwardRef('optuna.Trial'), typing.Dict\[str, typing.Any\]\] = Noneignore\_keys\_for\_eval: typing.Optional\[typing.List\[str\]\] = None\*\*kwargs )

Parameters

-   **resume\_from\_checkpoint** (`str` or `bool`, _optional_) — If a `str`, local path to a saved checkpoint as saved by a previous instance of [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer). If a `bool` and equals `True`, load the last checkpoint in _args.output\_dir_ as saved by a previous instance of [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer). If present, training will resume from the model/optimizer/scheduler states loaded here.
-   **trial** (`optuna.Trial` or `Dict[str, Any]`, _optional_) — The trial run or the hyperparameter dictionary for hyperparameter search.
-   **ignore\_keys\_for\_eval** (`List[str]`, _optional_) — A list of keys in the output of your model (if it is a dictionary) that should be ignored when gathering predictions for evaluation during the training.
-   **kwargs** (`Dict[str, Any]`, _optional_) — Additional keyword arguments used to hide deprecated arguments

Main training entry point.

#### training\_step

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L2750)

( model: Moduleinputs: typing.Dict\[str, typing.Union\[torch.Tensor, typing.Any\]\] ) → `torch.Tensor`

Parameters

-   **model** (`nn.Module`) — The model to train.
-   **inputs** (`Dict[str, Union[torch.Tensor, Any]]`) — The inputs and targets of the model.
    
    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the argument `labels`. Check your model’s documentation for all accepted arguments.
    

The tensor with training loss on this batch.

Perform a training step on a batch of inputs.

Subclass and override to inject custom behavior.

## Seq2SeqTrainer

### class transformers.Seq2SeqTrainer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_seq2seq.py#L41)

( model: typing.Union\[ForwardRef('PreTrainedModel'), torch.nn.modules.module.Module\] = Noneargs: TrainingArguments = Nonedata\_collator: typing.Optional\[ForwardRef('DataCollator')\] = Nonetrain\_dataset: typing.Optional\[torch.utils.data.dataset.Dataset\] = Noneeval\_dataset: typing.Union\[torch.utils.data.dataset.Dataset, typing.Dict\[str, torch.utils.data.dataset.Dataset\], NoneType\] = Nonetokenizer: typing.Optional\[ForwardRef('PreTrainedTokenizerBase')\] = Nonemodel\_init: typing.Union\[typing.Callable\[\[\], ForwardRef('PreTrainedModel')\], NoneType\] = Nonecompute\_metrics: typing.Union\[typing.Callable\[\[ForwardRef('EvalPrediction')\], typing.Dict\], NoneType\] = Nonecallbacks: typing.Optional\[typing.List\[ForwardRef('TrainerCallback')\]\] = Noneoptimizers: typing.Tuple\[torch.optim.optimizer.Optimizer, torch.optim.lr\_scheduler.LambdaLR\] = (None, None)preprocess\_logits\_for\_metrics: typing.Union\[typing.Callable\[\[torch.Tensor, torch.Tensor\], torch.Tensor\], NoneType\] = None )

#### evaluate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_seq2seq.py#L112)

( eval\_dataset: typing.Optional\[torch.utils.data.dataset.Dataset\] = Noneignore\_keys: typing.Optional\[typing.List\[str\]\] = Nonemetric\_key\_prefix: str = 'eval'\*\*gen\_kwargs )

Parameters

-   **eval\_dataset** (`Dataset`, _optional_) — Pass a dataset if you wish to override `self.eval_dataset`. If it is an [Dataset](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__` method.
-   **ignore\_keys** (`List[str]`, _optional_) — A list of keys in the output of your model (if it is a dictionary) that should be ignored when gathering predictions.
-   **metric\_key\_prefix** (`str`, _optional_, defaults to `"eval"`) — An optional prefix to be used as the metrics key prefix. For example the metrics “bleu” will be named “eval\_bleu” if the prefix is `"eval"` (default)
-   **max\_length** (`int`, _optional_) — The maximum target length to use when predicting with the generate method.
-   **num\_beams** (`int`, _optional_) — Number of beams for beam search that will be used when predicting with the generate method. 1 means no beam search. gen\_kwargs — Additional `generate` specific kwargs.

Run evaluation and returns metrics.

The calling script will be responsible for providing a method to compute metrics, as they are task-dependent (pass it to the init `compute_metrics` argument).

You can also subclass and override this method to inject custom behavior.

#### predict

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_seq2seq.py#L167)

( test\_dataset: Datasetignore\_keys: typing.Optional\[typing.List\[str\]\] = Nonemetric\_key\_prefix: str = 'test'\*\*gen\_kwargs )

Parameters

-   **test\_dataset** (`Dataset`) — Dataset to run the predictions on. If it is a [Dataset](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset), columns not accepted by the `model.forward()` method are automatically removed. Has to implement the method `__len__`
-   **ignore\_keys** (`List[str]`, _optional_) — A list of keys in the output of your model (if it is a dictionary) that should be ignored when gathering predictions.
-   **metric\_key\_prefix** (`str`, _optional_, defaults to `"eval"`) — An optional prefix to be used as the metrics key prefix. For example the metrics “bleu” will be named “eval\_bleu” if the prefix is `"eval"` (default)
-   **max\_length** (`int`, _optional_) — The maximum target length to use when predicting with the generate method.
-   **num\_beams** (`int`, _optional_) — Number of beams for beam search that will be used when predicting with the generate method. 1 means no beam search. gen\_kwargs — Additional `generate` specific kwargs.

Run prediction and returns predictions and potential metrics.

Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method will also return metrics, like in `evaluate()`.

If your predictions or labels have different sequence lengths (for instance because you’re doing dynamic padding in a token classification task) the predictions will be padded (on the right) to allow for concatenation into one array. The padding index is -100.

Returns: _NamedTuple_ A namedtuple with the following keys:

-   predictions (`np.ndarray`): The predictions on `test_dataset`.
-   label\_ids (`np.ndarray`, _optional_): The labels (if the dataset contained some).
-   metrics (`Dict[str, float]`, _optional_): The potential dictionary of metrics (if the dataset contained labels).

## TrainingArguments

### class transformers.TrainingArguments

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L161)

( output\_dir: stroverwrite\_output\_dir: bool = Falsedo\_train: bool = Falsedo\_eval: bool = Falsedo\_predict: bool = Falseevaluation\_strategy: typing.Union\[transformers.trainer\_utils.IntervalStrategy, str\] = 'no'prediction\_loss\_only: bool = Falseper\_device\_train\_batch\_size: int = 8per\_device\_eval\_batch\_size: int = 8per\_gpu\_train\_batch\_size: typing.Optional\[int\] = Noneper\_gpu\_eval\_batch\_size: typing.Optional\[int\] = Nonegradient\_accumulation\_steps: int = 1eval\_accumulation\_steps: typing.Optional\[int\] = Noneeval\_delay: typing.Optional\[float\] = 0learning\_rate: float = 5e-05weight\_decay: float = 0.0adam\_beta1: float = 0.9adam\_beta2: float = 0.999adam\_epsilon: float = 1e-08max\_grad\_norm: float = 1.0num\_train\_epochs: float = 3.0max\_steps: int = -1lr\_scheduler\_type: typing.Union\[transformers.trainer\_utils.SchedulerType, str\] = 'linear'warmup\_ratio: float = 0.0warmup\_steps: int = 0log\_level: typing.Optional\[str\] = 'passive'log\_level\_replica: typing.Optional\[str\] = 'warning'log\_on\_each\_node: bool = Truelogging\_dir: typing.Optional\[str\] = Nonelogging\_strategy: typing.Union\[transformers.trainer\_utils.IntervalStrategy, str\] = 'steps'logging\_first\_step: bool = Falselogging\_steps: float = 500logging\_nan\_inf\_filter: bool = Truesave\_strategy: typing.Union\[transformers.trainer\_utils.IntervalStrategy, str\] = 'steps'save\_steps: float = 500save\_total\_limit: typing.Optional\[int\] = Nonesave\_safetensors: typing.Optional\[bool\] = Falsesave\_on\_each\_node: bool = Falseno\_cuda: bool = Falseuse\_cpu: bool = Falseuse\_mps\_device: bool = Falseseed: int = 42data\_seed: typing.Optional\[int\] = Nonejit\_mode\_eval: bool = Falseuse\_ipex: bool = Falsebf16: bool = Falsefp16: bool = Falsefp16\_opt\_level: str = 'O1'half\_precision\_backend: str = 'auto'bf16\_full\_eval: bool = Falsefp16\_full\_eval: bool = Falsetf32: typing.Optional\[bool\] = Nonelocal\_rank: int = -1ddp\_backend: typing.Optional\[str\] = Nonetpu\_num\_cores: typing.Optional\[int\] = Nonetpu\_metrics\_debug: bool = Falsedebug: typing.Union\[str, typing.List\[transformers.debug\_utils.DebugOption\]\] = ''dataloader\_drop\_last: bool = Falseeval\_steps: typing.Optional\[float\] = Nonedataloader\_num\_workers: int = 0past\_index: int = -1run\_name: typing.Optional\[str\] = Nonedisable\_tqdm: typing.Optional\[bool\] = Noneremove\_unused\_columns: typing.Optional\[bool\] = Truelabel\_names: typing.Optional\[typing.List\[str\]\] = Noneload\_best\_model\_at\_end: typing.Optional\[bool\] = Falsemetric\_for\_best\_model: typing.Optional\[str\] = Nonegreater\_is\_better: typing.Optional\[bool\] = Noneignore\_data\_skip: bool = Falsesharded\_ddp: typing.Union\[typing.List\[transformers.trainer\_utils.ShardedDDPOption\], str, NoneType\] = ''fsdp: typing.Union\[typing.List\[transformers.trainer\_utils.FSDPOption\], str, NoneType\] = ''fsdp\_min\_num\_params: int = 0fsdp\_config: typing.Optional\[str\] = Nonefsdp\_transformer\_layer\_cls\_to\_wrap: typing.Optional\[str\] = Nonedeepspeed: typing.Optional\[str\] = Nonelabel\_smoothing\_factor: float = 0.0optim: typing.Union\[transformers.training\_args.OptimizerNames, str\] = 'adamw\_torch'optim\_args: typing.Optional\[str\] = Noneadafactor: bool = Falsegroup\_by\_length: bool = Falselength\_column\_name: typing.Optional\[str\] = 'length'report\_to: typing.Optional\[typing.List\[str\]\] = Noneddp\_find\_unused\_parameters: typing.Optional\[bool\] = Noneddp\_bucket\_cap\_mb: typing.Optional\[int\] = Noneddp\_broadcast\_buffers: typing.Optional\[bool\] = Nonedataloader\_pin\_memory: bool = Trueskip\_memory\_metrics: bool = Trueuse\_legacy\_prediction\_loop: bool = Falsepush\_to\_hub: bool = Falseresume\_from\_checkpoint: typing.Optional\[str\] = Nonehub\_model\_id: typing.Optional\[str\] = Nonehub\_strategy: typing.Union\[transformers.trainer\_utils.HubStrategy, str\] = 'every\_save'hub\_token: typing.Optional\[str\] = Nonehub\_private\_repo: bool = Falsehub\_always\_push: bool = Falsegradient\_checkpointing: bool = Falseinclude\_inputs\_for\_metrics: bool = Falsefp16\_backend: str = 'auto'push\_to\_hub\_model\_id: typing.Optional\[str\] = Nonepush\_to\_hub\_organization: typing.Optional\[str\] = Nonepush\_to\_hub\_token: typing.Optional\[str\] = Nonemp\_parameters: str = ''auto\_find\_batch\_size: bool = Falsefull\_determinism: bool = Falsetorchdynamo: typing.Optional\[str\] = Noneray\_scope: typing.Optional\[str\] = 'last'ddp\_timeout: typing.Optional\[int\] = 1800torch\_compile: bool = Falsetorch\_compile\_backend: typing.Optional\[str\] = Nonetorch\_compile\_mode: typing.Optional\[str\] = Nonedispatch\_batches: typing.Optional\[bool\] = Noneinclude\_tokens\_per\_second: typing.Optional\[bool\] = False )

TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop itself**.

Using [HfArgumentParser](/docs/transformers/v4.34.0/en/internal/trainer_utils#transformers.HfArgumentParser) we can turn this class into [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the command line.

Returns the log level to be used depending on whether this process is the main process of node 0, main process of node non-0, or a non-main process.

For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn’t do anything) unless overridden by `log_level` argument.

For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica` argument.

The choice between the main and replica process settings is made according to the return value of `should_log`.

#### get\_warmup\_steps

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2097)

( num\_training\_steps: int )

Get number of steps used for a linear warmup.

#### main\_process\_first

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2046)

( local = Truedesc = 'work' )

Parameters

-   **local** (`bool`, _optional_, defaults to `True`) — if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node rank 0 In multi-node environment with a shared filesystem you most likely will want to use `local=False` so that only the main process of the first node will do the processing. If however, the filesystem is not shared, then the main process of each node will need to do the processing, which is the default behavior.
-   **desc** (`str`, _optional_, defaults to `"work"`) — a work description to be used in debug logs

A context manager for torch distributed environment where on needs to do something on the main process, while blocking replicas, and when it’s finished releasing the replicas.

One such use is for `datasets`’s `map` feature which to be efficient should be run once on the main process, which upon completion saves a cached version of results and which then automatically gets loaded by the replicas.

#### set\_dataloader

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2608)

( train\_batch\_size: int = 8eval\_batch\_size: int = 8drop\_last: bool = Falsenum\_workers: int = 0pin\_memory: bool = Trueauto\_find\_batch\_size: bool = Falseignore\_data\_skip: bool = Falsesampler\_seed: typing.Optional\[int\] = None )

A method that regroups all arguments linked to the dataloaders creation.

Example:

```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_dataloader(train_batch_size=16, eval_batch_size=64)
>>> args.per_device_train_batch_size
16
```

#### set\_evaluate

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2218)

( strategy: typing.Union\[str, transformers.trainer\_utils.IntervalStrategy\] = 'no'steps: int = 500batch\_size: int = 8accumulation\_steps: typing.Optional\[int\] = Nonedelay: typing.Optional\[float\] = Noneloss\_only: bool = Falsejit\_mode: bool = False )

A method that regroups all arguments linked to the evaluation.

Example:

```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_evaluate(strategy="steps", steps=100)
>>> args.eval_steps
100
```

#### set\_logging

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2368)

( strategy: typing.Union\[str, transformers.trainer\_utils.IntervalStrategy\] = 'steps'steps: int = 500report\_to: typing.Union\[str, typing.List\[str\]\] = 'none'level: str = 'passive'first\_step: bool = Falsenan\_inf\_filter: bool = Falseon\_each\_node: bool = Falsereplica\_level: str = 'passive' )

A method that regroups all arguments linked to the evaluation.

Example:

```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_logging(strategy="steps", steps=100)
>>> args.logging_steps
100
```

#### set\_lr\_scheduler

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2563)

( name: typing.Union\[str, transformers.trainer\_utils.SchedulerType\] = 'linear'num\_epochs: float = 3.0max\_steps: int = -1warmup\_ratio: float = 0warmup\_steps: int = 0 )

Parameters

-   **name** (`str` or [SchedulerType](/docs/transformers/v4.34.0/en/main_classes/optimizer_schedules#transformers.SchedulerType), _optional_, defaults to `"linear"`) — The scheduler type to use. See the documentation of [SchedulerType](/docs/transformers/v4.34.0/en/main_classes/optimizer_schedules#transformers.SchedulerType) for all possible values.
-   **num\_epochs(`float`,** _optional_, defaults to 3.0) — Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).
-   **max\_steps** (`int`, _optional_, defaults to -1) — If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`. In case of using a finite iterable dataset the training may stop before reaching the set number of steps when all data is exhausted.
-   **warmup\_ratio** (`float`, _optional_, defaults to 0.0) — Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
-   **warmup\_steps** (`int`, _optional_, defaults to 0) — Number of steps used for a linear warmup from 0 to `learning_rate`. Overrides any effect of `warmup_ratio`.

A method that regroups all arguments linked to the learning rate scheduler and its hyperparameters.

Example:

```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_lr_scheduler(name="cosine", warmup_ratio=0.05)
>>> args.warmup_ratio
0.05
```

#### set\_optimizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2512)

( name: typing.Union\[str, transformers.training\_args.OptimizerNames\] = 'adamw\_torch'learning\_rate: float = 5e-05weight\_decay: float = 0beta1: float = 0.9beta2: float = 0.999epsilon: float = 1e-08args: typing.Optional\[str\] = None )

Parameters

-   **name** (`str` or `training_args.OptimizerNames`, _optional_, defaults to `"adamw_torch"`) — The optimizer to use: `"adamw_hf"`, `"adamw_torch"`, `"adamw_torch_fused"`, `"adamw_apex_fused"`, `"adamw_anyprecision"` or `"adafactor"`.
-   **learning\_rate** (`float`, _optional_, defaults to 5e-5) — The initial learning rate.
-   **weight\_decay** (`float`, _optional_, defaults to 0) — The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights.
-   **beta1** (`float`, _optional_, defaults to 0.9) — The beta1 hyperparameter for the adam optimizer or its variants.
-   **beta2** (`float`, _optional_, defaults to 0.999) — The beta2 hyperparameter for the adam optimizer or its variants.
-   **epsilon** (`float`, _optional_, defaults to 1e-8) — The epsilon hyperparameter for the adam optimizer or its variants.
-   **args** (`str`, _optional_) — Optional arguments that are supplied to AnyPrecisionAdamW (only useful when `optim="adamw_anyprecision"`).

A method that regroups all arguments linked to the optimizer and its hyperparameters.

Example:

```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_optimizer(name="adamw_torch", beta1=0.8)
>>> args.optim
'adamw_torch'
```

#### set\_push\_to\_hub

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2442)

( model\_id: strstrategy: typing.Union\[str, transformers.trainer\_utils.HubStrategy\] = 'every\_save'token: typing.Optional\[str\] = Noneprivate\_repo: bool = Falsealways\_push: bool = False )

A method that regroups all arguments linked to synchronizing checkpoints with the Hub.

Calling this method will set `self.push_to_hub` to `True`, which means the `output_dir` will begin a git directory synced with the repo (determined by `model_id`) and the content will be pushed each time a save is triggered (depending on`self.save_strategy`). Calling [save\_model()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.save_model) will also trigger a push.

Example:

```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_push_to_hub("me/awesome-model")
>>> args.hub_model_id
'me/awesome-model'
```

#### set\_save

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2319)

( strategy: typing.Union\[str, transformers.trainer\_utils.IntervalStrategy\] = 'steps'steps: int = 500total\_limit: typing.Optional\[int\] = Noneon\_each\_node: bool = False )

Parameters

-   **strategy** (`str` or [IntervalStrategy](/docs/transformers/v4.34.0/en/internal/trainer_utils#transformers.IntervalStrategy), _optional_, defaults to `"steps"`) — The checkpoint save strategy to adopt during training. Possible values are:
    
    -   `"no"`: No save is done during training.
    -   `"epoch"`: Save is done at the end of each epoch.
    -   `"steps"`: Save is done every `save_steps`.
    
-   **steps** (`int`, _optional_, defaults to 500) — Number of updates steps before two checkpoint saves if `strategy="steps"`.
-   **total\_limit** (`int`, _optional_) — If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in `output_dir`.
-   **on\_each\_node** (`bool`, _optional_, defaults to `False`) — When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on the main one.
    
    This should not be activated when the different nodes use the same storage as the files will be saved with the same names for each node.
    

A method that regroups all arguments linked to the evaluation.

Example:

```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_save(strategy="steps", steps=100)
>>> args.save_steps
100
```

#### set\_testing

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2279)

( batch\_size: int = 8loss\_only: bool = Falsejit\_mode: bool = False )

Parameters

-   **batch\_size** (`int` _optional_, defaults to 8) — The batch size per device (GPU/TPU core/CPU…) used for testing.
-   **loss\_only** (`bool`, _optional_, defaults to `False`) — Ignores all outputs except the loss.
-   **jit\_mode** (`bool`, _optional_) — Whether or not to use PyTorch jit trace for inference.

A method that regroups all basic arguments linked to testing on a held-out dataset.

Calling this method will automatically set `self.do_predict` to `True`.

Example:

```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_testing(batch_size=32)
>>> args.per_device_eval_batch_size
32
```

#### set\_training

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args.py#L2143)

( learning\_rate: float = 5e-05batch\_size: int = 8weight\_decay: float = 0num\_epochs: float = 3max\_steps: int = -1gradient\_accumulation\_steps: int = 1seed: int = 42gradient\_checkpointing: bool = False )

A method that regroups all basic arguments linked to the training.

Calling this method will automatically set `self.do_train` to `True`.

Example:

```
>>> from transformers import TrainingArguments

>>> args = TrainingArguments("working_dir")
>>> args = args.set_training(learning_rate=1e-4, batch_size=32)
>>> args.learning_rate
1e-4
```

Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates the token values by removing their value.

Serializes this instance to a JSON string.

Sanitized serialization to use with TensorBoard’s hparams

## Seq2SeqTrainingArguments

### class transformers.Seq2SeqTrainingArguments

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/training_args_seq2seq.py#L30)

( output\_dir: stroverwrite\_output\_dir: bool = Falsedo\_train: bool = Falsedo\_eval: bool = Falsedo\_predict: bool = Falseevaluation\_strategy: typing.Union\[transformers.trainer\_utils.IntervalStrategy, str\] = 'no'prediction\_loss\_only: bool = Falseper\_device\_train\_batch\_size: int = 8per\_device\_eval\_batch\_size: int = 8per\_gpu\_train\_batch\_size: typing.Optional\[int\] = Noneper\_gpu\_eval\_batch\_size: typing.Optional\[int\] = Nonegradient\_accumulation\_steps: int = 1eval\_accumulation\_steps: typing.Optional\[int\] = Noneeval\_delay: typing.Optional\[float\] = 0learning\_rate: float = 5e-05weight\_decay: float = 0.0adam\_beta1: float = 0.9adam\_beta2: float = 0.999adam\_epsilon: float = 1e-08max\_grad\_norm: float = 1.0num\_train\_epochs: float = 3.0max\_steps: int = -1lr\_scheduler\_type: typing.Union\[transformers.trainer\_utils.SchedulerType, str\] = 'linear'warmup\_ratio: float = 0.0warmup\_steps: int = 0log\_level: typing.Optional\[str\] = 'passive'log\_level\_replica: typing.Optional\[str\] = 'warning'log\_on\_each\_node: bool = Truelogging\_dir: typing.Optional\[str\] = Nonelogging\_strategy: typing.Union\[transformers.trainer\_utils.IntervalStrategy, str\] = 'steps'logging\_first\_step: bool = Falselogging\_steps: float = 500logging\_nan\_inf\_filter: bool = Truesave\_strategy: typing.Union\[transformers.trainer\_utils.IntervalStrategy, str\] = 'steps'save\_steps: float = 500save\_total\_limit: typing.Optional\[int\] = Nonesave\_safetensors: typing.Optional\[bool\] = Falsesave\_on\_each\_node: bool = Falseno\_cuda: bool = Falseuse\_cpu: bool = Falseuse\_mps\_device: bool = Falseseed: int = 42data\_seed: typing.Optional\[int\] = Nonejit\_mode\_eval: bool = Falseuse\_ipex: bool = Falsebf16: bool = Falsefp16: bool = Falsefp16\_opt\_level: str = 'O1'half\_precision\_backend: str = 'auto'bf16\_full\_eval: bool = Falsefp16\_full\_eval: bool = Falsetf32: typing.Optional\[bool\] = Nonelocal\_rank: int = -1ddp\_backend: typing.Optional\[str\] = Nonetpu\_num\_cores: typing.Optional\[int\] = Nonetpu\_metrics\_debug: bool = Falsedebug: typing.Union\[str, typing.List\[transformers.debug\_utils.DebugOption\]\] = ''dataloader\_drop\_last: bool = Falseeval\_steps: typing.Optional\[float\] = Nonedataloader\_num\_workers: int = 0past\_index: int = -1run\_name: typing.Optional\[str\] = Nonedisable\_tqdm: typing.Optional\[bool\] = Noneremove\_unused\_columns: typing.Optional\[bool\] = Truelabel\_names: typing.Optional\[typing.List\[str\]\] = Noneload\_best\_model\_at\_end: typing.Optional\[bool\] = Falsemetric\_for\_best\_model: typing.Optional\[str\] = Nonegreater\_is\_better: typing.Optional\[bool\] = Noneignore\_data\_skip: bool = Falsesharded\_ddp: typing.Union\[typing.List\[transformers.trainer\_utils.ShardedDDPOption\], str, NoneType\] = ''fsdp: typing.Union\[typing.List\[transformers.trainer\_utils.FSDPOption\], str, NoneType\] = ''fsdp\_min\_num\_params: int = 0fsdp\_config: typing.Optional\[str\] = Nonefsdp\_transformer\_layer\_cls\_to\_wrap: typing.Optional\[str\] = Nonedeepspeed: typing.Optional\[str\] = Nonelabel\_smoothing\_factor: float = 0.0optim: typing.Union\[transformers.training\_args.OptimizerNames, str\] = 'adamw\_torch'optim\_args: typing.Optional\[str\] = Noneadafactor: bool = Falsegroup\_by\_length: bool = Falselength\_column\_name: typing.Optional\[str\] = 'length'report\_to: typing.Optional\[typing.List\[str\]\] = Noneddp\_find\_unused\_parameters: typing.Optional\[bool\] = Noneddp\_bucket\_cap\_mb: typing.Optional\[int\] = Noneddp\_broadcast\_buffers: typing.Optional\[bool\] = Nonedataloader\_pin\_memory: bool = Trueskip\_memory\_metrics: bool = Trueuse\_legacy\_prediction\_loop: bool = Falsepush\_to\_hub: bool = Falseresume\_from\_checkpoint: typing.Optional\[str\] = Nonehub\_model\_id: typing.Optional\[str\] = Nonehub\_strategy: typing.Union\[transformers.trainer\_utils.HubStrategy, str\] = 'every\_save'hub\_token: typing.Optional\[str\] = Nonehub\_private\_repo: bool = Falsehub\_always\_push: bool = Falsegradient\_checkpointing: bool = Falseinclude\_inputs\_for\_metrics: bool = Falsefp16\_backend: str = 'auto'push\_to\_hub\_model\_id: typing.Optional\[str\] = Nonepush\_to\_hub\_organization: typing.Optional\[str\] = Nonepush\_to\_hub\_token: typing.Optional\[str\] = Nonemp\_parameters: str = ''auto\_find\_batch\_size: bool = Falsefull\_determinism: bool = Falsetorchdynamo: typing.Optional\[str\] = Noneray\_scope: typing.Optional\[str\] = 'last'ddp\_timeout: typing.Optional\[int\] = 1800torch\_compile: bool = Falsetorch\_compile\_backend: typing.Optional\[str\] = Nonetorch\_compile\_mode: typing.Optional\[str\] = Nonedispatch\_batches: typing.Optional\[bool\] = Noneinclude\_tokens\_per\_second: typing.Optional\[bool\] = Falsesortish\_sampler: bool = Falsepredict\_with\_generate: bool = Falsegeneration\_max\_length: typing.Optional\[int\] = Nonegeneration\_num\_beams: typing.Optional\[int\] = Nonegeneration\_config: typing.Union\[str, pathlib.Path, transformers.generation.configuration\_utils.GenerationConfig, NoneType\] = None )

TrainingArguments is the subset of the arguments we use in our example scripts **which relate to the training loop itself**.

Using [HfArgumentParser](/docs/transformers/v4.34.0/en/internal/trainer_utils#transformers.HfArgumentParser) we can turn this class into [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the command line.

Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON serialization support). It obfuscates the token values by removing their value.

## Checkpoints

By default, [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) will save all checkpoints in the `output_dir` you set in the [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments) you are using. Those will go in subfolder named `checkpoint-xxx` with xxx being the step at which the training was at.

Resuming training from a checkpoint can be done when calling [Trainer.train()](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer.train) with either:

-   `resume_from_checkpoint=True` which will resume training from the latest checkpoint
-   `resume_from_checkpoint=checkpoint_dir` which will resume training from the specific checkpoint in the directory passed.

In addition, you can easily save your checkpoints on the Model Hub when using `push_to_hub=True`. By default, all the models saved in intermediate checkpoints are saved in different commits, but not the optimizer state. You can adapt the `hub-strategy` value of your [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments) to either:

-   `"checkpoint"`: the latest checkpoint is also pushed in a subfolder named last-checkpoint, allowing you to resume training easily with `trainer.train(resume_from_checkpoint="output_dir/last-checkpoint")`.
-   `"all_checkpoints"`: all checkpoints are pushed like they appear in the output folder (so you will get one checkpoint folder per folder in your final repository)

## Logging

By default [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) will use `logging.INFO` for the main process and `logging.WARNING` for the replicas if any.

These defaults can be overridden to use any of the 5 `logging` levels with [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments)’s arguments:

-   `log_level` - for the main process
-   `log_level_replica` - for the replicas

Further, if [TrainingArguments](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.TrainingArguments)’s `log_on_each_node` is set to `False` only the main node will use the log level settings for its main process, all other nodes will use the log level settings for replicas.

Note that [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) is going to set `transformers`’s log level separately for each node in its `Trainer.__init__()`. So you may want to set this sooner (see the next example) if you tap into other `transformers` functionality before creating the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) object.

Here is an example of how this can be used in an application:

```
[...]
logger = logging.getLogger(__name__)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

trainer = Trainer(...)
```

And then if you only want to see warnings on the main node and all other nodes to not print any most likely duplicated warnings you could run it as:

```
my_app.py ... --log_level warning --log_level_replica error
```

In the multi-node environment if you also don’t want the logs to repeat for each node’s main process, you will want to change the above to:

```
my_app.py ... --log_level warning --log_level_replica error --log_on_each_node 0
```

and then only the main process of the first node will log at the “warning” level, and all other processes on the main node and all processes on other nodes will log at the “error” level.

If you need your application to be as quiet as possible you could do:

```
my_app.py ... --log_level error --log_level_replica error --log_on_each_node 0
```

(add `--log_on_each_node 0` if on multi-node environment)

## Randomness

When resuming from a checkpoint generated by [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) all efforts are made to restore the _python_, _numpy_ and _pytorch_ RNG states to the same states as they were at the moment of saving that checkpoint, which should make the “stop and resume” style of training as close as possible to non-stop training.

However, due to various default non-deterministic pytorch settings this might not fully work. If you want full determinism please refer to [Controlling sources of randomness](https://pytorch.org/docs/stable/notes/randomness). As explained in the document, that some of those settings that make things deterministic (.e.g., `torch.backends.cudnn.deterministic`) may slow things down, therefore this can’t be done by default, but you can enable those yourself if needed.

## Specific GPUs Selection

Let’s discuss how you can tell your program which GPUs are to be used and in what order.

When using [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) to use only a subset of your GPUs, you simply specify the number of GPUs to use. For example, if you have 4 GPUs, but you wish to use the first 2 you can do:

```
python -m torch.distributed.launch --nproc_per_node=2  trainer-program.py ...
```

if you have either [`accelerate`](https://github.com/huggingface/accelerate) or [`deepspeed`](https://github.com/microsoft/DeepSpeed) installed you can also accomplish the same by using one of:

```
accelerate launch --num_processes 2 trainer-program.py ...
```

```
deepspeed --num_gpus 2 trainer-program.py ...
```

You don’t need to use the Accelerate or [the Deepspeed integration](Deepspeed) features to use these launchers.

Until now you were able to tell the program how many GPUs to use. Now let’s discuss how to select specific GPUs and control their order.

The following environment variables help you control which GPUs to use and their order.

**`CUDA_VISIBLE_DEVICES`**

If you have multiple GPUs and you’d like to use only 1 or a few of those GPUs, set the environment variable `CUDA_VISIBLE_DEVICES` to a list of the GPUs to be used.

For example, let’s say you have 4 GPUs: 0, 1, 2 and 3. To run only on the physical GPUs 0 and 2, you can do:

```
CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch trainer-program.py ...
```

So now pytorch will see only 2 GPUs, where your physical GPUs 0 and 2 are mapped to `cuda:0` and `cuda:1` correspondingly.

You can even change their order:

```
CUDA_VISIBLE_DEVICES=2,0 python -m torch.distributed.launch trainer-program.py ...
```

Here your physical GPUs 0 and 2 are mapped to `cuda:1` and `cuda:0` correspondingly.

The above examples were all for `DistributedDataParallel` use pattern, but the same method works for [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) as well:

```
CUDA_VISIBLE_DEVICES=2,0 python trainer-program.py ...
```

To emulate an environment without GPUs simply set this environment variable to an empty value like so:

```
CUDA_VISIBLE_DEVICES= python trainer-program.py ...
```

As with any environment variable you can, of course, export those instead of adding these to the command line, as in:

```
export CUDA_VISIBLE_DEVICES=0,2
python -m torch.distributed.launch trainer-program.py ...
```

but this approach can be confusing since you may forget you set up the environment variable earlier and not understand why the wrong GPUs are used. Therefore, it’s a common practice to set the environment variable just for a specific run on the same command line as it’s shown in most examples of this section.

**`CUDA_DEVICE_ORDER`**

There is an additional environment variable `CUDA_DEVICE_ORDER` that controls how the physical devices are ordered. The two choices are:

1.  ordered by PCIe bus IDs (matches `nvidia-smi`’s order) - this is the default.

```
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

2.  ordered by GPU compute capabilities

```
export CUDA_DEVICE_ORDER=FASTEST_FIRST
```

Most of the time you don’t need to care about this environment variable, but it’s very helpful if you have a lopsided setup where you have an old and a new GPUs physically inserted in such a way so that the slow older card appears to be first. One way to fix that is to swap the cards. But if you can’t swap the cards (e.g., if the cooling of the devices gets impacted) then setting `CUDA_DEVICE_ORDER=FASTEST_FIRST` will always put the newer faster card first. It’ll be somewhat confusing though since `nvidia-smi` will still report them in the PCIe order.

The other solution to swapping the order is to use:

```
export CUDA_VISIBLE_DEVICES=1,0
```

In this example we are working with just 2 GPUs, but of course the same would apply to as many GPUs as your computer has.

Also if you do set this environment variable it’s the best to set it in your `~/.bashrc` file or some other startup config file and forget about it.

## Trainer Integrations

The [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) has been extended to support libraries that may dramatically improve your training time and fit much bigger models.

Currently it supports third party solutions, [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html), which implement parts of the paper [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, by Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He](https://arxiv.org/abs/1910.02054).

This provided support is new and experimental as of this writing. While the support for DeepSpeed and PyTorch FSDP is active and we welcome issues around it, we don’t support the FairScale integration anymore since it has been integrated in PyTorch main (see the [PyTorch FSDP integration](#pytorch-fully-sharded-data-parallel))

### CUDA Extension Installation Notes

As of this writing, Deepspeed require compilation of CUDA C++ code, before it can be used.

While all installation issues should be dealt with through the corresponding GitHub Issues of [Deepspeed](https://github.com/microsoft/DeepSpeed/issues), there are a few common issues that one may encounter while building any PyTorch extension that needs to build CUDA extensions.

Therefore, if you encounter a CUDA-related build issue while doing the following:

please, read the following notes first.

In these notes we give examples for what to do when `pytorch` has been built with CUDA `10.2`. If your situation is different remember to adjust the version number to the one you are after.

#### Possible problem #1

While, Pytorch comes with its own CUDA toolkit, to build these two projects you must have an identical version of CUDA installed system-wide.

For example, if you installed `pytorch` with `cudatoolkit==10.2` in the Python environment, you also need to have CUDA `10.2` installed system-wide.

The exact location may vary from system to system, but `/usr/local/cuda-10.2` is the most common location on many Unix systems. When CUDA is correctly set up and added to the `PATH` environment variable, one can find the installation location by doing:

If you don’t have CUDA installed system-wide, install it first. You will find the instructions by using your favorite search engine. For example, if you’re on Ubuntu you may want to search for: [ubuntu cuda 10.2 install](https://www.google.com/search?q=ubuntu+cuda+10.2+install).

#### Possible problem #2

Another possible common problem is that you may have more than one CUDA toolkit installed system-wide. For example you may have:

```
/usr/local/cuda-10.2
/usr/local/cuda-11.0
```

Now, in this situation you need to make sure that your `PATH` and `LD_LIBRARY_PATH` environment variables contain the correct paths to the desired CUDA version. Typically, package installers will set these to contain whatever the last version was installed. If you encounter the problem, where the package build fails because it can’t find the right CUDA version despite you having it installed system-wide, it means that you need to adjust the 2 aforementioned environment variables.

First, you may look at their contents:

```
echo $PATH
echo $LD_LIBRARY_PATH
```

so you get an idea of what is inside.

It’s possible that `LD_LIBRARY_PATH` is empty.

`PATH` lists the locations of where executables can be found and `LD_LIBRARY_PATH` is for where shared libraries are to looked for. In both cases, earlier entries have priority over the later ones. `:` is used to separate multiple entries.

Now, to tell the build program where to find the specific CUDA toolkit, insert the desired paths to be listed first by doing:

```
export PATH=/usr/local/cuda-10.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
```

Note that we aren’t overwriting the existing values, but prepending instead.

Of course, adjust the version number, the full path if need be. Check that the directories you assign actually do exist. `lib64` sub-directory is where the various CUDA `.so` objects, like `libcudart.so` reside, it’s unlikely that your system will have it named differently, but if it is adjust it to reflect your reality.

#### Possible problem #3

Some older CUDA versions may refuse to build with newer compilers. For example, you my have `gcc-9` but it wants `gcc-7`.

There are various ways to go about it.

If you can install the latest CUDA toolkit it typically should support the newer compiler.

Alternatively, you could install the lower version of the compiler in addition to the one you already have, or you may already have it but it’s not the default one, so the build system can’t see it. If you have `gcc-7` installed but the build system complains it can’t find it, the following might do the trick:

```
sudo ln -s /usr/bin/gcc-7  /usr/local/cuda-10.2/bin/gcc
sudo ln -s /usr/bin/g++-7  /usr/local/cuda-10.2/bin/g++
```

Here, we are making a symlink to `gcc-7` from `/usr/local/cuda-10.2/bin/gcc` and since `/usr/local/cuda-10.2/bin/` should be in the `PATH` environment variable (see the previous problem’s solution), it should find `gcc-7` (and `g++7`) and then the build will succeed.

As always make sure to edit the paths in the example to match your situation.

### PyTorch Fully Sharded Data parallel

To accelerate training huge models on larger batch sizes, we can use a fully sharded data parallel model. This type of data parallel paradigm enables fitting more data and larger models by sharding the optimizer states, gradients and parameters. To read more about it and the benefits, check out the [Fully Sharded Data Parallel blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/). We have integrated the latest PyTorch’s Fully Sharded Data Parallel (FSDP) training feature. All you need to do is enable it through the config.

**Required PyTorch version for FSDP support**: PyTorch Nightly (or 1.12.0 if you read this after it has been released) as the model saving with FSDP activated is only available with recent fixes.

**Usage**:

-   Make sure you have added the distributed launcher `-m torch.distributed.launch --nproc_per_node=NUMBER_OF_GPUS_YOU_HAVE` if you haven’t been using it already.
    
-   **Sharding Strategy**:
    
    -   FULL\_SHARD : Shards optimizer states + gradients + model parameters across data parallel workers/GPUs. For this, add `--fsdp full_shard` to the command line arguments.
    -   SHARD\_GRAD\_OP : Shards optimizer states + gradients across data parallel workers/GPUs. For this, add `--fsdp shard_grad_op` to the command line arguments.
    -   NO\_SHARD : No sharding. For this, add `--fsdp no_shard` to the command line arguments.
-   To offload the parameters and gradients to the CPU, add `--fsdp "full_shard offload"` or `--fsdp "shard_grad_op offload"` to the command line arguments.
    
-   To automatically recursively wrap layers with FSDP using `default_auto_wrap_policy`, add `--fsdp "full_shard auto_wrap"` or `--fsdp "shard_grad_op auto_wrap"` to the command line arguments.
    
-   To enable both CPU offloading and auto wrapping, add `--fsdp "full_shard offload auto_wrap"` or `--fsdp "shard_grad_op offload auto_wrap"` to the command line arguments.
    
-   Remaining FSDP config is passed via `--fsdp_config <path_to_fsdp_config.json>`. It is either a location of FSDP json config file (e.g., `fsdp_config.json`) or an already loaded json file as `dict`.
    
    -   If auto wrapping is enabled, you can either use transformer based auto wrap policy or size based auto wrap policy.
        -   For transformer based auto wrap policy, it is recommended to specify `fsdp_transformer_layer_cls_to_wrap` in the config file. If not specified, the default value is `model._no_split_modules` when available. This specifies the list of transformer layer class name (case-sensitive) to wrap ,e.g, `BertLayer`, `GPTJBlock`, `T5Block` … This is important because submodules that share weights (e.g., embedding layer) should not end up in different FSDP wrapped units. Using this policy, wrapping happens for each block containing Multi-Head Attention followed by couple of MLP layers. Remaining layers including the shared embeddings are conveniently wrapped in same outermost FSDP unit. Therefore, use this for transformer based models.
        -   For size based auto wrap policy, please add `fsdp_min_num_params` in the config file. It specifies FSDP’s minimum number of parameters for auto wrapping.
    -   `fsdp_backward_prefetch` can be specified in the config file. It controls when to prefetch next set of parameters. `backward_pre` and `backward_pos` are available options. For more information refer `torch.distributed.fsdp.fully_sharded_data_parallel.BackwardPrefetch`
    -   `fsdp_forward_prefetch` can be specified in the config file. It controls when to prefetch next set of parameters. If `"True"`, FSDP explicitly prefetches the next upcoming all-gather while executing in the forward pass.
    -   `limit_all_gathers` can be specified in the config file. If `"True"`, FSDP explicitly synchronizes the CPU thread to prevent too many in-flight all-gathers.
    -   `activation_checkpointing` can be specified in the config file. If `"True"`, FSDP activation checkpointing is a technique to reduce memory usage by clearing activations of certain layers and recomputing them during a backward pass. Effectively, this trades extra computation time for reduced memory usage.

**Few caveats to be aware of**

-   it is incompatible with `generate`, thus is incompatible with `--predict_with_generate` in all seq2seq/clm scripts (translation/summarization/clm etc.).  
    Please refer issue [#21667](https://github.com/huggingface/transformers/issues/21667)

### PyTorch/XLA Fully Sharded Data parallel

For all the TPU users, great news! PyTorch/XLA now supports FSDP. All the latest Fully Sharded Data Parallel (FSDP) training are supported. For more information refer to the [Scaling PyTorch models on Cloud TPUs with FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/) and [PyTorch/XLA implementation of FSDP](https://github.com/pytorch/xla/tree/master/torch_xla/distributed/fsdp) All you need to do is enable it through the config.

**Required PyTorch/XLA version for FSDP support**: >=2.0

**Usage**:

Pass `--fsdp "full shard"` along with following changes to be made in `--fsdp_config <path_to_fsdp_config.json>`:

-   `xla` should be set to `True` to enable PyTorch/XLA FSDP.
-   `xla_fsdp_settings` The value is a dictionary which stores the XLA FSDP wrapping parameters. For a complete list of options, please see [here](https://github.com/pytorch/xla/blob/master/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py).
-   `xla_fsdp_grad_ckpt`. When `True`, uses gradient checkpointing over each nested XLA FSDP wrapped layer. This setting can only be used when the xla flag is set to true, and an auto wrapping policy is specified through `fsdp_min_num_params` or `fsdp_transformer_layer_cls_to_wrap`.
-   You can either use transformer based auto wrap policy or size based auto wrap policy.
    -   For transformer based auto wrap policy, it is recommended to specify `fsdp_transformer_layer_cls_to_wrap` in the config file. If not specified, the default value is `model._no_split_modules` when available. This specifies the list of transformer layer class name (case-sensitive) to wrap ,e.g, `BertLayer`, `GPTJBlock`, `T5Block` … This is important because submodules that share weights (e.g., embedding layer) should not end up in different FSDP wrapped units. Using this policy, wrapping happens for each block containing Multi-Head Attention followed by couple of MLP layers. Remaining layers including the shared embeddings are conveniently wrapped in same outermost FSDP unit. Therefore, use this for transformer based models.
    -   For size based auto wrap policy, please add `fsdp_min_num_params` in the config file. It specifies FSDP’s minimum number of parameters for auto wrapping.

### Using Trainer for accelerated PyTorch Training on Mac

With PyTorch v1.12 release, developers and researchers can take advantage of Apple silicon GPUs for significantly faster model training. This unlocks the ability to perform machine learning workflows like prototyping and fine-tuning locally, right on Mac. Apple’s Metal Performance Shaders (MPS) as a backend for PyTorch enables this and can be used via the new `"mps"` device. This will map computational graphs and primitives on the MPS Graph framework and tuned kernels provided by MPS. For more information please refer official documents [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) and [MPS BACKEND](https://pytorch.org/docs/stable/notes/mps.html).

We strongly recommend to install PyTorch >= 1.13 (nightly version at the time of writing) on your MacOS machine. It has major fixes related to model correctness and performance improvements for transformer based models. Please refer to [https://github.com/pytorch/pytorch/issues/82707](https://github.com/pytorch/pytorch/issues/82707) for more details.

**Benefits of Training and Inference using Apple Silicon Chips**

1.  Enables users to train larger networks or batch sizes locally
2.  Reduces data retrieval latency and provides the GPU with direct access to the full memory store due to unified memory architecture. Therefore, improving end-to-end performance.
3.  Reduces costs associated with cloud-based development or the need for additional local GPUs.

**Pre-requisites**: To install torch with mps support, please follow this nice medium article [GPU-Acceleration Comes to PyTorch on M1 Macs](https://medium.com/towards-data-science/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1).

**Usage**: `mps` device will be used by default if available similar to the way `cuda` device is used. Therefore, no action from user is required. For example, you can run the official Glue text classififcation task (from the root folder) using Apple Silicon GPU with below command:

```
export TASK_NAME=mrpc

python examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir
```

**A few caveats to be aware of**

1.  Some PyTorch operations have not been implemented in mps and will throw an error. One way to get around that is to set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`, which will fallback to CPU for these operations. It still throws a UserWarning however.
2.  Distributed setups `gloo` and `nccl` are not working with `mps` device. This means that currently only single GPU of `mps` device type can be used.

Finally, please, remember that, 🤗 `Trainer` only integrates MPS backend, therefore if you have any problems or questions with regards to MPS backend usage, please, file an issue with [PyTorch GitHub](https://github.com/pytorch/pytorch/issues).

## Using Accelerate Launcher with Trainer

Accelerate now powers Trainer. In terms of what users should expect:

-   They can keep using the Trainer ingterations such as FSDP, DeepSpeed vis trainer arguments without any changes on their part.
-   They can now use Accelerate Launcher with Trainer (recommended).

Steps to use Accelerate Launcher with Trainer:

1.  Make sure 🤗 Accelerate is installed, you can’t use the `Trainer` without it anyway. If not `pip install accelerate`. You may also need to update your version of Accelerate: `pip install accelerate --upgrade`
    
2.  Run `accelerate config` and fill the questionnaire. Below are example accelerate configs: a. DDP Multi-node Multi-GPU config:
    
    ```
    compute_environment: LOCAL_MACHINE                                                                                             
    distributed_type: MULTI_GPU                                                                                                    
    downcast_bf16: 'no'
    gpu_ids: all
    machine_rank: 0 
    main_process_ip: 192.168.20.1
    main_process_port: 9898
    main_training_function: main
    mixed_precision: fp16
    num_machines: 2
    num_processes: 8
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```
    
    b. FSDP config:
    
    ```
    compute_environment: LOCAL_MACHINE
    distributed_type: FSDP
    downcast_bf16: 'no'
    fsdp_config:
      fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
      fsdp_backward_prefetch_policy: BACKWARD_PRE
      fsdp_forward_prefetch: true
      fsdp_offload_params: false
      fsdp_sharding_strategy: 1
      fsdp_state_dict_type: FULL_STATE_DICT
      fsdp_sync_module_states: true
      fsdp_transformer_layer_cls_to_wrap: BertLayer
      fsdp_use_orig_params: true
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 2
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```
    
    c. DeepSpeed config pointing to a file:
    
    ```
    compute_environment: LOCAL_MACHINE
    deepspeed_config:
      deepspeed_config_file: /home/user/configs/ds_zero3_config.json
      zero3_init_flag: true
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    machine_rank: 0
    main_training_function: main
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```
    
    d. DeepSpeed config using accelerate plugin:
    
    ```
    compute_environment: LOCAL_MACHINE                                                                                             
    deepspeed_config:                                                                                                              
      gradient_accumulation_steps: 1
      gradient_clipping: 0.7
      offload_optimizer_device: cpu
      offload_param_device: cpu
      zero3_init_flag: true
      zero_stage: 2
    distributed_type: DEEPSPEED
    downcast_bf16: 'no'
    machine_rank: 0
    main_training_function: main
    mixed_precision: bf16
    num_machines: 1
    num_processes: 4
    rdzv_backend: static
    same_network: true
    tpu_env: []
    tpu_use_cluster: false
    tpu_use_sudo: false
    use_cpu: false
    ```
    
3.  Run the Trainer script with args other than the ones handled above by accelerate config or launcher args. Below is an example to run `run_glue.py` using `accelerate launcher` with FSDP config from above.
    

```
cd transformers

accelerate launch \
./examples/pytorch/text-classification/run_glue.py \
--model_name_or_path bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

4.  You can also directly use the cmd args for `accelerate launch`. Above example would map to:

```
cd transformers

accelerate launch --num_processes=2 \
--use_fsdp \
--mixed_precision=bf16 \
--fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP  \
--fsdp_transformer_layer_cls_to_wrap="BertLayer" \
--fsdp_sharding_strategy=1 \
--fsdp_state_dict_type=FULL_STATE_DICT \
./examples/pytorch/text-classification/run_glue.py
--model_name_or_path bert-base-cased \
--task_name $TASK_NAME \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir /tmp/$TASK_NAME/ \
--overwrite_output_dir
```

For more information, please refer the 🤗 Accelerate CLI guide: [Launching your 🤗 Accelerate scripts](https://huggingface.co/docs/accelerate/basic_tutorials/launch).

Sections that were moved:

\[ [DeepSpeed](./deepspeed#deepspeed-trainer-integration) | [Installation](./deepspeed#deepspeed-installation) | [Deployment with multiple GPUs](./deepspeed#deepspeed-multi-gpu) | [Deployment with one GPU](./deepspeed#deepspeed-one-gpu) | [Deployment in Notebooks](./deepspeed#deepspeed-notebook) | [Configuration](./deepspeed#deepspeed-config) | [Passing Configuration](./deepspeed#deepspeed-config-passing) | [Shared Configuration](./deepspeed#deepspeed-config-shared) | [ZeRO](./deepspeed#deepspeed-zero) | [ZeRO-2 Config](./deepspeed#deepspeed-zero2-config) | [ZeRO-3 Config](./deepspeed#deepspeed-zero3-config) | [NVMe Support](./deepspeed#deepspeed-nvme) | [ZeRO-2 vs ZeRO-3 Performance](./deepspeed#deepspeed-zero2-zero3-performance) | [ZeRO-2 Example](./deepspeed#deepspeed-zero2-example) | [ZeRO-3 Example](./deepspeed#deepspeed-zero3-example) | [Optimizer](./deepspeed#deepspeed-optimizer) | [Scheduler](./deepspeed#deepspeed-scheduler) | [fp32 Precision](./deepspeed#deepspeed-fp32) | [Automatic Mixed Precision](./deepspeed#deepspeed-amp) | [Batch Size](./deepspeed#deepspeed-bs) | [Gradient Accumulation](./deepspeed#deepspeed-grad-acc) | [Gradient Clipping](./deepspeed#deepspeed-grad-clip) | [Getting The Model Weights Out](./deepspeed#deepspeed-weight-extraction) \]