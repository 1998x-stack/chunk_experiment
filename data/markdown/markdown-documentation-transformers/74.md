# Optimization

The `.optimization` module provides:

-   an optimizer with weight decay fixed that can be used to fine-tuned models, and
-   several schedules in the form of schedule objects that inherit from `_LRSchedule`:
-   a gradient accumulation class to accumulate the gradients of multiple batches

## AdamW (PyTorch)

### class transformers.AdamW

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L378)

( params: typing.Iterable\[torch.nn.parameter.Parameter\]lr: float = 0.001betas: typing.Tuple\[float, float\] = (0.9, 0.999)eps: float = 1e-06weight\_decay: float = 0.0correct\_bias: bool = Trueno\_deprecation\_warning: bool = False )

Parameters

-   **params** (`Iterable[nn.parameter.Parameter]`) — Iterable of parameters to optimize or dictionaries defining parameter groups.
-   **lr** (`float`, _optional_, defaults to 1e-3) — The learning rate to use.
-   **betas** (`Tuple[float,float]`, _optional_, defaults to (0.9, 0.999)) — Adam’s betas parameters (b1, b2).
-   **eps** (`float`, _optional_, defaults to 1e-6) — Adam’s epsilon for numerical stability.
-   **weight\_decay** (`float`, _optional_, defaults to 0) — Decoupled weight decay to apply.
-   **correct\_bias** (`bool`, _optional_, defaults to `True`) — Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
-   **no\_deprecation\_warning** (`bool`, _optional_, defaults to `False`) — A flag used to disable the deprecation warning (set to `True` to disable the warning).

Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).

#### step

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L429)

( closure: typing.Callable = None )

Parameters

-   **closure** (`Callable`, _optional_) — A closure that reevaluates the model and returns the loss.

Performs a single optimization step.

## AdaFactor (PyTorch)

### class transformers.Adafactor

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L492)

( paramslr = Noneeps = (1e-30, 0.001)clip\_threshold = 1.0decay\_rate = -0.8beta1 = Noneweight\_decay = 0.0scale\_parameter = Truerelative\_step = Truewarmup\_init = False )

AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code: [https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py](https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py)

Paper: _Adafactor: Adaptive Learning Rates with Sublinear Memory Cost_ [https://arxiv.org/abs/1804.04235](https://arxiv.org/abs/1804.04235) Note that this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and `warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and `relative_step=False`.

This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

Recommended T5 finetuning settings ([https://discuss.huggingface.co/t/t5-finetuning-tips/684/3](https://discuss.huggingface.co/t/t5-finetuning-tips/684/3)):

-   Training without LR warmup or clip\_threshold is not recommended.
    
    -   use scheduled LR warm-up to fixed LR
    -   use clip\_threshold=1.0 ([https://arxiv.org/abs/1804.04235](https://arxiv.org/abs/1804.04235))
-   Disable relative updates
    
-   Use scale\_parameter=False
    
-   Additional optimizer operations like gradient clipping should not be used alongside Adafactor
    

Example:

```
Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
```

Others reported the following combination to work well:

```
Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
```

When using `lr=None` with [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) you will most likely need to use `AdafactorSchedule`

scheduler as following:

```
from transformers.optimization import Adafactor, AdafactorSchedule

optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
lr_scheduler = AdafactorSchedule(optimizer)
trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
```

Usage:

```
optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False,
)
```

#### step

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L638)

( closure = None )

Parameters

-   **closure** (callable, optional) — A closure that reevaluates the model and returns the loss.

Performs a single optimization step

## AdamWeightDecay (TensorFlow)

### class transformers.AdamWeightDecay

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization_tf.py#L172)

( learning\_rate: typing.Union\[float, keras.src.optimizers.schedules.learning\_rate\_schedule.LearningRateSchedule\] = 0.001beta\_1: float = 0.9beta\_2: float = 0.999epsilon: float = 1e-07amsgrad: bool = Falseweight\_decay\_rate: float = 0.0include\_in\_weight\_decay: typing.Optional\[typing.List\[str\]\] = Noneexclude\_from\_weight\_decay: typing.Optional\[typing.List\[str\]\] = Nonename: str = 'AdamWeightDecay'\*\*kwargs )

Adam enables L2 weight decay and clip\_by\_global\_norm on gradients. Just adding the square of the weights to the loss function is _not_ the correct way of using L2 regularization/weight decay with Adam, since that will interact with the m and v parameters in strange ways as shown in [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).

Instead we want to decay the weights in a manner that doesn’t interact with the m/v parameters. This is equivalent to adding the square of the weights to the loss with plain (non-momentum) SGD.

Creates an optimizer from its config with WarmUp custom object.

#### transformers.create\_optimizer

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization_tf.py#L88)

( init\_lr: floatnum\_train\_steps: intnum\_warmup\_steps: intmin\_lr\_ratio: float = 0.0adam\_beta1: float = 0.9adam\_beta2: float = 0.999adam\_epsilon: float = 1e-08adam\_clipnorm: typing.Optional\[float\] = Noneadam\_global\_clipnorm: typing.Optional\[float\] = Noneweight\_decay\_rate: float = 0.0power: float = 1.0include\_in\_weight\_decay: typing.Optional\[typing.List\[str\]\] = None )

Creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.

## Schedules

### Learning Rate Schedules (Pytorch)

### class transformers.SchedulerType

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer_utils.py#L368)

( valuenames = Nonemodule = Nonequalname = Nonetype = Nonestart = 1 )

An enumeration.

#### transformers.get\_scheduler

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L335)

( name: typing.Union\[str, transformers.trainer\_utils.SchedulerType\]optimizer: Optimizernum\_warmup\_steps: typing.Optional\[int\] = Nonenum\_training\_steps: typing.Optional\[int\] = None )

Parameters

-   **name** (`str` or `SchedulerType`) — The name of the scheduler to use.
-   **optimizer** (`torch.optim.Optimizer`) — The optimizer that will be used during training.
-   **num\_warmup\_steps** (`int`, _optional_) — The number of warmup steps to do. This is not required by all schedulers (hence the argument being optional), the function will raise an error if it’s unset and the scheduler type requires it.
-   **num\_training\_steps** (\`int“, _optional_) — The number of training steps to do. This is not required by all schedulers (hence the argument being optional), the function will raise an error if it’s unset and the scheduler type requires it.

Unified API to get any scheduler from its name.

#### transformers.get\_constant\_schedule

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L39)

( optimizer: Optimizerlast\_epoch: int = -1 )

Parameters

-   **optimizer** (`~torch.optim.Optimizer`) — The optimizer for which to schedule the learning rate.
-   **last\_epoch** (`int`, _optional_, defaults to -1) — The index of the last epoch when resuming training.

Create a schedule with a constant learning rate, using the learning rate set in optimizer.

#### transformers.get\_constant\_schedule\_with\_warmup

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L77)

( optimizer: Optimizernum\_warmup\_steps: intlast\_epoch: int = -1 )

Parameters

-   **optimizer** (`~torch.optim.Optimizer`) — The optimizer for which to schedule the learning rate.
-   **num\_warmup\_steps** (`int`) — The number of steps for the warmup phase.
-   **last\_epoch** (`int`, _optional_, defaults to -1) — The index of the last epoch when resuming training.

Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate increases linearly between 0 and the initial lr set in the optimizer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_constant_schedule.png)

#### transformers.get\_cosine\_schedule\_with\_warmup

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L140)

( optimizer: Optimizernum\_warmup\_steps: intnum\_training\_steps: intnum\_cycles: float = 0.5last\_epoch: int = -1 )

Parameters

-   **optimizer** (`~torch.optim.Optimizer`) — The optimizer for which to schedule the learning rate.
-   **num\_warmup\_steps** (`int`) — The number of steps for the warmup phase.
-   **num\_training\_steps** (`int`) — The total number of training steps.
-   **num\_cycles** (`float`, _optional_, defaults to 0.5) — The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0 following a half-cosine).
-   **last\_epoch** (`int`, _optional_, defaults to -1) — The index of the last epoch when resuming training.

Create a schedule with a learning rate that decreases following the values of the cosine function between the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the initial lr set in the optimizer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_schedule.png)

#### transformers.get\_cosine\_with\_hard\_restarts\_schedule\_with\_warmup

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L185)

( optimizer: Optimizernum\_warmup\_steps: intnum\_training\_steps: intnum\_cycles: int = 1last\_epoch: int = -1 )

Parameters

-   **optimizer** (`~torch.optim.Optimizer`) — The optimizer for which to schedule the learning rate.
-   **num\_warmup\_steps** (`int`) — The number of steps for the warmup phase.
-   **num\_training\_steps** (`int`) — The total number of training steps.
-   **num\_cycles** (`int`, _optional_, defaults to 1) — The number of hard restarts to use.
-   **last\_epoch** (`int`, _optional_, defaults to -1) — The index of the last epoch when resuming training.

Create a schedule with a learning rate that decreases following the values of the cosine function between the initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases linearly between 0 and the initial lr set in the optimizer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_hard_restarts_schedule.png)

#### transformers.get\_linear\_schedule\_with\_warmup

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L104)

( optimizernum\_warmup\_stepsnum\_training\_stepslast\_epoch = -1 )

Parameters

-   **optimizer** (`~torch.optim.Optimizer`) — The optimizer for which to schedule the learning rate.
-   **num\_warmup\_steps** (`int`) — The number of steps for the warmup phase.
-   **num\_training\_steps** (`int`) — The total number of training steps.
-   **last\_epoch** (`int`, _optional_, defaults to -1) — The index of the last epoch when resuming training.

Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_linear_schedule.png)

#### transformers.get\_polynomial\_decay\_schedule\_with\_warmup

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L239)

( optimizernum\_warmup\_stepsnum\_training\_stepslr\_end = 1e-07power = 1.0last\_epoch = -1 )

Parameters

-   **optimizer** (`~torch.optim.Optimizer`) — The optimizer for which to schedule the learning rate.
-   **num\_warmup\_steps** (`int`) — The number of steps for the warmup phase.
-   **num\_training\_steps** (`int`) — The total number of training steps.
-   **lr\_end** (`float`, _optional_, defaults to 1e-7) — The end LR.
-   **power** (`float`, _optional_, defaults to 1.0) — Power factor.
-   **last\_epoch** (`int`, _optional_, defaults to -1) — The index of the last epoch when resuming training.

Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the optimizer to end lr defined by _lr\_end_, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

Note: _power_ defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT implementation at [https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37](https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37)

#### transformers.get\_inverse\_sqrt\_schedule

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization.py#L293)

( optimizer: Optimizernum\_warmup\_steps: inttimescale: int = Nonelast\_epoch: int = -1 )

Parameters

-   **optimizer** (`~torch.optim.Optimizer`) — The optimizer for which to schedule the learning rate.
-   **num\_warmup\_steps** (`int`) — The number of steps for the warmup phase.
-   **timescale** (`int`, _optional_, defaults to `num_warmup_steps`) — Time scale.
-   **last\_epoch** (`int`, _optional_, defaults to -1) — The index of the last epoch when resuming training.

Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

### Warmup (TensorFlow)

### class transformers.WarmUp

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization_tf.py#L30)

( initial\_learning\_rate: floatdecay\_schedule\_fn: typing.Callablewarmup\_steps: intpower: float = 1.0name: str = None )

Parameters

-   **initial\_learning\_rate** (`float`) — The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end of the warmup).
-   **decay\_schedule\_fn** (`Callable`) — The schedule function to apply after the warmup for the rest of training.
-   **warmup\_steps** (`int`) — The number of steps for the warmup part of training.
-   **power** (`float`, _optional_, defaults to 1) — The power to use for the polynomial warmup (defaults is a linear warmup).
-   **name** (`str`, _optional_) — Optional name prefix for the returned tensors during the schedule.

Applies a warmup schedule on a given learning rate decay schedule.

## Gradient Strategies

### GradientAccumulator (TensorFlow)

### class transformers.GradientAccumulator

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/optimization_tf.py#L302)

( )

Gradient accumulation utility. When used with a distribution strategy, the accumulator should be called in a replica context. Gradients will be accumulated locally on each replica and without synchronization. Users should then call `.gradients`, scale the gradients if required, and pass the result to `apply_gradients`.

Resets the accumulated gradients on the current replica.