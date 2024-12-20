# Hyperparameter Search using Trainer API

🤗 Transformers provides a [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) provides API for hyperparameter search. This doc shows how to enable it in example.

## Hyperparameter Search backend

[Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) supports four hyperparameter search backends currently: [optuna](https://optuna.org/), [sigopt](https://sigopt.com/), [raytune](https://docs.ray.io/en/latest/tune/index.html) and [wandb](https://wandb.ai/site/sweeps).

you should install them before using them as the hyperparameter search backend

```
pip install optuna/sigopt/wandb/ray[tune] 
```

## How to enable Hyperparameter search in example

Define the hyperparameter search space, different backends need different format.

For sigopt, see sigopt [object\_parameter](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter), it’s like following:

```
>>> def sigopt_hp_space(trial):
...     return [
...         {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double"},
...         {
...             "categorical_values": ["16", "32", "64", "128"],
...             "name": "per_device_train_batch_size",
...             "type": "categorical",
...         },
...     ]
```

For optuna, see optuna [object\_parameter](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html#sphx-glr-tutorial-10-key-features-002-configurations-py), it’s like following:

```
>>> def optuna_hp_space(trial):
...     return {
...         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
...         "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
...     }
```

Optuna provides multi-objective HPO. You can pass `direction` in `hyperparameter_search` and define your own compute\_objective to return multiple objective values. The Pareto Front (`List[BestRun]`) will be returned in hyperparameter\_search, you should refer to the test case `TrainerHyperParameterMultiObjectOptunaIntegrationTest` in [test\_trainer](https://github.com/huggingface/transformers/blob/main/tests/trainer/test_trainer.py). It’s like following

```
>>> best_trials = trainer.hyperparameter_search(
...     direction=["minimize", "maximize"],
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

For raytune, see raytune [object\_parameter](https://docs.ray.io/en/latest/tune/api/search_space.html), it’s like following:

```
>>> def ray_hp_space(trial):
...     return {
...         "learning_rate": tune.loguniform(1e-6, 1e-4),
...         "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
...     }
```

For wandb, see wandb [object\_parameter](https://docs.wandb.ai/guides/sweeps/configuration), it’s like following:

```
>>> def wandb_hp_space(trial):
...     return {
...         "method": "random",
...         "metric": {"name": "objective", "goal": "minimize"},
...         "parameters": {
...             "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
...             "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
...         },
...     }
```

Define a `model_init` function and pass it to the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer), as an example:

```
>>> def model_init(trial):
...     return AutoModelForSequenceClassification.from_pretrained(
...         model_args.model_name_or_path,
...         from_tf=bool(".ckpt" in model_args.model_name_or_path),
...         config=config,
...         cache_dir=model_args.cache_dir,
...         revision=model_args.model_revision,
...         use_auth_token=True if model_args.use_auth_token else None,
...     )
```

Create a [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) with your `model_init` function, training arguments, training and test datasets, and evaluation function:

```
>>> trainer = Trainer(
...     model=None,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
...     tokenizer=tokenizer,
...     model_init=model_init,
...     data_collator=data_collator,
... )
```

Call hyperparameter search, get the best trial parameters, backend could be `"optuna"`/`"sigopt"`/`"wandb"`/`"ray"`. direction can be`"minimize"` or `"maximize"`, which indicates whether to optimize greater or lower objective.

You could define your own compute\_objective function, if not defined, the default compute\_objective will be called, and the sum of eval metric like f1 is returned as objective value.

```
>>> best_trial = trainer.hyperparameter_search(
...     direction="maximize",
...     backend="optuna",
...     hp_space=optuna_hp_space,
...     n_trials=20,
...     compute_objective=compute_objective,
... )
```

## Hyperparameter search For DDP finetune

Currently, Hyperparameter search for DDP is enabled for optuna and sigopt. Only the rank-zero process will generate the search trial and pass the argument to other ranks.