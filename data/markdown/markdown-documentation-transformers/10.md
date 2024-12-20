# Load adapters with 🤗 PEFT

[Parameter-Efficient Fine Tuning (PEFT)](https://huggingface.co/blog/peft) methods freeze the pretrained model parameters during fine-tuning and add a small number of trainable parameters (the adapters) on top of it. The adapters are trained to learn task-specific information. This approach has been shown to be very memory-efficient with lower compute usage while producing results comparable to a fully fine-tuned model.

Adapters trained with PEFT are also usually an order of magnitude smaller than the full model, making it convenient to share, store, and load them.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png)

The adapter weights for a OPTForCausalLM model stored on the Hub are only ~6MB compared to the full size of the model weights, which can be ~700MB.

If you’re interested in learning more about the 🤗 PEFT library, check out the [documentation](https://huggingface.co/docs/peft/index).

## Setup

Get started by installing 🤗 PEFT:

If you want to try out the brand new features, you might be interested in installing the library from source:

```
pip install git+https://github.com/huggingface/peft.git
```

## Supported PEFT models

🤗 Transformers natively supports some PEFT methods, meaning you can load adapter weights stored locally or on the Hub and easily run or train them with a few lines of code. The following methods are supported:

-   [Low Rank Adapters](https://huggingface.co/docs/peft/conceptual_guides/lora)
-   [IA3](https://huggingface.co/docs/peft/conceptual_guides/ia3)
-   [AdaLoRA](https://arxiv.org/abs/2303.10512)

If you want to use other PEFT methods, such as prompt learning or prompt tuning, or about the 🤗 PEFT library in general, please refer to the [documentation](https://huggingface.co/docs/peft/index).

## Load a PEFT adapter

To load and use a PEFT adapter model from 🤗 Transformers, make sure the Hub repository or local directory contains an `adapter_config.json` file and the adapter weights, as shown in the example image above. Then you can load the PEFT adapter model using the `AutoModelFor` class. For example, to load a PEFT adapter model for causal language modeling:

1.  specify the PEFT model id
2.  pass it to the [AutoModelForCausalLM](/docs/transformers/v4.34.0/en/model_doc/auto#transformers.AutoModelForCausalLM) class

```
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
```

You can load a PEFT adapter with either an `AutoModelFor` class or the base model class like `OPTForCausalLM` or `LlamaForCausalLM`.

You can also load a PEFT adapter by calling the `load_adapter` method:

```
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "facebook/opt-350m"
peft_model_id = "ybelkada/opt-350m-lora"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)
```

## Load in 8bit or 4bit

The `bitsandbytes` integration supports 8bit and 4bit precision data types, which are useful for loading large models because it saves memory (see the `bitsandbytes` integration [guide](./quantization#bitsandbytes-integration) to learn more). Add the `load_in_8bit` or `load_in_4bit` parameters to [from\_pretrained()](/docs/transformers/v4.34.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) and set `device_map="auto"` to effectively distribute the model to your hardware:

```
from transformers import AutoModelForCausalLM, AutoTokenizer

peft_model_id = "ybelkada/opt-350m-lora"
model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", load_in_8bit=True)
```

## Add a new adapter

You can use `~peft.PeftModel.add_adapter` to add a new adapter to a model with an existing adapter as long as the new adapter is the same type as the current one. For example, if you have an existing LoRA adapter attached to a model:

```
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    init_lora_weights=False
)

model.add_adapter(lora_config, adapter_name="adapter_1")
```

To add a new adapter:

```
model.add_adapter(lora_config, adapter_name="adapter_2")
```

Now you can use `~peft.PeftModel.set_adapter` to set which adapter to use:

```
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))


model.set_adapter("adapter_2")
output_enabled = model.generate(**inputs)
print(tokenizer.decode(output_enabled[0], skip_special_tokens=True))
```

## Enable and disable adapters

Once you’ve added an adapter to a model, you can enable or disable the adapter module. To enable the adapter module:

```
from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig

model_id = "facebook/opt-350m"
adapter_model_id = "ybelkada/opt-350m-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(model_id)
peft_config = PeftConfig.from_pretrained(adapter_model_id)


peft_config.init_lora_weights = False

model.add_adapter(peft_config)
model.enable_adapters()
output = model.generate(**inputs)
```

To disable the adapter module:

```
model.disable_adapters()
output = model.generate(**inputs)
```

## Train a PEFT adapter

PEFT adapters are supported by the [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer) class so that you can train an adapter for your specific use case. It only requires adding a few more lines of code. For example, to train a LoRA adapter:

If you aren’t familiar with fine-tuning a model with [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer), take a look at the [Fine-tune a pretrained model](training) tutorial.

1.  Define your adapter configuration with the task type and hyperparameters (see `~peft.LoraConfig` for more details about what the hyperparameters do).

```
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
```

2.  Add adapter to the model.

```
model.add_adapter(peft_config)
```

3.  Now you can pass the model to [Trainer](/docs/transformers/v4.34.0/en/main_classes/trainer#transformers.Trainer)!

```
trainer = Trainer(model=model, ...)
trainer.train()
```

To save your trained adapter and load it back:

```
model.save_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)
```