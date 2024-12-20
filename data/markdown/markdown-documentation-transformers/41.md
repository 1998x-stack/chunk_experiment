# Efficient Inference on a Multiple GPUs

This document contains information on how to efficiently infer on a multiple GPUs.

Note: A multi GPU setup can use the majority of the strategies described in the [single GPU section](./perf_infer_gpu_one). You must be aware of simple techniques, though, that can be used for a better usage.

## Flash Attention 2

Flash Attention 2 integration also works in a multi-GPU setup, check out the appropriate section in the [single GPU section](./perf_infer_gpu_one#Flash-Attention-2)

## BetterTransformer

[BetterTransformer](https://huggingface.co/docs/optimum/bettertransformer/overview) converts 🤗 Transformers models to use the PyTorch-native fastpath execution, which calls optimized kernels like Flash Attention under the hood.

BetterTransformer is also supported for faster inference on single and multi-GPU for text, image, and audio models.

Flash Attention can only be used for models using fp16 or bf16 dtype. Make sure to cast your model to the appropriate dtype before using BetterTransformer.

### Decoder models

For text models, especially decoder-based models (GPT, T5, Llama, etc.), the BetterTransformer API converts all attention operations to use the [`torch.nn.functional.scaled_dot_product_attention` operator](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) (SDPA) that is only available in PyTorch 2.0 and onwards.

To convert a model to BetterTransformer:

```
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

model.to_bettertransformer()


```

SDPA can also call [Flash Attention](https://arxiv.org/abs/2205.14135) kernels under the hood. To enable Flash Attention or to check that it is available in a given setting (hardware, problem size), use [`torch.backends.cuda.sdp_kernel`](https://pytorch.org/docs/master/backends.html#torch.backends.cuda.sdp_kernel) as a context manager:

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to("cuda")
# convert the model to BetterTransformer
model.to_bettertransformer()

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

+ with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

If you see a bug with a traceback saying

```
RuntimeError: No available kernel.  Aborting execution.
```

try using the PyTorch nightly version, which may have a broader coverage for Flash Attention:

```
pip3 install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

Have a look at this [blog post](https://pytorch.org/blog/out-of-the-box-acceleration/) to learn more about what is possible with the BetterTransformer + SDPA API.

### Encoder models

For encoder models during inference, BetterTransformer dispatches the forward call of encoder layers to an equivalent of [`torch.nn.TransformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) that will execute the fastpath implementation of the encoder layers.

Because `torch.nn.TransformerEncoderLayer` fastpath does not support training, it is dispatched to `torch.nn.functional.scaled_dot_product_attention` instead, which does not leverage nested tensors but can use Flash Attention or Memory-Efficient Attention fused kernels.

More details about BetterTransformer performance can be found in this [blog post](https://medium.com/pytorch/bettertransformer-out-of-the-box-performance-for-huggingface-transformers-3fbe27d50ab2), and you can learn more about BetterTransformer for encoder models in this [blog](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/).

## Advanced usage: mixing FP4 (or Int8) and BetterTransformer

You can combine the different methods described above to get the best performance for your model. For example, you can use BetterTransformer with FP4 mixed-precision inference + flash attention:

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", quantization_config=quantization_config)

input_text = "Hello my dog is cute and"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```