import os
import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    from datasets import load_dataset
    import numpy as np

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc

def autogptq(model, quantize_model, wbits, group_size=128, desc_act=False, sym=True, use_triton=False, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindataset, testenc = get_wikitext2(128, 0, 2048, tokenizer)

    quantize_config = BaseQuantizeConfig(
        bits=wbits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
    )

    if os.path.exists(quantize_model):
        model = AutoGPTQForCausalLM.from_quantized(quantize_model, use_triton=use_triton).to(device)
    else:
        model = AutoGPTQForCausalLM.from_pretrained(model, quantize_config).to(device)
        model.quantize(traindataset, use_triton=use_triton)
        model.save_quantized(quantize_model, use_safetensors=True)
        tokenizer.save_pretrained(quantize_model)

    return model, tokenizer
