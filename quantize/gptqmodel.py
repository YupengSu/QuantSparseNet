import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig

# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(tokenizer, nsamples, seqlen):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(
        lambda x: len(x["text"]) >= seqlen)

    return [tokenizer(example["text"]) for example in traindata.select(range(nsamples))]


def gptqmodel(model, quantize_model, wbits, group_size=128, desc_act=False, sym=True, use_triton=False, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    traindataset = get_wikitext2(tokenizer, nsamples=256, seqlen=1024)

    quantize_config = QuantizeConfig(
        bits=wbits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
    )

    if os.path.exists(quantize_model):
        model = GPTQModel.load(quantize_model).to(device)
    else:
        model = GPTQModel.load(model, quantize_config).to(device)
        model.quantize(traindataset, tokenizer=tokenizer, backend=3)
        model.save(quantize_model)
        tokenizer.save_pretrained(quantize_model)

    return model, tokenizer