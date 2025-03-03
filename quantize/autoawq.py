import os
import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

def autoawq(model, quantize_model, wbits, group_size=128, zero_point=False, use_triton=False, device="cuda:0"):
    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": wbits,
        "version": "GEMM"
    }

    if os.path.exists(quantize_model):
        model = AutoAWQForCausalLM.from_quantized(quantize_model, use_triton=use_triton)
    else:
        model = AutoAWQForCausalLM.from_pretrained(model)
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(quantize_model, use_safetensors=True)
        tokenizer.save_pretrained(quantize_model)

    return model, tokenizer