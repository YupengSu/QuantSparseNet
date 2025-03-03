import torch
import torch.nn as nn
import time

from quant import Quantizer
from lib.quant_lib.gptq import GPTQ
from lib.utils import find_layers,get_loaders 

def prepare_calibration_input(model, dataloader, nsamples, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'token_type_ids': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['token_type_ids'] = kwargs['token_type_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    token_type_ids = cache['token_type_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, token_type_ids


@torch.no_grad()
def quant_gptq(args, model, tokenizer, device=torch.device("cuda:0")):
    ## GPTQ code available at: https://github.com/IST-DASLab/gptq/blob/main/llama.py
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, token_type_ids = prepare_calibration_input(model, dataloader, args.nsamples, device)
    
    layers = model.model.layers
    start_time = time.time()
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = GPTQ(subset[name])
            wrapped_layers[name].quantizer = Quantizer()
            wrapped_layers[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False
            )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, token_type_ids=token_type_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"quantizing layer {i} name {name}")
            wrapped_layers[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
            )
            quantizers['model.layers.%d.%s' % (i, name)] = wrapped_layers[name].quantizer
            scale = wrapped_layers[name].quantizer.scale
            zero = wrapped_layers[name].quantizer.zero
            weight = layer[name].weight.data
            print(f"layer {name} scale: {scale.shape} zero: {zero.shape}")
            print(f"layer {name} weight: {weight.shape}")
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,  token_type_ids=token_type_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return quantizers

