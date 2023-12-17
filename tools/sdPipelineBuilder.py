'''
Build a stable diffusion pipeline for inference.
Uses the config.yaml file to set up the pipeline.
'''

import yaml

import torch
from diffusers import StableDiffusionPipeline, AutoencoderTiny
from diffusers.models.attention_processor import AttnProcessor2_0, XFormersAttnProcessor
import tomesd

#Get config
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

def getSDPipeline():
    '''
    Build a stable diffusion pipeline based on the config.yaml file.
    '''
    #Precision
    if cfg['precision'] == 'half':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(cfg['model_path'], torch_dtype=torch_dtype)
    
    #Attention Processors
    if cfg['attention'] == 'SDPA':
        pipe.unet.set_attn_processor(AttnProcessor2_0())
    elif cfg['attention'] == 'xFormers':
        pipe.unet.set_attn_processor(XFormersAttnProcessor())
    elif cfg['attention'] == 'vanilla':
        pipe.unet.set_default_attn_processor()
    elif cfg['attention'] == 'sliced':
        pipe.enable_attention_slicing() #should not be combined with the others
    
    #Token merging
    if cfg['token_merging']['use_tome']:
        tomesd.apply_patch(pipe, ratio=cfg['token_merging']['ratio'])

    #Tiny autoencoder
    if cfg['tiny_autoencoder']:
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch_dtype)

    #JIT
    if cfg['jit_compile']['use_jit']:
        if 'unet' in cfg['jit_compile']['components']:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        if 'vae' in cfg['jit_compile']['components']:
            pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
    
    return pipe
