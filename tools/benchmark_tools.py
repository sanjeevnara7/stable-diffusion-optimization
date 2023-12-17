'''
Benchmarking utilities.
'''
import os
import yaml

import torch
from torch import autocast
from torchmetrics.functional.multimodal import clip_score
from functools import partial

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

#Decorator function to time events
def benchmark_timer(n_reps):
    def _decorator(func):
        def inner(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            res = func(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            benchmark_res = ((start.elapsed_time(end))/n_reps)/1000
            print(f'[*] Number of Reps: {n_reps}, Elapsed time (per rep): {benchmark_res} s')
            return res, benchmark_res
        return inner
    return _decorator

#Utility function to calculate CLIP score
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

#Utility function to generate images using a StableDiffusionPipeline
@benchmark_timer(len(cfg['prompts']))
@torch.inference_mode()
def generate_images(sdPipeline, device, rng):
    gen_imgs = []

    if cfg['autocast']:
        with autocast(device_type=device):
            for prompt in cfg['prompts']:
                images = sdPipeline(
                        prompt=prompt,
                        generator=rng,
                        **cfg['inference_config']
                    ).images
                gen_imgs.extend(images)
    else:
        for prompt in cfg['prompts']:
            images = sdPipeline(
                    prompt=prompt,
                    generator=rng,
                    **cfg['inference_config']
                ).images
            gen_imgs.extend(images)
    return gen_imgs

#Utility function to save images
def save_images(images):
    os.makedirs(cfg['run_name'], exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(cfg['run_name'], f'{i}.jpg'))

