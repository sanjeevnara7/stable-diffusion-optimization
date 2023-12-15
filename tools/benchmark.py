'''
Benchmark script for Stable Diffusion. Also logs benchmark results to wandb (requires login)
'''
from pprint import pprint
import numpy as np
import yaml

import wandb

import torch

from sdPipelineBuilder import getSDPipeline
from benchmark_tools import generate_images, save_images, calculate_clip_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('*'*60)
print('[*] Device: ', device)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

def benchmark():
    with wandb.init(entity='columbia-cs', 
                    project='stable-diffusion-optimization', 
                    name=cfg['run_name'],
                    config = cfg) as run:
        print('[*] Building StableDiffusionPipeline...')
        pipe = getSDPipeline()
        pipe.to(device)
        rng = torch.Generator(device=device) #Generator for seeding
        seed = 1337
        rng.manual_seed(seed)

        if cfg['jit_compile']:
            #Do one inference to build computation graph
            print('[*] JIT Compile, running inference to build graph...')
            with torch.inference_mode():
                _ = pipe(
                        prompt=cfg['prompts'][0],
                        generator=rng,
                        **cfg['inference_config']
                )

        print('[*] Running Benchmark...')
        gen_imgs, benchmark_result = generate_images(
            sdPipeline=pipe,
            device=device,
            rng=rng)

        images_np = np.stack([np.array(img) for img in gen_imgs])
        sd_clip_score = calculate_clip_score(images_np, cfg['prompts'])
        print(f"[*] CLIP score: {sd_clip_score}")
        # savepath = cfg['run_name']+'/'
        # print(f'[*] Saving images to {savepath}...')
        # save_images(gen_imgs)

        #Log to wandb
        run.log({"inference_time": benchmark_result})
        run.log({"CLIP_score": sd_clip_score})
        log_images = []
        for i, img in enumerate(gen_imgs):
            p = cfg['prompts'][i]
            image = wandb.Image(img, caption=f'Generated Image: {p}')
            log_images.append(image)
        run.log({"generated_images": log_images})

if __name__ == "__main__":
    print('[*] Specified Benchmark Config:')
    pprint(cfg)
    benchmark()