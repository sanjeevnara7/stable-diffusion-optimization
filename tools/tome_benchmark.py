'''
Benchmark script for Stable Diffusion with Token Merging (ToME)
#Requires TomeSD #pip install tomesd
'''

import torch
from diffusers import StableDiffusionPipeline

try:
    import tomesd
except ImportError:
    print('Package tomesd not found.')
    exit(0)

from torchmetrics.functional.multimodal import clip_score
from functools import partial
from torch.profiler import profile, record_function, ProfilerActivity

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('*'*60)
print('[*] Device: ', device)

MODEL_CKPT = 'runwayml/stable-diffusion-v1-5'

TOKEN_MERGING_RATIO = 1.0

prompts = [
    "a photo of an astronaut riding a horse on mars",
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
]

inference_config = {
    "height": 512,
    "width": 512,
    "negative prompt": "ugly, deformed",
    "num_images_per_prompt": 1,
    "num_inference_steps": 30,
    "guidance_scale": 7.5
}

def getSDPipeline():
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_CKPT, torch_dtype=torch.float32)
    pipe.to(device)
    tomesd.apply_patch(pipe, ratio=TOKEN_MERGING_RATIO)
    return pipe

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

def benchmark():
    sdPipeline = getSDPipeline()
    
    rng = torch.Generator(device=device) #Generator for seed
    seed = 1337
    rng.manual_seed(seed)
    gen_imgs = []
    
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            for prompt in prompts:
                with record_function("model_inference"):
                    images = sdPipeline(
                        prompt=prompt,
                        generator=rng,
                        **inference_config
                    ).images
                gen_imgs.extend(images)
    print('[*] Generated images: ', len(gen_imgs))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    images_np = np.stack([np.array(img) for img in gen_imgs])
    sd_clip_score = calculate_clip_score(images_np, prompts)
    print(f"CLIP score: {sd_clip_score}")

if __name__ == "__main__":
    benchmark()