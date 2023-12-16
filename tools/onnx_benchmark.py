'''
Script to Benchmark a ONNX StableDiffusionPipeline.
The pipeline must be converted to ONNX and saved using the onnx_export script.
'''
import yaml
import wandb
import numpy as np

import torch

from diffusers import OnnxStableDiffusionPipeline
from benchmark_tools import generate_images, calculate_clip_score

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
        print('[*] Building OnnxStableDiffusionPipeline...')

        executionProvider = 'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'

        pipe = OnnxStableDiffusionPipeline.from_pretrained('./onnx-model', provider=executionProvider)
        
        rng = r = np.random.RandomState(1337)

        print('[*] Running Benchmark...')
        gen_imgs, benchmark_result = generate_images(
            sdPipeline=pipe,
            device=device,
            rng=rng)

        images_np = np.stack([np.array(img) for img in gen_imgs])
        sd_clip_score = calculate_clip_score(images_np, cfg['prompts'])
        print(f"[*] CLIP score: {sd_clip_score}")

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
    print('[*] Running ONNX Pipeline Benchmark')
    benchmark()


