from diffusers import StableDiffusionPipeline
import torch

MODEL_CKPT = 'runwayml/stable-diffusion-v1-5'

prompts = [
    "a photo of an astronaut riding a horse on mars",
    "A high tech solarpunk utopia in the Amazon rainforest",
    "A pikachu fine dining with a view to the Eiffel Tower",
    "A mecha robot in a favela in expressionist style",
    "an insect robot preparing a delicious meal",
    "A small cabin on top of a snowy mountain in the style of Disney, artstation",
]

pipe_config = {
    "height": 512,
    "width": 512,
    "negative prompt": "ugly, deformed",
    "num_images_per_prompt": 2,
    "num_inference_steps": 30,
    "guidance_scale": 7.5
}

def getSDPipeline():
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_CKPT, torch_dtype=torch.float32)
    return pipe

def main():
    sdPipeline = getSDPipeline()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sdPipeline.to(device)

    rng = torch.Generator(device=device) #Generator for seed
    seed = 1337
    rng.manual_seed(seed)

    with torch.inference_mode():
        images = sdPipeline(
            prompt=prompts,
            generator=rng,
            **pipe_config
        ).images
    
    print('Generated images: ', len(images))

if __name__ == "__main__":
    main()