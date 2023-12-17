# A Study of Stable Diffusion Inference Optimization techniques
<img src="https://img.shields.io/badge/python-3.10-green" /> <img src="https://img.shields.io/badge/torch-2.1-orange" /> <img src="https://img.shields.io/badge/diffusers-0.24-yellow" /><br>
Comparison / Benchmarks of different Stable Diffusion (SD) optimization techniques on [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5). We explore methods to improve the inference time of SD while maintaining image quality (measure by CLIP score). The benchmark will generate images with respect to a fixed set of prompts, all at 512x512 resolution with 30 inference steps.

<p align="center">
    <img src="https://github.com/sanjeevnara7/stable-diffusion-optimization/blob/main/figures/img_comp.png" width="80%">
    <div align="center">
      <figcaption>Figures - (Top) Comparison of generated images with optimizations incrementally applied (right to left). (Bottom) List of optimizations applied and inference time/speedup.</figcaption>
    </div>
</p>
<p align="center">
    <img src="https://github.com/sanjeevnara7/stable-diffusion-optimization/blob/main/figures/opt_checklist.png" width="72%">
    <img src="https://github.com/sanjeevnara7/stable-diffusion-optimization/blob/main/figures/inference_times.png" width="27%">
</p>

## Requirements
- ```PyTorch >= 2.0```
- ```Diffusers```
- ```Transformers```
- ```TomeSD (Token Merging)```
- ```onnx, onnx-runtime```
- ```wandb (logging)```
- ```PyYAML```

Refer to requirements.txt for additional details.

## Organization
- `config.yaml` contains the optimizations and inference configuration.
- `tools/` folder contains the benchmark script and other utility files:
  - `tools/benchmark.py` - main benchmark script used for measuring inference time of image generation based on the config file.
  - `tools/benchmark_tools` - contains utility methods for benchmarking.
  - `tools/sdPipelineBuilder` - builds the StableDiffusionPipeline based on the config.
  - `tools/onnx_export` - ONNX export script for StableDiffusionPipeline.
- `Benchmark_Visualization.ipynb` - notebook that demonstrates before vs after applying optimizations to SD inference.

## Usage

The benchmark code is designed to use a modular ```config.yaml``` file with the following prompts and inference config:
```
#Prompt config
prompts:
  - a photo of an astronaut riding a horse on mars
  - A high tech solarpunk utopia in the Amazon rainforest
  - A pikachu fine dining with a view to the Eiffel Tower
  - A mecha robot in a favela in expressionist style
  - an insect robot preparing a delicious meal
  - A small cabin on top of a snowy mountain in the style of Disney, artstation

#Inference config for image generation
inference_config:
  height: 512
  width: 512
  negative_prompt: ugly, deformed
  num_images_per_prompt: 1
  num_inference_steps: 30
  guidance_scale: 7.5
```

1. ```git clone https://github.com/sanjeevnara7/stable-diffusion-optimization.git``` to clone the repo.
2. ```pip install -r requirements.txt``` to install necessary requirements.
3. [Important] Modify the ```config.yaml``` file to specify which optimizations should be enabled/disabled for the benchmark.
4. (Optional) Login to wandb ```wandb login``` to enable wandb logging.
5. Run ```tools/benchmark.py``` to run the inference benchmark with the specified optimizations in the config file. By default, the script will try to use the GPU.

## References
[1] Token Merging for Stable Diffusion (ToMESD). https://github.com/dbolya/tomesd

[2] Segmind-Distill-SD Knowledge-distilled, smaller versions of Stable Diffusion. Unofficial implementation as described in BK-SDM. https://github.com/segmind/distill-sd

[3] Tiny AutoEncoder for Stable Diffusion, TAESD. https://github.com/madebyollin/taesd

[4] Speed up Stable Diffusion Inference, HuggingFace. https://huggingface.co/docs/diffusers/optimization/fp16
