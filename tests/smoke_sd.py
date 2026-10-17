import torch
from diffusers import StableDiffusionPipeline
from src.device import get_device

device = get_device()
dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=dtype
).to(device)
pipe.enable_attention_slicing()

image = pipe(
    "a watercolor painting of a mountain village",
    num_inference_steps=20, height=512, width=512,
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("outputs/smoke_test.png")
print("Saved outputs/smoke_test.png")