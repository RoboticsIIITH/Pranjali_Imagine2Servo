from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import argparse
import os
import transformers
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Instructpix2pix test script options")
    parser.add_argument("--file_path", type=str, default="./test",
                        help="Path to the root data directory")
    parser.add_argument("--save_path", type=str, default="./test_door_scene/",
                        help="Path to saved models")
    parser.add_argument(
        "--load_weights_folder",
        type=str,
        default="./instruct-pix2pix-model/",
        help="Path to a pretrained model used for initialization")
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    return parser.parse_args()

def mse_(image_1, image_2):
	imageA=np.asarray(image_1)
	imageB=np.asarray(image_2)  
	err = np.sum((imageA.astype("float") - imageB.astype("float"))**2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

def main():
    opt = get_args()
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    model_id = opt.load_weights_folder  
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, use_safetensors=True, safety_checker = None,
    ).to("cuda")
    generator = torch.Generator("cuda").manual_seed(0)

    prompts = ["Go to the door"]

    num_inference_steps = opt.num_inference_steps
    image_guidance_scale = opt.image_guidance_scale
    guidance_scale = opt.guidance_scale

    err = 1e5

    url = os.path.join(opt.file_path)

    init_image = Image.open(url).convert("RGB")

    edited_image = pipe(
        prompts[0],
        image=init_image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    img_save = url.replace(".png", "_next.png")
    edited_image.save(img_save)

    gen_image = Image.open(img_save).convert("RGB")
    err = mse_(init_image, gen_image)
    if err < 200:
        print("Minimum error reached!")

if __name__ == "__main__":
    main()
