from pathlib import Path

import cv2
import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torchvision.transforms.functional import resize

import sys
sys.path.append(str(Path(__file__).resolve().parent / "../../third_party/SDS-Bridge/2D_experiments"))
from guidance import Guidance, GuidanceConfig
sys.path.pop()


class TranslationGuidance:
    """
    """
    def __init__(self, checkpoint: Path | str, scale=40):
        self.guidance = Guidance(GuidanceConfig(sd_pretrained_model_or_path=checkpoint))
        self.guidance_scale = scale

    @torch.no_grad()
    def encode(self, image: Float[torch.Tensor, "B H W 3"]):
        image = resize(image.permute(0, 3, 1, 2), (512, 512))
        image = image / 255.
        image = image.to(self.guidance.unet.device)
        return self.guidance.encode_image(image)

    @torch.no_grad()
    def decode(self, latent: Float[torch.Tensor, "B C latentH latentW"]):
        image = self.guidance.decode_latent(latent)
        image = image.permute(0, 2, 3, 1) * 255
        return image.to(torch.uint8)

    def step(
        self,
        latent: Float[torch.Tensor, "B C latentH latentW"],
        prompt: str,
        source_prompt: str,
        target_prompt: str
    ):
        return self.guidance.bridge_stage_two(
            latent, 
            prompt=prompt,
            extra_src_prompts=source_prompt,
            extra_tgt_prompts=target_prompt, 
            cfg_scale=self.guidance_scale,
            return_dict=True
        )


class Translation:
    """
    """
    def __init__(self, image: Float[np.ndarray, "H W 3"], guidance: TranslationGuidance, lr=1e-3):
        """
        """
        self.guidance = guidance
        
        self.latent = self.guidance.encode(torch.from_numpy(image)[None, ...])
        self.latent.requires_grad = True
        self.latent.retain_grad()
        self.latent_optim = torch.optim.AdamW([self.latent], lr=lr, betas=(0.9, 0.99), eps=1e-15)

    def step(
        self, 
        prompt: str,
        source_prompt: str,
        target_prompt: str,
    ):
        """
        """
        loss_dict = self.guidance.step(self.latent, prompt, source_prompt, target_prompt)
        self.latent.backward(gradient=loss_dict['grad'])
        self.latent_optim.step()
        self.latent_optim.zero_grad()

    def image(self, pil=False):
        image = self.guidance.decode(self.latent)[0].detach().cpu().numpy()
        if pil:
            return Image.fromarray(image)
        return image


class ImageAccumulator:
    """
    """
    def __init__(self, fps=30):
        self.fps = fps
        self.images = []

    def append(self, image: Float[np.ndarray, "H W 3"]):
        self.images.append(image)

    def save_video(self, path: Path | str):
        H, W, _ = self.images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, self.fps, (W, H))
        for image in self.images:
            out.write(image[:, :, ::-1]) # BGR to RGB
        out.release()


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    from tqdm import tqdm

    guidance = TranslationGuidance("stabilityai/stable-diffusion-2-1-base")

    name = "street"
    image = np.array(Image.open(f"assets/{name}.png").convert("RGB"))
    translation = Translation(image, guidance, lr=2e-3) # TODO tune lr

    config = OmegaConf.load(f"assets/{name}.yaml")
    accum = ImageAccumulator()
    for i in tqdm(range(1000)):
        translation.step(
            config["prompt"],
            config["source_prompt"],
            config["target_prompt1"] if (i % 4 == 0) else config["target_prompt2"], # TODO one domain may dominate the other => dynamic weighting adjustment using clip
        )
        accum.append(translation.image())
    os.makedirs("outputs", exist_ok=True)
    accum.save_video(f"outputs/{name}.mp4")