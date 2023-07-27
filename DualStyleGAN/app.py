import os
from argparse import Namespace

import cv2
import numpy as np
import torch
import torchvision
from fastapi import FastAPI
from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

app = FastAPI()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
generator.eval()
ckpt = torch.load(
    os.path.join(
        "/opt/ml/level3_cv_finalproject-cv-15/DualStyleGAN/checkpoint/",
        "cartoon",
        "generator.pt",
    ),
    map_location=lambda storage, loc: storage,
)
generator.load_state_dict(ckpt["g_ema"])
generator = generator.to(device)

model_path = os.path.join(
    "/opt/ml/level3_cv_finalproject-cv-15/DualStyleGAN/checkpoint/", "encoder.pt"
)
ckpt = torch.load(model_path, map_location="cpu")
opts = ckpt["opts"]
opts["checkpoint_path"] = model_path
if "output_size" not in opts:
    opts["output_size"] = 1024
opts = Namespace(**opts)
opts.device = device
encoder = pSp(opts)
encoder.eval()
encoder.to(device)

exstyles = np.load(
    os.path.join(
        "/opt/ml/level3_cv_finalproject-cv-15/DualStyleGAN/checkpoint/",
        "cartoon",
        "exstyle_code.npy",
    ),
    allow_pickle="TRUE",
).item()


def load_image(filename):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    img = Image.open(filename)
    img = transform(img)
    return img.unsqueeze(dim=0)


def save_image(img, filename):
    tmp = ((img.detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))


@app.get("/gan_predict/")
async def generate_image(image_path: str):
    z_plus_latent = True
    return_z_plus_latent = True
    input_is_latent = False

    with torch.no_grad():
        viz = []
        # load content image
        image = load_image("../temp/" + image_path + ".jpg").to(device)
        viz += [image]

        # reconstructed content image and its intrinsic style code
        img_rec, instyle = encoder(
            F.adaptive_avg_pool2d(image, 256),
            randomize_noise=False,
            return_latents=True,
            z_plus_latent=z_plus_latent,
            return_z_plus_latent=return_z_plus_latent,
            resize=False,
        )
        img_rec = torch.clamp(img_rec.detach(), -1, 1)

        for style_id in [64, 87, 88, 107, 221, 268, 282, 299]:
            stylename = list(exstyles.keys())[style_id]
            latent = torch.tensor(exstyles[stylename]).to(device)

            # if False and not False:
            if False and not False:
                latent[:, 7:18] = instyle[:, 7:18]
            # extrinsic styte code
            exstyle = generator.generator.style(
                latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])
            ).reshape(latent.shape)
            if False and False:
                exstyle[:, 7:18] = instyle[:, 7:18]

            # style transfer
            # input_is_latent: instyle is not in W space
            # z_plus_latent: instyle is in Z+ space
            # use_res: use extrinsic style path, or the style is not transferred
            # interp_weights: weight vector for style combination of two paths
            img_gen, _ = generator(
                [instyle],
                exstyle,
                input_is_latent=input_is_latent,
                z_plus_latent=z_plus_latent,
                truncation=0.75,
                truncation_latent=0,
                use_res=True,
                interp_weights=[0.75] * 7 + [1] * 11,
            )
            img_gen = torch.clamp(img_gen.detach(), -1, 1)
            viz += [img_gen]

        print("Generate images successfully!")

        save_image(
            torchvision.utils.make_grid(
                F.adaptive_avg_pool2d(torch.cat(viz[1:], dim=0), 256), 4, 2
            ).cpu(),
            "../temp/" + image_path + "_transfer.jpg",
        )

        print("Save images successfully!")
        return image_path
