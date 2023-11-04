import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator,to_output,imwrite
from util.logger import get_logger
import numpy as np
from torchvision.transforms.functional import adjust_gamma


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label','recon_measurement','material']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
    
    
    
         
    # Do Inference
    for i, ref_img in enumerate(loader):
        
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
     
        ########################## Prepare###################################
        
        
        
        # Exception) In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        elif 'raytracing' in measure_config['operator'] ['name'] :
              
              y_n = operator.forward_gt(spp=512)
        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n =noiser(y)
        
        plt.imsave(os.path.join(out_path, 'input', fname), to_output(y_n)) #to_output(y_n)
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        
        
        
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
        
        if cond_config['method']== 'rt-ps+':
            _,recon_m=operator.forward_multiview(sample,spp=512)
            recon_m=recon_m.detach()
        elif cond_config['method']== 'ps+':
            recon_m=operator.forward(sample,spp=512).detach()
            
        plt.imsave(os.path.join(out_path, 'recon', fname), to_output((sample+1.)/2))
        plt.imsave(os.path.join(out_path, 'recon_measurement', fname), to_output(recon_m.clamp(0.,1.)))#to_output(operator.forward(sample))
        
        torch.save({
            'basecolor': operator.basecolor,
            'metallic': operator.metallic,
          'roughness': operator.roughness,
            }, os.path.join(out_path, 'material', str(i).zfill(5) + '.pt'))
        
        imwrite(operator.basecolor.detach().cpu(),os.path.join(out_path, 'material','basecolor', str(i).zfill(5) + '.exr'),gamma=1)
        imwrite(operator.metallic.detach().cpu(),os.path.join(out_path, 'material','metallic', str(i).zfill(5) + '.exr'),gamma=1)
        imwrite(operator.roughness.detach().cpu(),os.path.join(out_path, 'material','roughness', str(i).zfill(5) + '.exr'),gamma=1)
        cond_method.operator.reset_optimizer()
if __name__ == '__main__':
    main()
