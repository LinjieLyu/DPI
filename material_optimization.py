import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')

from mitsuba.scalar_rgb import Transform4f as T

import torch
import torch.nn as nn
import os 
import random
import argparse
import yaml
from util.img_utils import imread,imwrite
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


parser = argparse.ArgumentParser()
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
task_config = load_yaml(args.task_config)
operator_config = task_config['measurement']['operator']


# Load scene
scene_name=operator_config['scene_name']
image_path=operator_config['image_path']
scene=mi.load_file(operator_config['scene_path'])
cam_scene=mi.load_file(operator_config['camera_path'])
params = mi.traverse(scene)
cam_params = mi.traverse(cam_scene)
texture_res=operator_config['texture_res']
n_images=operator_config['n_images']


rgb_images = []                                
for i in range(n_images):
    if operator_config['ldr']:  
        rgb_img = imread('{}/{}.png'.format(image_path,i),gamma=1)[:,:,:3]
    else:
        rgb_img = imread('{}/{}.exr'.format(image_path,i))[:,:,:3]
    rgb_images.append(rgb_img.to(device))

gamma=operator_config['illumi_gamma']
scale=operator_config['illumi_scale']
normal=operator_config['illumi_normalize']



# Working directory
out_path = os.path.join(args.save_dir, operator_config['name'])
os.makedirs(out_path, exist_ok=True)

for img_dir in ['envmap','recon', 'roughness', 'metallic','basecolor']:
    os.makedirs(os.path.join(out_path,'material_refined' ,img_dir), exist_ok=True)
    

# rendering_function
@dr.wrap_ad(source='torch', target='drjit')
def render_texture(envmap,basecolor,roughness,metallic, spp=4, cam=0,seed=0): 
    if cam==0:
        params['PerspectiveCamera.to_world']=cam_params['PerspectiveCamera.to_world']
    else:
        params['PerspectiveCamera.to_world']=cam_params['PerspectiveCamera_{}.to_world'.format(cam)]
        
    params['EnvironmentMapEmitter.data']=envmap
    params['OBJMesh.bsdf.base_color.data']=basecolor
    params['OBJMesh.bsdf.roughness.data']=roughness
    params['OBJMesh.bsdf.metallic.data']=metallic

    params.update()

    rendered_gt=rgb_images[cam].cuda()
    rendered_opt=mi.render(scene, params, spp=spp)

    return rendered_gt,rendered_opt

# optimization

DIR = os.path.join(out_path, 'recon')
n_samples=len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
logger.info(f"total sample number is {n_samples}.")

for sample in range(n_samples):
    logger.info(f"optimize sample {sample}.")
    envmap = torch.ones([256,257,3],device=device) 
    envmap[:,:256,:3]=imread(os.path.join(out_path, 'recon', str(sample).zfill(5) + '.png'),gamma=1)
    envmap[:,:256,:3]=scale*torch.pow(envmap[:,:256,:3]/normal,gamma)
    envmap[:,256,:3]=envmap[:,255,:3]

    envmap_scale = torch.ones([1,1,1],device=device) 
    envmap_scale.requires_grad=True

    
    basecolor=imread(os.path.join(out_path, 'material','basecolor', str(sample).zfill(5) + '.exr'),gamma=1).cuda()
    metallic=imread(os.path.join(out_path, 'material','metallic', str(sample).zfill(5) + '.exr'),gamma=1).view(256,256,1).cuda()
    roughness=imread(os.path.join(out_path, 'material','roughness', str(sample).zfill(5) + '.exr'),gamma=1).view(256,256,1).cuda()
    
        
    
    basecolor.requires_grad=True
    metallic.requires_grad=True
    roughness.requires_grad=True
    
    optimizer = torch.optim.Adam([{'params': [envmap_scale]},
                                    {'params': [basecolor], 'lr': 1e-2},
                                    {'params': [metallic,roughness], 'lr': 1e-2},
                                  
                ], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)
    
    
    loss_fn = nn.MSELoss() #nn.L1Loss()
    
    # Optimization hyper-parameters
    iteration_count = 201
    spp = 35
    
    for i in range(iteration_count):
        optimizer.zero_grad()
        roughness3=torch.cat([roughness,roughness,roughness],dim=-1)
        metallic3=torch.cat([metallic,metallic,metallic],dim=-1)
        
        
        target_img,rendered_img = render_texture(envmap*envmap_scale,basecolor,roughness,metallic, spp=spp, cam=random.randint(0,n_images-1),seed=i)
         
        
        loss = loss_fn(rendered_img, target_img)
                
            
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            
            loss_accum = loss.item()
            optimizer.step()
            scheduler.step()
  
        basecolor.data=basecolor.data.clamp(1e-8,1-1e-8)
        metallic.data=torch.nan_to_num(metallic.data.clamp(1e-8,1-1e-8),nan=0.0)
        roughness.data=torch.nan_to_num(roughness.data.clamp(1e-8,1-1e-8),nan=1.0)

        rendered_img=rendered_img.detach()
        
        
        print(f'Training iteration {i}/{iteration_count}, loss: {loss_accum}', end='\r')
  
            
        if i%100==0:

            mi.util.write_bitmap(os.path.join(out_path, 'material_refined','recon', str(sample).zfill(5) + '.png'), torch.cat((rendered_img,target_img),1))           
            mi.util.write_bitmap(os.path.join(out_path, 'material_refined','envmap', str(sample).zfill(5) + '.exr'),envmap_scale*envmap[:,:,:3]) 
            mi.util.write_bitmap(os.path.join(out_path, 'material_refined','roughness', str(sample).zfill(5) + '.exr'), roughness3[:,:,:3])
            mi.util.write_bitmap(os.path.join(out_path, 'material_refined','basecolor', str(sample).zfill(5) + '.exr'), basecolor[:,:,:3])
            mi.util.write_bitmap(os.path.join(out_path, 'material_refined','metallic', str(sample).zfill(5) + '.exr'), metallic3[:,:,:3])
           
            
        del(loss,rendered_img)
         
