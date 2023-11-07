'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''
import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
from mitsuba.scalar_rgb import Transform4f as T

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch


from util.resizer import Resizer
from util.img_utils import fft2_m,imread

import numpy as np
import random


import math

from  torch.cuda.amp import autocast

import os
import glob

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if 'raytracing' in name:
        name='raytracing' 
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data



class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 


@register_operator(name='raytracing')
class RaytracingOperator(NonLinearOperator):
    def __init__(self,scene_name,image_path,n_images,ldr,scene_path,camera_path,illumi_gamma,illumi_scale,illumi_normalize,texture_res,device):
        self.device = device
        self.ldr=ldr
            
        
        
        self.scene_name=scene_name
        self.image_path=image_path
        self.scene=mi.load_file(scene_path)
        self.cam_scene=mi.load_file(camera_path)
        self.params = mi.traverse(self.scene)
        self.cam_params = mi.traverse(self.cam_scene)
        
        self.texture_res=texture_res
        self.n_images=n_images
        self.rgb_images = []                                
        for i in range(self.n_images):
            if self.ldr:  
                rgb_img = imread('{}/{}.png'.format(self.image_path,i),gamma=1)[:,:,:3]
            else:
                rgb_img = imread('{}/{}.exr'.format(self.image_path,i))[:,:,:3]
        
            
            self.rgb_images.append(rgb_img.to(self.device))
        

 
        
        self.basecolor=0.01*torch.ones([self.texture_res,self.texture_res,3],device=torch.device('cuda')) 
        self.metallic=self.params['OBJMesh.bsdf.metallic.data'].torch().clone()[:,:,0:1]
        self.roughness=self.params['OBJMesh.bsdf.roughness.data'].torch().clone()[:,:,0:1]
        
        self.basecolor.requires_grad=True
        self.metallic.requires_grad=True
        self.roughness.requires_grad=True
        
        self.optimizer = torch.optim.Adam([
                                        {'params': [self.basecolor]},
                                        {'params': [self.roughness], 'lr': 1.5e-2},
                                        {'params': [self.metallic], 'lr': 1e-2},
                                      
                    ], lr=2e-2)#
        
        
        self.loss_fn = torch.nn.MSELoss()
        
        self.gamma=illumi_gamma
        self.scale=illumi_scale
        self.normal=illumi_normalize
        
     
    def reset_optimizer(self):
        
        self.basecolor=0.01*torch.ones([self.texture_res,self.texture_res,3],device=torch.device('cuda')) 
        self.metallic=self.metallic.detach().clone()
        self.roughness=self.roughness.detach().clone()
        
        self.basecolor.data=torch.nan_to_num(self.basecolor.data.clamp(1e-8,1-1e-8))
        self.metallic.data=torch.nan_to_num(self.metallic.data.clamp(1e-8,1-1e-8))
        self.roughness.data=torch.nan_to_num(self.roughness.data.clamp(1e-8,1-1e-8))
        
        self.basecolor.requires_grad=True
        self.metallic.requires_grad=True
        self.roughness.requires_grad=True
        
        self.optimizer = torch.optim.Adam([
                                        {'params': [self.basecolor]},
                                        {'params': [self.roughness], 'lr': 1.5e-2},
                                        {'params': [self.metallic], 'lr': 1e-2},
                                      
                    ], lr=2e-2)#
        
        
############# rendering method ###################        
    @dr.wrap_ad(source='torch', target='drjit')
    def render_envmap(self,envmap,spp=256):
        
        self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        
        self.params['EnvironmentMapEmitter.data']=envmap  
        self.params.update()
        rendered_img=mi.render(self.scene, self.params, spp=spp)

        return rendered_img
       
        
    @dr.wrap_ad(source='torch', target='drjit')
    def render_multiview(self,envmap,spp=256):
        
        cam=random.randint(0,self.n_images-1)
        if cam==0:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        else:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera_{}.to_world'.format(cam)]
        
        
        
        self.params['EnvironmentMapEmitter.data']=envmap
        self.params.update()
        
        rendered_gt=self.rgb_images[cam].cuda()
        
        
        rendered_img=mi.render(self.scene, self.params, spp=spp)
            
        
       

        return rendered_gt,rendered_img
    
    @dr.wrap_ad(source='torch', target='drjit')
    def render_multiview_with_material(self,envmap,basecolor,roughness,metallic,spp=256):
        
        
        cam=random.randint(0,self.n_images-1)
 
        if cam==0:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        else:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera_{}.to_world'.format(cam)]
        
      
        
        self.params['EnvironmentMapEmitter.data']=envmap
        self.params['OBJMesh.bsdf.base_color.data']=basecolor
        self.params['OBJMesh.bsdf.roughness.data']=roughness
        self.params['OBJMesh.bsdf.metallic.data']=metallic

        self.params.update()
        

        
        rendered_gt=self.rgb_images[cam].cuda()
        
        
        rendered_img=mi.render(self.scene, self.params, spp=spp)

       

        return rendered_gt,rendered_img
    

############# forward method ###################       
    def forward(self, data,spp=16, **kwargs):
        data=(data+1.)/2
        data=data.clamp(0,1)
        data=self.scale*torch.pow(data/self.normal, self.gamma)
        
        envmap=torch.ones([256,257,3],device=self.device)
        envmap[:,:,:3]*=1e-8

        envmap[:,:256,:3]=data.squeeze().permute(1,2,0)
        envmap[:,256,:3]=envmap[:,255,:3]
            
        rendered_img=self.render_envmap(envmap,spp=spp) 
        rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)
        
        
        
        return rendered_img
    
    def forward_gt(self,spp=16, **kwargs):

        cam=random.randint(0,self.n_images-1)
        rendered_img=self.rgb_images[cam].cuda()
            
        return rendered_img.permute(2,0,1).unsqueeze(0)
    
    def forward_multiview(self, data,spp=16, **kwargs):
        data_scale=(data+1.)/2
        data_scale=data_scale.clamp(0,1)
        data_scale=self.scale*torch.pow(data_scale/self.normal, self.gamma)
        
        
        envmap=torch.ones([256,257,3],device=self.device)
        envmap[:,:,:3]*=1e-8

        envmap[:,:256,:3]=data_scale.squeeze().permute(1,2,0)
        envmap[:,256,:3]=envmap[:,255,:3]
            
        rendered_gt,rendered_img=self.render_multiview(envmap,spp=spp)
        
        
        
        rendered_gt=rendered_gt.permute(2,0,1).unsqueeze(0)
        rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)
        
        
        
        
        
        return rendered_gt,rendered_img    
   
    def update_material(self, data,spp=32,t=0., **kwargs):
        with autocast():
            data_scale=(data+1.)/2
            data_scale=data_scale.clamp(0,1)
            data_scale=self.scale*torch.pow(data_scale/self.normal, self.gamma)
            
            
            
            
            envmap=torch.ones([256,257,3],device=self.device)
            envmap[:,:,:3]*=1e-8
            envmap[:,:256,:3]=data_scale.squeeze().permute(1,2,0).detach()
            envmap[:,256,:3]=envmap[:,255,:3]
            
          
            
            rendered_gt,rendered_img=self.render_multiview_with_material(envmap,self.basecolor,self.roughness,self.metallic,spp=spp)
            
            loss = self.loss_fn(rendered_img, rendered_gt)     
            
        
        
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.basecolor.data=torch.nan_to_num(self.basecolor.data.clamp(min=1e-8))
            self.metallic.data=torch.nan_to_num(self.metallic.data.clamp(1e-8,1-1e-8))
            self.roughness.data=torch.nan_to_num(self.roughness.data.clamp(min=1e-8,max=1.),nan=1.0)
            if int(t*1000)%1.5==0:
                self.basecolor.data=self.basecolor.data/self.basecolor.data.max()
                self.roughness.data=self.roughness.data/self.roughness.data.max()
           
        del(loss,rendered_img,envmap)
    

        
# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)