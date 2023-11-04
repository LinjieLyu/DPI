from abc import ABC, abstractmethod
import torch
import time

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm

    
@register_conditioning_method(name='rt-ps+')
class RayTracingPosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale_method = kwargs.get('scale_method', 'step')
        self.use_log = kwargs.get('use_log', False)
        self.scales = kwargs.get('scales', [0.1,1.0])
        self.spp=16
    def conditioning(self, x_prev, x_t, x_0_hat, measurement,timestep, **kwargs):
        
      

        x_0_hat_noise = x_0_hat+ 0.05 * torch.rand_like(x_0_hat) 
        x_0_hat_noise=x_0_hat_noise.clamp(-1.,1.)
        
        measurement,esitimation= self.operator.forward_multiview(x_0_hat_noise)
        
        if self.use_log:            
            measurement,esitimation=torch.log(measurement+1),torch.log(esitimation+1)
        
        
        difference = measurement - esitimation
        
        norm= torch.linalg.norm(difference) 
   
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev,retain_graph =False)[0]
        norm_grad=torch.nan_to_num(norm_grad)
        while torch.abs(norm_grad).max().item()>0.1:
            norm_grad=norm_grad*0.1
        


        
        
        if self.scale_method=='step':
            if timestep>=0.9:
                scale=0.01
            elif timestep<0.9 and timestep>=0.5:
                scale=self.scales[0]
            elif timestep<0.5 and timestep>=0.:
                scale=self.scales[1]  
                
                
        elif self.scale_method=='linear':
            if timestep>=0.9:
                scale=0.01
            elif timestep<0.9 and timestep>=0.5:
                scale=0.1-(timestep-0.9)*(self.scales[0]-0.1)/(0.9-0.5)
           
            elif timestep<0.5:
                scale=self.scales[0]+(timestep-0.5)*(self.scales[0]-self.scales[1])/0.5
        else:
            raise NotImplementedError
            
        x_t -= norm_grad * scale
     
        self.operator.update_material(x_0_hat,self.spp,t=timestep)
        
        norm_d=norm.detach()
        
        del norm_grad,esitimation,norm,difference

        return x_t, norm_d