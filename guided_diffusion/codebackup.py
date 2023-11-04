self.scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'prb',#'path',#
                       # 'max_depth': 2,
                        'hide_emitters':True},
        'sensor':  {
            'type': 'perspective',
            'to_world': T.look_at(
                            origin=(0, 0, -2),
                            target=(0, 0, 0),
                            up=(0, 1, 0)
                        ),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width':  512,
                'height': 512,
            },
        },
        'glass_sphere': {
            'type': 'obj',
            'filename': '/HPS/VisibilityLearning/work/Programs/IDR/logs/mitsuba/mesh/sphere.obj',
            'to_world': T.translate([0, 0, -1]).scale(0.5),
            'bsdf': {
                "type": "roughconductor",
                'alpha':{
                    'type': 'bitmap',
                    'filename': '/HPS/VisibilityLearning/work/Programs/IDR/logs/mitsuba/mesh/alpha_64.png'
                }
            },
        },
        'emitter': {
            'type': 'envmap',
            'filename': '/HPS/VisibilityLearning/work/Programs/IDR/logs/nf_glow/gt/bedroom0.jpg'
        }
    })
    
   
            
    
    self.device = device
    self.params = mi.traverse(self.scene)
    

  
    
    self.loss_fn = torch.nn.MSELoss()
    
    self.gt_alpha=mi.traverse(self.scene)['glass_sphere.bsdf.alpha.data'].torch().clone()                  
    self.alpha = 0.001*torch.ones([64,64,1],device=torch.device('cuda')) 
    self.alpha.requires_grad=True
    # self.alpha=pyredner.imread('/HPS/VisibilityLearning/work/Programs/IDR/logs/mitsuba/mesh/sphere_xyz.png',gamma=1).cuda()

    self.optimizer =torch.optim.AdamW([self.alpha], lr=5e-4, weight_decay=1e-5)   #self.model.parameters()
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=2)                           
    self.seed=0
    self.loss_fn = torch.nn.MSELoss()#torch.nn.GaussianNLLLoss(eps=1e-5)#

def reset(self):
    
    
    self.alpha = 0.001*torch.ones([64,64,1],device=torch.device('cuda')) 
    self.alpha.requires_grad=True
    
    self.optimizer =torch.optim.AdamW([self.alpha], lr=5e-4, weight_decay=1e-5)   #self.model.parameters()
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=300, gamma=2) 
    self.seed=0
    
def reset_optimizer(self):
   
    
    self.optimizer =torch.optim.AdamW([self.alpha], lr=1e-2, weight_decay=1e-5)   #self.model.parameters()
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
    self.seed=0
    
@dr.wrap_ad(source='torch', target='drjit')
def render_envmap(self,envmap,spp=256):
    
    
    self.params['emitter.data']=envmap #EnvironmentMapEmitter#
    self.params['glass_sphere.bsdf.alpha.data']=self.gt_alpha
    self.params.update()
    rendered_img=mi.render(self.scene, self.params, spp=spp, seed=self.seed)

    return rendered_img
   

@dr.wrap_ad(source='torch', target='drjit')
def render_alpha(self,envmap,alpha,spp=256):#var,
    
    
    self.params['emitter.data']=envmap #EnvironmentMapEmitter
    self.params['glass_sphere.bsdf.alpha.data']=alpha
    self.params.update()
    rendered_img=mi.render(self.scene, self.params, spp=spp, seed=self.seed)
    
   
    return rendered_img

def forward(self, data,spp=16, **kwargs):
    data=(data+1.)/2
    data=adjust_gamma(data, 2.2)   
    
    envmap=torch.ones([256, 257, 4],device=self.device)
    envmap[:,:,:3]*=1e-8

    envmap[:,:256,:3]=data.squeeze().permute(1,2,0)
    envmap[:,256,:3]=envmap[:,255,:3]
        
    rendered_img=self.render_envmap(envmap,spp=spp)
      
    rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)
    
    
    
    return rendered_img



def forward_full(self, data,spp=16, **kwargs):
    with autocast():
        data=(data+1.)/2
        data=adjust_gamma(data, 2.2)
   
        envmap=torch.ones([256, 257, 4],device=self.device)
        envmap[:,:,:3]*=1e-8
        envmap[:,:256,:3]=data.squeeze().permute(1,2,0)
        envmap[:,256,:3]=envmap[:,255,:3]
            
        

        alpha3=torch.cat([self.alpha,self.alpha,self.alpha],dim=-1).clamp(1e-9,1-1e-9)
        rendered_img=self.render_alpha(envmap,alpha3,spp=spp)
            
        rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)
                          
    return rendered_img


def update(self, data,y_n,spp=32, **kwargs):
    with autocast():
        data=(data+1.)/2
        data=adjust_gamma(data, 2.2)
        
        target_img=y_n.squeeze().permute(1,2,0).detach()
        
        
        envmap=torch.ones([256, 257, 4],device=self.device)
        envmap[:,:,:3]*=1e-8
        envmap[:,:256,:3]=data.squeeze().permute(1,2,0).detach()
        envmap[:,256,:3]=envmap[:,255,:3]
        
      
        alpha3=torch.cat([self.alpha,self.alpha,self.alpha],dim=-1).clamp(1e-9,1-1e-9)
        rendered_img=self.render_alpha(envmap,alpha3,spp=spp)
        
        loss = self.loss_fn(rendered_img, target_img)     
        
    
    
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        self.alpha.data=self.alpha.data.clamp(1e-9,1-1e-9)
        self.seed+=1
    del(loss,rendered_img,envmap)
    
############################## Synthetic ################################
@register_operator(name='raytracing')
class RaytracingOperator(NonLinearOperator):
    def __init__(self,device):
        
        #self.scene=mi.load_file('/HPS/VisibilityLearning/work/Programs/IDR/data/lego/lego/scene.xml')
        self.device = device
        self.gt_scene=mi.load_file('/HPS/VisibilityLearning/work/Programs/IDR/data/teapot/scene.xml')
        self.gt_params = mi.traverse(self.gt_scene)
        self.gt_envmap=self.gt_params['EnvironmentMapEmitter.data'].torch().clone()
        
        self.scene=mi.load_dict({
            'type': 'scene',
            'integrator': {'type': 'path',#'prb',#
                            'max_depth': 4,
                            'hide_emitters':True},
            'PerspectiveCamera':  {
                'type': 'perspective',
                'to_world': T.look_at(
                                origin=(-1.2, 0.5, 1.2),
                                target=(0.4, 0.45, 0.5),
                                up=(0, 1, 0)
                            ),
                'fov': 39,
                'film': {
                    'type': 'hdrfilm',
                    'width':  512,
                    'height': 512,
                },
            },
            'lego': {
                'type': 'obj',
                'filename': '/HPS/VisibilityLearning/work/Programs/IDR/data/teapot/mesh.obj',
               
                'bsdf': {
                    'type': 'principled',
                        'base_color': {
                            'type': 'bitmap',
                            'filename': '/HPS/VisibilityLearning/work/Programs/IDR/logs/mitsuba/teapot/basecolor/2800.png'
                        },
                        'metallic':{
                            'type': 'bitmap',
                            'filename': '/HPS/VisibilityLearning/work/Programs/IDR/logs/mitsuba/teapot/metallic/2800.png'
                        },
                        'specular': 0.5,
                        'roughness': {
                            'type': 'bitmap',
                            'filename': '/HPS/VisibilityLearning/work/Programs/IDR/logs/mitsuba/teapot/roughness/2800.png'
                        },
                        'spec_tint': 0.0,
                        'anisotropic': 0.0,
                        'sheen': 0.0,
                        'sheen_tint': 0.0,
                        'clearcoat': 0.0,
                        # 'clearcoat_glass': 0.3,
                        'spec_trans': 0.0
                },
            },
            'EnvironmentMapEmitter': {
                'type': 'envmap',        
                'filename': '/HPS/VisibilityLearning/work/Programs/diffusion-posterior-sampling/results/raytracing/label/00001.png',
                'scale': 2.
            }
        })
        self.params = mi.traverse(self.scene)
        
        self.basecolor=self.params['lego.bsdf.base_color.data'].torch().clone()
        self.metallic=self.params['lego.bsdf.metallic.data'].torch().clone()[:,:,0:1]
        self.roughness=self.params['lego.bsdf.roughness.data'].torch().clone()
        
        self.basecolor.requires_grad=True
        self.metallic.requires_grad=True
        self.roughness.requires_grad=True
        
        self.optimizer = torch.optim.Adam([
                                        {'params': [self.basecolor]},
                                        {'params': [self.metallic,self.roughness], 'lr': 5e-3},
                                      
                    ], lr=1e-2)#
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=600, gamma=0.5)
        
        self.loss_fn = torch.nn.MSELoss()#nn.GaussianNLLLoss(eps=1e-5)#
    def reset(self):
        
        
        return None
        
    def reset_optimizer(self):
       
        
        return None
        
    @dr.wrap_ad(source='torch', target='drjit')
    def render_envmap(self,envmap,spp=256):
        
        
        # self.params['EnvironmentMapEmitter.data']=envmap     
        # self.params.update()
        rendered_img=mi.render(self.gt_scene, self.gt_params, spp=spp)

        return rendered_img
       
        
    @dr.wrap_ad(source='torch', target='drjit')
    def render_multiview(self,envmap,spp=256):
        cam=random.randint(0,20)
        ########## lego ##############
        # r=1.8
        # phi=20.0 * cam
        # theta=-60.
        # origin = T.rotate([0, 1, 0], phi).rotate([1, 0, 0], theta) @ mi.ScalarPoint3f([0, 0, r])
        # self.params['PerspectiveCamera.to_world']=T.look_at(
        #                     origin=origin,
        #                     target=(0.4, 0.45, 0.5),
        #                     up=(0, 1, 0)
        #                 )               
        
        ########## teapot ##############
        r=16
        phi=20.0* cam
        theta=20.0* cam
        origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])
        
        self.params['PerspectiveCamera.to_world']=T.look_at(
                            origin=origin,
                            target=(0., 0., 0.),
                            up=(0, 1, 0)
                        )
        self.params['lego.bsdf.base_color.data']=self.basecolor
        self.params['lego.bsdf.roughness.data']=self.roughness
        self.params['lego.bsdf.metallic.data']=self.metallic
        
        self.params.update()
        self.gt_params['PerspectiveCamera.to_world']=T.look_at(
                            origin=origin,
                            target=(0., 0., 0.),
                            up=(0, 1, 0)
                        )
        
        
        self.gt_params.update()        
        rendered_gt=mi.render(self.gt_scene, self.gt_params, spp=spp)
        
        self.params['EnvironmentMapEmitter.data']=envmap        
        self.params.update()
        rendered_img=mi.render(self.scene, self.params, spp=spp)

        return rendered_gt,rendered_img
    
    @dr.wrap_ad(source='torch', target='drjit')
    def render_multiview_full(self,envmap,basecolor,roughness,metallic,spp=256):
        cam=random.randint(0,20)
        ########## lego ##############
        # r=1.8
        # phi=20.0 * cam
        # theta=-60.
        # origin = T.rotate([0, 1, 0], phi).rotate([1, 0, 0], theta) @ mi.ScalarPoint3f([0, 0, r])
        # self.params['PerspectiveCamera.to_world']=T.look_at(
        #                     origin=origin,
        #                     target=(0.4, 0.45, 0.5),
        #                     up=(0, 1, 0)
        #                 )               
        
        ########## teapot ##############
        r=16
        phi=20.0* cam
        theta=20.0* cam
        origin = T.rotate([0, 0, 1], phi).rotate([0, 1, 0], theta) @ mi.ScalarPoint3f([0, 0, r])
        
        self.params['PerspectiveCamera.to_world']=T.look_at(
                            origin=origin,
                            target=(0., 0., 0.),
                            up=(0, 1, 0)
                        )
        self.params['lego.bsdf.base_color.data']=basecolor
        self.params['lego.bsdf.roughness.data']=roughness
        self.params['lego.bsdf.metallic.data']=metallic
        
        self.params.update()
        
        self.gt_params['PerspectiveCamera.to_world']=T.look_at(
                            origin=origin,
                            target=(0., 0., 0.),
                            up=(0, 1, 0)
                        )
        
        self.gt_params.update()        
        rendered_gt=mi.render(self.gt_scene, self.gt_params, spp=spp)
        
        self.params['EnvironmentMapEmitter.data']=envmap        
        self.params.update()
        rendered_img=mi.render(self.scene, self.params, spp=spp)

        return rendered_gt,rendered_img
    
    def forward(self, data,spp=16, **kwargs):
        # data=(data+1.)/2
        # data=adjust_gamma(data, 2.2)   
        
        # envmap=torch.ones([256, 257, 4],device=self.device)
        # envmap[:,:,:3]*=1e-8

        # envmap[:,:256,:3]=data.squeeze().permute(1,2,0)
        # envmap[:,256,:3]=envmap[:,255,:3]
            
        rendered_img=self.render_envmap(self.gt_envmap,spp=spp)
          
        rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)
        
        
        
        return rendered_img
    
    def forward_multiview(self, data,spp=16, **kwargs):
        data=(data+1.)/2
        data=adjust_gamma(data, 2.2)   
        
        envmap=torch.ones([256, 257, 4],device=self.device)
        envmap[:,:,:3]*=1e-8

        envmap[:,:256,:3]=data.squeeze().permute(1,2,0)
        envmap[:,256,:3]=envmap[:,255,:3]
            
        rendered_gt,rendered_img=self.render_multiview(envmap,spp=spp)
        
        rendered_gt=rendered_gt.permute(2,0,1).unsqueeze(0)
        rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)
        
        
        
        return rendered_gt,rendered_img    
   
    def update(self, data,spp=32, **kwargs):
        with autocast():
            data=(data+1.)/2
            data=adjust_gamma(data, 2.2)
            
            
            
            
            envmap=torch.ones([256, 257, 4],device=self.device)
            envmap[:,:,:3]*=1e-8
            envmap[:,:256,:3]=data.squeeze().permute(1,2,0).detach()
            envmap[:,256,:3]=envmap[:,255,:3]
            
          
            
            rendered_gt,rendered_img=self.render_multiview_full(envmap,self.basecolor,self.roughness,self.metallic,spp=spp)
            
            loss = self.loss_fn(rendered_img, rendered_gt)     
            
        
        
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            self.basecolor.data=self.basecolor.data.clamp(1e-8,1-1e-8)
            self.metallic.data=self.metallic.data.clamp(1e-8,1-1e-8)
            self.roughness.data=self.roughness.data.clamp(1e-8,1-1e-8)
           
        del(loss,rendered_img,envmap)
        
    
############################# Real world######################
@register_operator(name='raytracing')
class RaytracingOperator(NonLinearOperator):
    def __init__(self,device):
        def load_image_raw(fn) -> np.ndarray:
            return imageio.imread(fn)

        def load_image(fn) -> np.ndarray:
            img = load_image_raw(fn)
            if img.dtype == np.float32: # HDR image
                return img
            else: # LDR image
                return img.astype(np.float32) / 255
        def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
            return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

        def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
            assert f.shape[-1] == 3 or f.shape[-1] == 4
            out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
            assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
            return out
        
        def _load_mask(fn):
            img = torch.tensor(load_image(fn), dtype=torch.float32)
            if len(img.shape) == 2:
                img = img[..., None].repeat(1, 1, 3)
            return img

        def _load_img(fn):
            img = load_image_raw(fn)
            if img.dtype != np.float32: # LDR image
                img = torch.tensor(img / 255, dtype=torch.float32)
                img[..., 0:3] = srgb_to_rgb(img[..., 0:3])
            else:
                img = torch.tensor(img, dtype=torch.float32)
            return img
        
        def focal_length_to_fovy(focal_length, sensor_height):
            return 2 * np.arctan(0.5 * sensor_height / focal_length)
        
        def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.sum(x*y, -1, keepdim=True)

        def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            return 2*dot(x, n)*n - x

        def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
            return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
        
        def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
            return x / length(x, eps)
        
        def lines_focal(o, d):
            d = safe_normalize(d)
            I = torch.eye(3, dtype=o.dtype, device=o.device)
            S = torch.sum(d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...], dim=0)
            C = torch.sum((d[..., None] @ torch.transpose(d[..., None], 1, 2) - I[None, ...]) @ o[..., None], dim=0).squeeze(1)
            return torch.linalg.pinv(S) @ C
        
        base_dir='/HPS/VisibilityLearning/work/Programs/nvdiffrec/data/nerd/'
        scene_name='moldGoldCape_rescaled'#'ethiopianHead_rescaled'#'moldGoldCape_rescaled'#
       

        all_img = [f for f in sorted(glob.glob(os.path.join(base_dir,scene_name, "images", "*"))) 
                   if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        all_mask = [f for f in sorted(glob.glob(os.path.join(base_dir,scene_name, "masks", "*"))) 
                    if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]

        resolution = _load_img(all_img[0]).shape[0:2]


        poses_bounds = np.load(os.path.join(base_dir, scene_name,'poses_bounds.npy'))
        poses        = poses_bounds[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
        poses        = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # Taken from nerf, swizzles from LLFF to expected coordinate system
        poses        = np.moveaxis(poses, -1, 0).astype(np.float32)

        lcol         = np.array([0,0,0,1], dtype=np.float32)[None, None, :].repeat(poses.shape[0], 0)
        imvs    = torch.tensor(np.concatenate((poses[:, :, 0:4], lcol), axis=1), dtype=torch.float32)
        aspect  = resolution[1] / resolution[0] # width / height
        fovy    = focal_length_to_fovy(poses[:, 2, 4], poses[:, 0, 4]).astype(np.float16)

        # Recenter scene so lookat position is origin
        center                = lines_focal(imvs[..., :3, 3], -imvs[..., :3, 2])
        imvs[..., :3, 3] = imvs[..., :3, 3] - center[None, ...]

        self.num_cam=imvs.shape[0]
        self.imvs=imvs
        self.fovy=fovy
        self.all_img=[_load_img(img) for img in all_img]
        self.all_mask=[_load_mask(mask) for mask in all_mask]
        
        m=imvs[0]
        m[:,0]=-m[:,0]
        m[:,2]=-m[:,2]

        R=T(m.numpy().astype(np.float16))
        fov=math.degrees(fovy[0])

        self.scene = mi.load_dict({
            'type': 'scene',
            'integrator': {'type': 'prb',#'path',#
                           'max_depth': 4,
                            'hide_emitters':True},
            'PerspectiveCamera':  {
                'type': 'perspective',
                'to_world':R ,
                            
                'fov': fov,
                'film': {
                    'type': 'hdrfilm',
                    'width':  resolution[1],
                    'height': resolution[0],
                },
            },
            'mesh': {
                'type': 'obj',
                'filename': '/HPS/VisibilityLearning/work/Programs/nvdiffrec/out/nerd_gold/mesh/mesh.obj',#nerd_ehead#
                # 'to_world': T.translate([0, 0, -1]).scale(0.5),
                'bsdf': {
                    'type': 'principled',
                        'base_color': {
                            'type': 'bitmap',
                            'filename': "/HPS/VisibilityLearning/work/Programs/IDR/logs/mitsuba/{}/basecolor/{}.png".format(scene_name,2400)
                        },
                        'metallic':{
                            'type': 'bitmap',
                            'filename': "/HPS/VisibilityLearning/work/Programs/IDR/logs/mitsuba/{}/metallic/{}.png".format(scene_name,2400)
                        },
                        'specular': 0.5,
                        'roughness': {
                            'type': 'bitmap',
                            'filename': "/HPS/VisibilityLearning/work/Programs/IDR/logs/mitsuba/{}/roughness/{}.png".format(scene_name,2400)
                        },
                        'spec_tint': 0.0,
                        'anisotropic': 0.0,
                        'sheen': 0.0,
                        'sheen_tint': 0.0,
                        'clearcoat': 0.0,
                        # 'clearcoat_glass': 0.3,
                        'spec_trans': 0.0
                },
            },
            'EnvironmentMapEmitter': {
                'type': 'envmap',
                'scale': 2.,
                'filename': '/HPS/VisibilityLearning/work/Programs/IDR/logs/nf_glow/gt/bedroom0.jpg'
            }
        })
        
        
        self.device = device
        self.params = mi.traverse(self.scene)
        
       
        
    @dr.wrap_ad(source='torch', target='drjit')
    def render_envmap(self,envmap,spp=256):
        
        
        self.params['EnvironmentMapEmitter.data']=envmap #EnvironmentMapEmitter#        
        self.params.update()
        rendered_img=mi.render(self.scene, self.params, spp=spp)

        return rendered_img
       
        
    @dr.wrap_ad(source='torch', target='drjit')
    def render_multiview(self,envmap,spp=256):
        cam=random.randint(0,self.num_cam-1)
        m=self.imvs[cam]
        m[:,0]=-m[:,0]
        m[:,2]=-m[:,2]

        R=T(m.numpy().astype(np.float16))  
        
        self.params['PerspectiveCamera.to_world']=R
        self.params['EnvironmentMapEmitter.data']=envmap        
        self.params.update()
        
        rendered_img=mi.render(self.scene, self.params, spp=spp)
        rendered_gt=self.all_img[cam].cuda()
        
        return rendered_gt,rendered_img
   
    
    def forward(self, data,spp=16, **kwargs):
        cam=random.randint(0,self.num_cam-1)
        rendered_gt=self.all_img[cam].cuda()
        
        
        rendered_gt=rendered_gt.permute(2,0,1).unsqueeze(0)
        print(rendered_gt.shape)
        return rendered_gt
    
    def forward_multiview(self, data,spp=16, **kwargs):
        data=(data+1.)/2
        data=adjust_gamma(data, 2.2)   
        
        envmap=torch.ones([256, 257, 4],device=self.device)
        envmap[:,:,:3]*=1e-8

        envmap[:,:256,:3]=data.squeeze().permute(1,2,0)
        envmap[:,256,:3]=envmap[:,255,:3]
            
        rendered_gt,rendered_img=self.render_multiview(envmap,spp=spp)
        
        rendered_gt=rendered_gt.permute(2,0,1).unsqueeze(0)
        rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)
       
        return rendered_gt,rendered_img   
    
        
    def update(self, data,spp=32, **kwargs):
        
        return None