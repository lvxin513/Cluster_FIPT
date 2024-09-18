import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as NF
import torch.optim as optim
from torch.utils.data import DataLoader
import torch_scatter
import time
import lpips
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')
from utils.Cluster import Cluster,Cluster_Manager
from utils.cluster_albedo import Cluster_albedo,Cluster_Manager_albedo
import math

from pathlib import Path
from argparse import Namespace, ArgumentParser


from configs.config import default_options
from utils.dataset import InvRealDataset,RealDataset,InvSyntheticDataset,SyntheticDataset
from utils.ops import *
from utils.path_tracing import ray_intersect
from model.mlps import ImplicitMLP
from model.brdf import NGPBRDF
import numpy as np
def nor(img):
    img = img.cpu().numpy()
    ldr = img ** (1 / 2.2)
    ldr = np.clip(ldr * 255, 0, 255).astype('uint8')
    return ldr
def nor1(img):
    im = img.cpu().numpy()
    im = im ** (1 / 0.8)
    im = np.clip(im * 255, 0, 255).astype('uint8')
    return im


from skimage.metrics import structural_similarity as ssim

class ModelTrainer(pl.LightningModule):
    """ BRDF-emission mask training code """
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(ModelTrainer, self).__init__()
        self.save_hyperparameters(hparams)
        
        # load scene geometry
        self.scene = mitsuba.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': 'obj',
                'filename': os.path.join(hparams.dataset[1],'scene.obj')
            }
        })
        self.count = 0
        self.cluster_manager1 = Cluster_Manager(class_num=256)
        self.cluster_manager2 = Cluster_Manager(class_num=256)
        self.cluster_albedo = Cluster_Manager_albedo(class_num=256)
        self.test_results = {
            'lpips': [],
            'ssim': [],
            'psnr_a': [],
            'lpips_rou': [],
            'ssim_rou': [],
            'psnr_rou': [],
        }
        # initiallize BRDF
        mask = torch.load(hparams.voxel_path,map_location='cpu')
        self.material = NGPBRDF(mask['voxel_min'],mask['voxel_max'])
       
        # intiialize emission mask
        self.emission_mask = ImplicitMLP(6,128,[3],1,10)
        
        
    def __repr__(self):
        return repr(self.hparams)

    def configure_optimizers(self):
        if(self.hparams.optimizer == 'SGD'):
            opt = optim.SGD
        if(self.hparams.optimizer == 'Adam'):
            opt = optim.Adam
        
        optimizer = opt(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)    
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.hparams.milestones,gamma=self.hparams.scheduler_rate)
        return [optimizer], [scheduler]
    
    def train_dataloader(self,):
        dataset_name,dataset_path,cache_path = self.hparams.dataset
        
        if dataset_name == 'synthetic':
            dataset = InvSyntheticDataset(dataset_path,cache_path,pixel=True,split='train',
                                       batch_size=self.hparams.batch_size,has_part=self.hparams.has_part)
        elif dataset_name == 'real':
            dataset = InvRealDataset(dataset_path,cache_path,pixel=True,split='train',
                                       batch_size=self.hparams.batch_size)
       
        return DataLoader(dataset, batch_size=None, num_workers=self.hparams.num_workers)
        # return DataLoader(dataset, batch_size=None, num_workers=0)
    # self.hparams.num_workers
       
    def on_train_epoch_start(self,):
        """ resample training batch """
        self.train_dataloader().dataset.resample()
    
    def val_dataloader(self,):
        dataset_name,dataset_path,cache_path = self.hparams.dataset
        self.dataset_name = dataset_name

        if dataset_name == 'synthetic':
            dataset = InvSyntheticDataset(dataset_path,"outputs_val/kitchen",pixel=False,split='val',has_part=self.hparams.has_part)
        elif dataset_name == 'real':
            dataset = InvRealDataset(dataset_path,"outputs/classroom",pixel=False,split='val')
        
        self.img_hw = dataset.img_hw
        return DataLoader(dataset, shuffle=False, batch_size=None, num_workers=self.hparams.num_workers)
        # return DataLoader(dataset, shuffle=False, batch_size=None, num_workers=0)
    def test_dataloader(self, ):
        dataset_name, dataset_path, cache_path = self.hparams.dataset
        self.dataset_name = dataset_name

        if dataset_name == 'synthetic':
            dataset = InvSyntheticDataset(dataset_path,"outputs_val/kitchen",pixel=False,split='val',has_part=self.hparams.has_part)
        elif dataset_name == 'real':
            dataset = InvRealDataset(dataset_path,"outputs/classroom",pixel=False,split='val')

        self.img_hw = dataset.img_hw
        return DataLoader(dataset, shuffle=False, batch_size=None, num_workers=self.hparams.num_workers)
        # return DataLoader(dataset, shuffle=False, batch_size=None, num_workers=0)

    def forward(self, points, view):
        return

    def gamma(self,x):
        """ tone mapping function """
        mask = x <= 0.0031308
        ret = torch.empty_like(x)
        ret[mask] = 12.92*x[mask]
        mask = ~mask
        ret[mask] = 1.055*x[mask].pow(1/2.4) - 0.055
        return ret
    
    def training_step(self, batch, batch_idx):
        """ one training step """
        rays,rgbs_gt = batch['rays'], batch['rgbs']
        xs,ds = rays[...,:3],rays[...,3:6]
        ds = NF.normalize(ds,dim=-1)
        
        if self.dataset_name == 'synthetic': # only available for synthetic scene
            albedos_gt = batch['albedo']

        # fetch shadings
        diffuse = batch['diffuse']
        specular0 = batch['specular0']
        specular1 = batch['specular1']
        
        # fetch segmentation
        segmentation = batch['segmentation'].long()
        
        # find surface intersection
        positions,normals,_,_,valid = ray_intersect(self.scene,xs,ds)

        if not valid.any():
            return None
        
        # optimize only valid surface
        normals = normals[valid]
        rgbs_gt = rgbs_gt[valid]
        positions = positions[valid]
        diffuse=diffuse[valid]
        specular0 = specular0[valid]
        specular1 = specular1[valid]
        if self.dataset_name == 'synthetic':
            albedos_gt = albedos_gt[valid]
  
        segmentation = segmentation[valid]
        
        # get brdf
        std_dev = 0.01
        epsilon = std_dev * torch.randn_like(positions)
        mat = self.material(positions)
        mat_b = self.material(positions+epsilon)
        albedo_b,metallic_b,roughness_b = mat_b['albedo'],mat_b['metallic'],mat_b['roughness']
        albedo,metallic,roughness = mat['albedo'],mat['metallic'],mat['roughness']
        loss_ab = 0.05 / 3 * torch.sum(torch.abs(albedo_b-albedo)) / (1024 * 16)
        loss_mb = 0.05 / 3 * torch.sum(torch.abs(metallic_b-metallic)) / (1024 * 16)
        loss_rb = 0.05 / 3 * torch.sum(torch.abs(roughness_b - roughness)) / (1024 * 16)
        # diffuse and specular reflectance
        kd = albedo*(1-metallic)
        ks = 0.04*(1-metallic) + albedo*metallic
       
        # diffuse component and specular component
        Ld = kd*diffuse
        Ls = ks*lerp_specular(specular0,roughness)+lerp_specular(specular1,roughness)
        rgbs = Ld+Ls
        

        # get emission mask
        emission_mask = self.emission_mask(positions)
        alpha = (1-torch.exp(-emission_mask.relu()))    
        
        
        # mask out emissive regions
        rgbs = (1-alpha)*rgbs+rgbs_gt*alpha
        # tonemapped mse loss
        loss_c = NF.mse_loss(self.gamma(rgbs),self.gamma(rgbs_gt))
        
        # regualrize emission mask to be small
        loss_e = self.hparams.le * emission_mask.abs().mean()
        
        # diffuse regualrization
        loss_d = self.hparams.ld * ((roughness-1).abs().mean()+metallic.mean())

        # roughness-metallic propagation regularization
        if self.hparams.has_part:
            # with part segmentation

            # find mean roughness-metallic for each segmentation id
            seg_idxs,inv_idxs = segmentation.unique(return_inverse=True)
            weight_seg = torch.zeros(len(seg_idxs),device=seg_idxs.device)
            mean_metallic = torch.zeros(len(seg_idxs),device=seg_idxs.device)
            mean_roughness = torch.zeros(len(seg_idxs),device=seg_idxs.device)

            weight_seg_ = Ls.data.mean(-1)+1e-4 # weight surface with high reflection more

            mean_metallic = torch_scatter.scatter(
                metallic.squeeze(-1)*weight_seg_,inv_idxs,0,mean_metallic,reduce='sum').unsqueeze(-1)
            mean_roughness = torch_scatter.scatter(
                roughness.squeeze(-1)*weight_seg_,inv_idxs,0,mean_roughness,reduce='sum').unsqueeze(-1)
            weight_seg = torch_scatter.scatter(weight_seg_,inv_idxs,0,weight_seg,reduce='sum').unsqueeze(-1)


            mean_metallic = mean_metallic/weight_seg
            mean_roughness = mean_roughness/weight_seg

            # propagation loss
            loss_seg = (metallic-mean_metallic[inv_idxs]).abs().mean()\
                     + (roughness-mean_roughness[inv_idxs]).abs().mean()
            loss_seg = self.hparams.lp*loss_seg
        else:
            # with semantic segmentation

            # normalize input position
            positions = (positions-self.material.voxel_min)/(self.material.voxel_max-self.material.voxel_min)*2-1

            # find mean amount all the pixels is expensive, only sample subset (1024) of them
            seg_idxs,inv_idxs,seg_counts = segmentation.unique(return_inverse=True,return_counts=True)
            ii,jj = [],[]
            for seg_idx,seg_count in zip(seg_idxs,seg_counts):
                sample_batch = 1024
                i = torch.where(segmentation==seg_idx)[0]
                if sample_batch > seg_count:
                    sample_batch = seg_count
                    j = torch.arange(seg_count,device=seg_idxs.device)[None].repeat_interleave(sample_batch,0).reshape(-1)
                else:
                    j = torch.randint(0,seg_count,(seg_count*sample_batch,),device=seg_idxs.device)
                j = i[j]
                i = i.repeat_interleave(sample_batch,0)
                ii.append(i)
                jj.append(j)
            ii = torch.cat(ii,0)
            jj = torch.cat(jj,0)


            # weight more of close pixels with similar albedo
            weight_seg_ = torch.exp(-(
                        (albedo.data[ii]-albedo.data[jj]).pow(2).sum(-1)
                        /self.hparams.sigma_albedo**2)/2.0)
            weight_seg_ *= torch.exp(-((positions[ii]-positions[jj]).pow(2).sum(-1)
                        /self.hparams.sigma_pos**2)/2.0)


            weight_seg = torch.zeros(len(positions),device=positions.device)+1e-4
            roughness_mean = torch.zeros(len(roughness),device=roughness.device)
            metallic_mean = torch.zeros(len(metallic),device=metallic.device)

            # calculate mean for each pixel
            roughness_mean.scatter_add_(0,ii,roughness[jj].squeeze(-1)*weight_seg_)
            metallic_mean.scatter_add_(0,ii,metallic[jj].squeeze(-1)*weight_seg_)
            weight_seg.scatter_add_(0,ii,weight_seg_)
            roughness_mean = roughness_mean/weight_seg
            metallic_mean = metallic_mean/weight_seg


            loss_seg_ = (roughness_mean-roughness.squeeze(-1)).abs()+(metallic_mean-metallic.squeeze(-1)).abs()

            # propagation
            loss_seg = torch.zeros(len(seg_idxs),device=seg_idxs.device)
            loss_seg = torch_scatter.scatter(loss_seg_,inv_idxs,0,loss_seg,reduce='mean')
            loss_seg = self.hparams.ls*loss_seg.sum()


        # vsualize rendering brdf
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        loss = loss_c + loss_e + loss_d + loss_seg
        if self.global_step >= 1000:
        # if batch_idx > 500:
            seg = segmentation.detach().cpu().numpy()
            seg_ = segmentation.clone()
            albedo_cluster = albedo.detach().cpu().numpy()
            albedo_cluster_ = albedo.clone()
            roughness_cluster = roughness.detach().cpu().numpy()
            roughness_cluster_ = roughness.clone()
            metallic_cluster = metallic.detach().cpu().numpy()
            metallic_cluster_ = metallic.clone()
            if self.count % 50 == 0:
                self.cluster_albedo.update_center(seg,albedo_cluster,quantile=0.3,
                                               n_samples=7000, band_factor=0.5)
            mapped_albedo = self.cluster_albedo.dest_color(albedo_cluster_,seg_)
            if self.count % 50 ==0:
                self.cluster_manager1.update_center(seg, roughness_cluster, quantile=0.3,
                                               n_samples=7000, band_factor=0.5)
            mapped_roughness = self.cluster_manager1.dest_color(roughness_cluster_, seg_)

            if self.count % 50 == 0:
                self.cluster_manager2.update_center(seg, metallic_cluster, quantile=0.3,
                                               n_samples=7000, band_factor=0.5)

            mapped_metallic = self.cluster_manager2.dest_color(metallic_cluster_, seg_)

            loss_l1 = torch.nn.L1Loss()
            loss_a = 0.025 * loss_l1(mapped_albedo, albedo)
            loss_r = 0.01 * loss_l1(mapped_roughness, roughness)
            loss_m = 0.01 * loss_l1(mapped_metallic, metallic)
            loss = loss + loss_r + loss_m + loss_a

            self.count += 1


        if self.dataset_name == 'synthetic':
            albedo_loss = NF.mse_loss(albedos_gt,kd.data)
            self.log('train/albedo', albedo_loss)
        self.log('train/loss', loss)
        self.log('train/psnr', psnr)

        return loss

    def validation_step(self, batch, batch_idx):
        """ visualize diffuse reflectance kd
        """
        rays, rgb_gt = batch['rays'], batch['rgbs']
        diffuse = batch['diffuse']
        specular0 = batch['specular0']
        specular1 = batch['specular1']
        # albedo_truth = batch['albedo']
        # roughness_truth = batch['roughness']
        # metallic_truth = batch['metallic']
        if self.dataset_name == 'synthetic':
            emission_mask_gt = batch['emission'].mean(-1, keepdim=True) == 0
        else:
            emission_mask_gt = torch.ones_like(rays[..., :1])
        rays_x = rays[:, :3]
        rays_d = NF.normalize(rays[:, 3:6], dim=-1)

        positions, normals, _, _, valid = ray_intersect(self.scene, rays_x, rays_d)
        position = positions[valid]

        # batched rendering diffuse reflectance
        B = valid.sum()
        batch_size = 10240
        albedo_ = []
        roughness_ = []
        metallic_ = []
        emit_ = []
        rgbs_ = []
        for b in range(math.ceil(B * 1.0 / batch_size)):
            b0 = b * batch_size
            b1 = min(b0 + batch_size, B)
            emit = self.emission_mask(position[b0:b1])
            emit_.append(emit)
            mat = self.material(position[b0:b1])
            albedo = mat['albedo']
            albedo_.append(albedo)
            roughness = mat['roughness']
            roughness_.append(roughness)
            metallic = mat['metallic']
            metallic_.append(metallic)
            kd = albedo * (1 - metallic)
            ks = 0.04 * (1 - metallic) + albedo * metallic
            Ld = kd * diffuse[b0:b1]
            # roughness[roughness < 0.02] = 0.02
            Ls = ks * lerp_specular(specular0[b0:b1], roughness) + lerp_specular(specular1[b0:b1], roughness)
            rgbs = Ld + Ls
            emission_mask = self.emission_mask(positions[b0:b1])
            alpha = (1 - torch.exp(-emission_mask.relu()))

            # mask out emissive regions
            rgbs = (1 - alpha) * rgbs + rgb_gt[b0:b1] * alpha
            rgbs_.append(rgbs)

        albedo_ = torch.cat(albedo_)
        albedo = torch.zeros(len(valid), 3, device=valid.device)
        albedo[valid] = albedo_
        # albedo = nor(albedo)
        emit_ = torch.cat(emit_)
        emit =  torch.zeros(len(valid),1,device=valid.device)
        emit [valid] = emit_
        roughness_ = torch.cat(roughness_)
        roughness = torch.zeros(len(valid), 1, device=valid.device)
        roughness[valid] = roughness_
        # roughness = nor(roughness)

        metallic_ = torch.cat(metallic_)
        metallic = torch.zeros(len(valid), 1, device=valid.device)
        metallic[valid] = metallic_
        # metallic =nor(metallic)

        rgbs_ = torch.cat(rgbs_)
        rgbs = torch.zeros(len(valid), 3, device=valid.device)
        rgbs[valid] = rgbs_
        # rgbs = nor(rgbs)
        if self.dataset_name == 'synthetic':
            albedo_gt = batch['albedo']
        else:  # show rgb is no ground truth kd
            albedo_gt = rgb_gt.pow(1 / 2.2).clamp(0, 1)

        # mask out emissive regions
        albedo = albedo*emission_mask_gt
        albedo_gt = albedo_gt * emission_mask_gt
        # albedo = normalize_array(albedo)
        # roughness = normalize_array(roughness)
        # metallic = normalize_array(metallic)
        # albedo = torch.tensor(albedo).to('cuda')
        # roughness = torch.tensor(roughness).to('cuda')
        # metallic = torch.tensor(metallic).to('cuda')
        # roughness = roughness * emission_mask_gt
        # roughness_gt = roughness_truth * emission_mask_gt
        # roughness_min = roughness.min()
        # roughness_max = roughness.max()
        # roughness = (roughness - roughness_min) / (roughness_max - roughness_min)
        roughness = roughness * emission_mask_gt
        #
        # # metallic_min = metallic.min()
        # # metallic_max = metallic.max()
        # # metallic = (metallic-metallic_min) / (metallic_max - metallic_min)
        metallic = metallic * emission_mask_gt
        loss_r = NF.mse_loss(rgbs, rgb_gt)
        loss_a = NF.mse_loss(albedo, albedo_gt)
        # loss_rou = NF.mse_loss(roughness_gt, roughness)
        # loss = loss_c
        psnr_r = -10.0 * math.log10(loss_r.clamp_min(1e-5))
        psnr_a = -10.0 * math.log10(loss_a.clamp_min(1e-5))
        # psnr_rou = -10.0 * math.log10(loss_rou.clamp_min(1e-5))


        self.log('val/loss', loss_r)
        # self.log('val/loss_a', loss_a)
        # # self.log('val/psnr_rou', psnr_rou)
        # self.log('val/psnr_a', psnr_a)
        # self.log('val/psnr_r', psnr_r)
        # self.log('val/psnr_rou', psnr_rou)

        self.logger.experiment.add_image('val/emit_image',emit.reshape(1,*self.img_hw) , batch_idx)
        self.logger.experiment.add_image('val/roughness_image', roughness.reshape(1, *self.img_hw), batch_idx)
        self.logger.experiment.add_image('val/metallic_image', metallic.reshape(1, *self.img_hw), batch_idx)
        self.logger.experiment.add_image('val/input_image',
                                         nor(rgb_gt.reshape(*self.img_hw, 3).permute(2, 0, 1)), batch_idx)
        self.logger.experiment.add_image('val/albedo', nor(albedo.reshape(*self.img_hw, 3).permute(2, 0, 1)),
                                         batch_idx)
        # self.logger.experiment.add_image('val/render', nor(rgbs.reshape(*self.img_hw, 3).permute(2, 0, 1)),
        #                                  batch_idx)

        return

    def test_step(self, batch, batch_idx):
        """ visualize diffuse reflectance kd
        """
        rays, rgb_gt = batch['rays'], batch['rgbs']
        diffuse = batch['diffuse']
        specular0 = batch['specular0']
        specular1 = batch['specular1']
        if self.dataset_name == 'synthetic':
            emission_mask_gt = batch['emission'].mean(-1, keepdim=True) == 0
        else:
            emission_mask_gt = torch.ones_like(rays[..., :1])
        rays_x = rays[:, :3]
        rays_d = NF.normalize(rays[:, 3:6], dim=-1)

        positions, normals, _, _, valid = ray_intersect(self.scene, rays_x, rays_d)
        position = positions[valid]

        # batched rendering diffuse reflectance
        B = valid.sum()
        batch_size = 10240
        albedo_ = []
        roughness_ = []
        metallic_ = []
        emit_ = []
        rgbs_ = []
        for b in range(math.ceil(B * 1.0 / batch_size)):
            b0 = b * batch_size
            b1 = min(b0 + batch_size, B)
            emit = self.emission_mask(position[b0:b1])
            emit_.append(emit)
            mat = self.material(position[b0:b1])
            albedo = mat['albedo']
            albedo_.append(albedo)
            roughness = mat['roughness']
            roughness_.append(roughness)
            metallic = mat['metallic']
            metallic_.append(metallic)
            kd = albedo * (1 - metallic)
            ks = 0.04 * (1 - metallic) + albedo * metallic
            Ld = kd * diffuse[b0:b1]
            # roughness[roughness < 0.02] = 0.02
            Ls = ks * lerp_specular(specular0[b0:b1], roughness) + lerp_specular(specular1[b0:b1], roughness)
            rgbs = Ld + Ls
            emission_mask = self.emission_mask(positions[b0:b1])
            alpha = (1 - torch.exp(-emission_mask.relu()))

            # mask out emissive regions
            rgbs = (1 - alpha) * rgbs + rgb_gt[b0:b1] * alpha
            rgbs_.append(rgbs)

        albedo_ = torch.cat(albedo_)
        albedo = torch.zeros(len(valid), 3, device=valid.device)
        albedo[valid] = albedo_
        # albedo = nor(albedo)
        emit_ = torch.cat(emit_)
        emit =  torch.zeros(len(valid),1,device=valid.device)
        emit [valid] = emit_
        roughness_ = torch.cat(roughness_)
        roughness = torch.zeros(len(valid), 1, device=valid.device)
        roughness[valid] = roughness_
        # roughness = nor(roughness)

        metallic_ = torch.cat(metallic_)
        metallic = torch.zeros(len(valid), 1, device=valid.device)
        metallic[valid] = metallic_
        # metallic =nor(metallic)

        rgbs_ = torch.cat(rgbs_)
        rgbs = torch.zeros(len(valid), 3, device=valid.device)
        rgbs[valid] = rgbs_
        # rgbs = nor(rgbs)
        if self.dataset_name == 'synthetic':
            albedo_gt = batch['albedo']
            roughness_gt = batch['roughness']
        else:  # show rgb is no ground truth kd
            albedo_gt = rgb_gt.pow(1 / 2.2).clamp(0, 1)

        # mask out emissive regions
        albedo = albedo*emission_mask_gt
        albedo_gt = albedo_gt * emission_mask_gt
        roughness = roughness*emission_mask_gt
        roughness_gt = roughness_gt.reshape(-1,1)* emission_mask_gt
        # albedo = normalize_array(albedo)
        # roughness = normalize_array(roughness)
        # metallic = normalize_array(metallic)
        # albedo = torch.tensor(albedo).to('cuda')
        # roughness = torch.tensor(roughness).to('cuda')
        # metallic = torch.tensor(metallic).to('cuda')
        # roughness = roughness * emission_mask_gt
        # roughness_gt = roughness_truth * emission_mask_gt
        # roughness_min = roughness.min()
        # roughness_max = roughness.max()
        # roughness = (roughness - roughness_min) / (roughness_max - roughness_min)
        # metallic_min = metallic.min()
        # metallic_max = metallic.max()
        # metallic = (metallic-metallic_min) / (metallic_max - metallic_min)
        metallic = metallic * emission_mask_gt
        # loss_r = NF.mse_loss(rgbs, rgb_gt)
        loss_a = NF.mse_loss(albedo, albedo_gt)
        loss_rou = NF.mse_loss(roughness_gt, roughness)
        # loss = loss_c
        psnr_rou = -10.0 * math.log10(loss_rou.clamp_min(1e-5))
        psnr_a = -10.0 * math.log10(loss_a.clamp_min(1e-5))
        # psnr_rou = -10.0 * math.log10(loss_rou.clamp_min(1e-5))
        a = (albedo.cpu().numpy()).reshape(*self.img_hw, 3)
        a_gt = (albedo_gt.cpu().numpy()).reshape(*self.img_hw, 3)
        rou = (roughness.cpu().numpy()).reshape(*self.img_hw, 1)
        rou_gt = (roughness_gt.cpu().numpy()).reshape(*self.img_hw, 1)
        ssim_value = ssim(a, a_gt,channel_axis=-1,data_range=1.0)
        ssim_value_rou = ssim(rou, rou_gt,channel_axis=-1,data_range=1.0)
        lpips_model = lpips.LPIPS(net='alex').to('cuda')
        lpips_value = lpips_model(albedo.reshape(1,3,*self.img_hw), albedo_gt.reshape(1,3,*self.img_hw))
        lpips_value_rou = lpips_model(roughness.reshape(1,1,*self.img_hw), roughness_gt.reshape(1,1,*self.img_hw))
        self.test_results['lpips'].append(lpips_value.item())
        self.test_results['ssim'].append(ssim_value)
        # self.test_results['loss_a'].append(loss_a.item())
        self.test_results['psnr_a'].append(psnr_a)
        self.test_results['lpips_rou'].append(lpips_value_rou.item())
        self.test_results['ssim_rou'].append(ssim_value_rou)
        self.test_results['psnr_rou'].append(psnr_rou)
        # self.logger.experiment.add_image('test/emit_image',emit.reshape(1,*self.img_hw) , batch_idx)
        # self.logger.experiment.add_image('test/roughness_image', roughness.reshape(1, *self.img_hw), batch_idx)
        # self.logger.experiment.add_image('test/metallic_image', metallic.reshape(1, *self.img_hw), batch_idx)
        # self.logger.experiment.add_image('test/input_image',
        #                                  nor(rgb_gt.reshape(*self.img_hw, 3).permute(2, 0, 1)), batch_idx)
        # self.logger.experiment.add_image('test/albedo', nor(albedo.reshape(*self.img_hw, 3).permute(2, 0, 1)),
        #                                  batch_idx)
        # self.logger.experiment.add_image('test/render', nor(rgbs.reshape(*self.img_hw, 3).permute(2, 0, 1)),
        #                                  batch_idx)

        return

    def on_test_epoch_end(self):
        # Compute the average values over all test batches
        print('the length of data',len(self.test_results['lpips']))
        avg_lpips_value = sum(self.test_results['lpips']) / len(self.test_results['lpips'])
        avg_ssim_value = sum(self.test_results['ssim']) / len(self.test_results['ssim'])
        avg_psnr_a = sum(self.test_results['psnr_a']) / len(self.test_results['psnr_a'])
        avg_lpips_value_rou = sum(self.test_results['lpips_rou']) / len(self.test_results['lpips_rou'])
        avg_ssim_value_rou = sum(self.test_results['ssim_rou']) / len(self.test_results['ssim_rou'])
        avg_psnr_rou = sum(self.test_results['psnr_rou']) / len(self.test_results['psnr_rou'])
        print(f"Average LPIPS value: {avg_lpips_value}")
        print(f"Average SSIM value: {avg_ssim_value}")
        print(f"Average PSNR A: {avg_psnr_a}")
        print(f"Average LPIPS_rou value: {avg_lpips_value_rou}")
        print(f"Average SSIM_rou value: {avg_ssim_value_rou}")
        print(f"Average PSNR rou: {avg_psnr_rou}")

def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser
        
if __name__ == '__main__':

    torch.manual_seed(9)
    torch.cuda.manual_seed(9)

    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--ft', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    # parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', type=int, required=False,default=None)

    # parser.set_defaults(resume=False)
    args = parser.parse_args()
    args.gpus = [args.device]
    experiment_name = args.experiment_name

    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    log_path = Path(args.log_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val/loss', save_top_k=1)
    logger = TensorBoardLogger(log_path, name=experiment_name)

    # last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    # if (last_ckpt is None) or (not (last_ckpt.exists())):
    #     last_ckpt = None
    # else:
    #     last_ckpt = str(last_ckpt)
    
    # setup model trainer
    model = ModelTrainer(hparams)

    # model_test = ModelTrainer.load_from_checkpoint("/proj/users/xlv/lvxin/fipt-origin/checkpoints/livingroom/cluster_part.ckpt",hparams=hparams)
    # model_test = ModelTrainer.load_from_checkpoint(
    #     "/proj/users/xlv/lvxin/fipt-origin/checkpoints/kitchen/epoch=0-step=5050.ckpt", hparams=hparams)
    # trainer = Trainer.from_argparse_args(
    #     args,
    #     resume_from_checkpoint=last_ckpt,
    #     logger=logger,
    #     checkpoint_callback=checkpoint_callback,
    #     flush_logs_every_n_steps=1,
    #     log_every_n_steps=1,
    #     max_epochs=args.max_epochs
    # )

    # trainer.fit(model)
    
    # Update to lightning 1.9
    # trainer = Trainer.from_argparse_args(
    #     args,
    #     accelerator='gpu', devices=[0], gpus=None,
    #     logger=logger,
    #     callbacks=[checkpoint_callback],
    #     log_every_n_steps=1,
    #     max_epochs=args.max_epochs,
    # )
    trainer = Trainer.from_argparse_args(
        args,
        accelerator='gpu', devices=[0], gpus=None,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        max_epochs=args.max_epochs,
    )

    start_time = time.time()
    #
    trainer.fit(
        model,
        # ckpt_path=last_ckpt,
    )
    trainer.test(model)


    print('[train - BRDF-emission] time (s): ', time.time()-start_time)
