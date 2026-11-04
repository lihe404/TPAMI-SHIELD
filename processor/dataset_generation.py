import logging
import os
import time
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.img_metric import SSIM
from torch.cuda import amp
import torch.distributed as dist
import numpy as np
from torch import optim
from diffattack.diff_latent_attack import preprocess, encoder, ddim_reverse_sample, diffusion_step, register_attention_control, reset_attention_control, register_attention_control_DDP
from diffattack.distances import LpDistance
from diffattack.attentionControl import AttentionControlEdit
from diffattack.utils import aggregate_attention
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import re

# import clip 
import timm

import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing

# TORCH_DTYPE = torch.bfloat16

def save_image(img, pro_img, batch_num, cfg, step, is_test, dataset):

    img_32 = img.to(torch.float32)
    pro_img_32 = pro_img.to(torch.float32)
    if is_test :
        img_dir = cfg.IMG_TEST_DIR
    else:
        img_dir = cfg.IMG_DIR

    img_array = img_32[0].detach().permute(1, 2, 0).cpu().numpy()  # 从 [1, 3, 256, 128] 到 [256, 128, 3]
    ori_img = (((img_array*0.5)+0.5)*255).astype(np.uint8)
    ori_image = Image.fromarray(ori_img)
    if is_test :
        ori_path = img_dir + dataset + '_bs_' + str(batch_num) + 'No_0' + 'ori.jpg'
    else:
        ori_path = img_dir + 'bs_' + str(batch_num) + 'No_0' + 'ori.jpg'

    proimg_array = pro_img_32[0].detach().permute(1, 2, 0).cpu().numpy()  # 从 [1, 3, 256, 128] 到 [256, 128, 3]
    proimg = (((proimg_array*0.5)+0.5)*255).astype(np.uint8)
    pro_image = Image.fromarray(proimg)
    if is_test:
        pro_path = img_dir + dataset + '_bs_' + str(batch_num) + 'No_0' + 'step_' + str(step) + 'pro.jpg'
    else:
        pro_path = img_dir + 'bs_' + str(batch_num) + 'No_0' + 'step_' + str(step) + 'pro.jpg'
    
    absolute_diff = (np.abs( ((proimg_array*0.5)+0.5) - ((img_array*0.5)+0.5) ) * 255).astype(np.uint8)
    diff_image = Image.fromarray(absolute_diff)
    if is_test:
        diff_path = img_dir + dataset + '_bs_' + str(batch_num) + 'No_0' + 'step_' + str(step) + 'abs_diff.jpg'
    else:
        diff_path = img_dir + 'bs_' + str(batch_num) + 'No_0' + 'step_' + str(step) + 'abs_diff.jpg'

    
    # img_array1 = img[1].detach().permute(1, 2, 0).cpu().numpy()  # 从 [1, 3, 256, 128] 到 [256, 128, 3]
    # ori_img1 = (((img_array1*0.5)+0.5)*255).astype(np.uint8)
    # ori_image1 = Image.fromarray(ori_img1)
    # ori_path1 = img_dir + 'bs_' + str(batch_num) + 'No_0' + 'ori1.jpg'

    # proimg_array1 = pro_img[1].detach().permute(1, 2, 0).cpu().numpy()  # 从 [1, 3, 256, 128] 到 [256, 128, 3]
    # proimg1 = (((proimg_array1*0.5)+0.5)*255).astype(np.uint8)
    # pro_image1 = Image.fromarray(proimg1)
    # pro_path1 = img_dir + 'bs_' + str(batch_num) + 'No_0' + 'step_' + str(step) + 'pro1.jpg'
    
    # absolute_diff1 = (np.abs( ((proimg_array1*0.5)+0.5) - ((img_array1*0.5)+0.5) ) * 255).astype(np.uint8)
    # diff_image1 = Image.fromarray(absolute_diff1)
    # diff_path1 = img_dir + 'bs_' + str(batch_num) + 'No_0' + 'step_' + str(step) + 'abs_diff1.jpg'
    
    if batch_num % 100==0 and step == 0 :
        ori_image.save(ori_path)
        # ori_image1.save(ori_path1)
    if batch_num % 100==0 and step != 0:
        pro_image.save(pro_path)
        diff_image.save(diff_path)
        # pro_image1.save(pro_path1)
        # diff_image1.save(diff_path1)

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))
    a = ((img_32[0].permute(1, 2, 0).detach().unsqueeze(0))*0.5)+0.5
    a = a.cpu().numpy()
    b = ((pro_img_32[0].permute(1, 2, 0).detach().unsqueeze(0))*0.5)+0.5
    b = b.cpu().numpy()
    print("L1: {}\tL2: {}\tLinf: {}".format(L1(a, b), L2(a,b), Linf(a,b)))

def get_batch_latent(cfg, img, prompt, model, guidance_scale=0):
    batch_size = img.shape[0]
    latent, inversion_latents = ddim_reverse_sample(img, prompt, model, batch_size=batch_size, 
            num_inference_steps=cfg.DIFFUSION.NUM_INFERENCE_STEPS, guidance_scale=guidance_scale, res=cfg.INPUT.SIZE_TRAIN)
    inversion_latents = inversion_latents[::-1]
    latent = inversion_latents[cfg.DIFFUSION.START_STEP - 1]
    print(f"Totally get {batch_size} latents, with shape of {latent[0].shape}")
    
    return latent, inversion_latents

def ori_latent_2_pro_image(cfg, img, epsilon, model, latents, contexts, guiding_scale=0):
    batch_size = img.shape[0]
    latents = latents + epsilon
    latent_pro = latents.clone()
    for ind, t in enumerate(model.scheduler.timesteps[1 + cfg.DIFFUSION.START_STEP - 1:]):
        latents = diffusion_step(model, latents, contexts[ind], t, guiding_scale)
    pro_img = model.vae.decode(1 / 0.18215 * latents)['sample']
    print(f"Totally get {pro_img.shape[0]} pro_imgs, with shape of {pro_img[0].shape}")

    return pro_img, latent_pro

def latent_2_image(cfg, model, latents, contexts, guiding_scale=0):
    # all_mid_feat = []
    try:
        unet_param = next(model.unet.parameters())
        target_device = unet_param.device
        target_dtype = unet_param.dtype
        latents = latents.to(device=target_device, dtype=target_dtype)
        # contexts may be a tensor or batch of tensors
        if isinstance(contexts, torch.Tensor):
            contexts = contexts.to(device=target_device, dtype=target_dtype)
        else:
            # if contexts is e.g. a list/tuple, convert elements
            contexts = type(contexts)(c.to(device=target_device, dtype=target_dtype) if isinstance(c, torch.Tensor) else c for c in contexts)
    except Exception:
        # fallback: keep original if anything unexpected
        pass

    for ind, t in enumerate(model.scheduler.timesteps[1 + cfg.DIFFUSION.START_STEP - 1:]):
        # all_layer_outputs = {}
        # def hook_fn(module, input, output):
        # # 将每一层的名称和输出保存到字典中
        #     layer_name = module.__class__.__name__  # 获取当前层的名字
        #     all_layer_outputs[layer_name] = output
        # for name, layer in model.unet.named_modules():
        #     if isinstance(layer, torch.nn.Module):
        #         layer.register_forward_hook(hook_fn)
        latents = diffusion_step(model, latents, contexts[ind], t, guiding_scale)
        # all_mid_feat.append(all_layer_outputs['UNetMidBlock2DCrossAttn'])
    if cfg.MODEL.DIST_TRAIN:
        img = model.vae.module.decode(1 / 0.18215 * latents)['sample']
    else:
        img = model.vae.decode(1 / 0.18215 * latents)['sample']

    try:
        img = img.to(dtype=torch.float32)
    except Exception:
        img = img.float()

    # return img, all_mid_feat
    return img

def compute_loss(logger, batch_latent, ori_latent, out_image, img, pro_feat, ori_feat, pro_feat_bn=0, ori_feat_bn=0, clip_ori_feat=0, clip_pro_feat=0, mid_feat=0):
    loss_1 = torch.nn.L1Loss(reduction='mean')(batch_latent, ori_latent)
    print("loss_1: MSELoss between ori_lt & pro_lt: {:.4f}".format(loss_1))

    loss_2 = torch.nn.L1Loss(reduction='mean')(out_image, img)
    loss_MSE = torch.nn.MSELoss(reduction='mean')(out_image, img)
    print("loss_2: MSELoss between out_image & img: {:.4f}".format(loss_2))

    loss_3 = torch.nn.MSELoss(reduction='mean')(pro_feat, ori_feat)
    logger.info("loss_3: MSELoss between pro_feat & ori_feat: {:.4f}".format(loss_3))

    # loss_7 = torch.nn.MSELoss(reduction='mean')(pro_feat_bn, ori_feat_bn)
    # logger.info("loss_3: MSELoss between pro_feat_bn & ori_feat_bn: {:.4f}".format(loss_7))
    
    # loss_2 = torch.nn.L1Loss(reduction='mean')(out_image, img)
    # loss_cross_att = img1_true_label_attention_map.var() + img2_true_label_attention_map.var()
    #loss_3 = reverse_tri_loss(CLIP_pro_output.vision_model_output.pooler_output, target)
    # loss_4 = reverse_tri_loss(pro_feat_bn, target)
    # # print("loss_4: After BN Feature reverse tri: {:.4f}".format(loss_4))
    # logger.info("loss_4: After BN Feature reverse tri: {:.4f}".format(loss_4))
    # loss_4 = reverse_tri_loss(pro_feat, target)
    # # print("loss_4: Before BN Feature reverse tri: {:.4f}".format(loss_4))
    # logger.info("loss_4: Before BN Feature reverse tri: {:.4f}".format(loss_4))
    # print("loss_3: adv_image in sup_model: {:.4f}".format(loss_3))
    # logger.info("loss_3: adv_image in sup_model: {:.4f}".format(loss_3))

    loss_5 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(pro_feat, ori_feat, (-1 * torch.ones(pro_feat.shape[0])).to('cuda'))
    print("loss_5: Before BN Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_5))
    logger.info("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_5))

    # loss_6 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(pro_feat_bn, ori_feat_bn, (-1 * torch.ones(pro_feat.shape[0])).to('cuda'))
    # print("loss_6: After BN Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_6))
    # logger.info("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_6))

    loss_10 = torch.max(torch.abs(batch_latent - ori_latent))
    print("loss_10: MaxABSLoss between ori_lt & pro_lt: {:.4f}".format(loss_10))

    # loss_11 = []
    # for a in mid_feat:
    #     # 计算 a[:2] 和 a[2:4] 之间的 MSE
    #     a_flat1 = a[:2].view(a[:2].shape[0], -1)  # 展平最后两个维度
    #     a_flat2 = a[2:4].view(a[2:4].shape[0], -1)  # 展平最后两个维度
    #     loss = torch.nn.CosineEmbeddingLoss(reduction='mean')(a_flat1, a_flat2, (-1 * torch.ones(a_flat1.shape[0])).to('cuda'))
    #     loss_11.append(loss)
    # loss_11 = sum(loss_11) / len(loss_11)

    # loss_clip = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(clip_ori_feat, clip_pro_feat, (-1 * torch.ones(clip_ori_feat.shape[0])).to('cuda'))
    # print("loss_clip: CLIP Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_clip))
    # logger.info("loss_clip: CLIP Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_clip))

    # loss_clip_rn = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(clip_ori_feat_rn, clip_pro_feat_rn, (-1 * torch.ones(clip_ori_feat_rn.shape[0])).to('cuda'))
    # print("loss_clip_rn: CLIP_rn Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_clip_rn))
    # logger.info("loss_clip_rn: CLIP_rn Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_clip_rn))

    # def adaptive_weighted_mse_loss(original, generated, epsilon=1e-8, temperature=1.0):
    #     """
    #     Args:
    #         original: 原始图像 (Tensor, shape: [B, C, H, W])
    #         generated: 生成图像 (Tensor, shape: [B, C, H, W])
    #         epsilon: 防止除以零的小常数
    #         temperature: 控制权重分布的平滑度（值越大，权重越均匀）
    #     """
    #     # 计算原始图像各通道的均值（按 batch 和空间维度求平均）
    #     mean_r = torch.mean(original[:, 0, :, :])  # R 通道均值
    #     mean_g = torch.mean(original[:, 1, :, :])  # G 通道均值
    #     mean_b = torch.mean(original[:, 2, :, :])  # B 通道均值
        
    #     # 使用 softmax 归一化均值，得到权重
    #     mean_values = torch.stack([mean_r, mean_g, mean_b])
    #     weights = torch.softmax(mean_values / temperature, dim=0)
        
    #     # 按通道计算 MSE 并加权
    #     mse_r = torch.mean((original[:, 0] - generated[:, 0]) ** 2)
    #     mse_g = torch.mean((original[:, 1] - generated[:, 1]) ** 2)
    #     mse_b = torch.mean((original[:, 2] - generated[:, 2]) ** 2)
        
    #     loss = weights[0] * mse_r + weights[1] * mse_g + weights[2] * mse_b
    #     return loss
    # loss_adaptive_mse = adaptive_weighted_mse_loss(out_image, img)
    # def lab_loss(original, generated, alpha=0.7):
    #     """
    #     Args:
    #         alpha: 控制明度（L）和颜色（a, b）的权重（0.5 表示平衡）
    #     """
    #     # 转换到 Lab 颜色空间
    #     original_lab = T.functional.rgb_to_lab(original)
    #     generated_lab = T.functional.rgb_to_lab(generated)
        
    #     # 拆分 L, a, b 通道
    #     l_orig, a_orig, b_orig = torch.split(original_lab, 1, dim=1)
    #     l_gen, a_gen, b_gen = torch.split(generated_lab, 1, dim=1)
        
    #     # 计算损失
    #     loss_l = torch.mean((l_orig - l_gen) ** 2)           # 明度损失
    #     loss_ab = torch.mean((a_orig - a_gen) ** 2 + (b_orig - b_gen) ** 2)  # 颜色损失
    #     loss = (1 - alpha) * loss_l + alpha * loss_ab
    #     return loss
    # loss_lab_loss = lab_loss(img, out_image)
    # gray_out = T.Grayscale(num_output_channels=1)(out_image)
    # gray_ori = T.Grayscale(num_output_channels=1)(img)
    # gray_loss = torch.nn.MSELoss(reduction='mean')(gray_ori, gray_out)


    # loss = loss_5 + 10 * loss_2 + 10 * loss_1 + 2 * loss_6 - 0.1 * loss_3  # 1e-1 30iter train_new

    # loss = 3 * loss_5 + 5 * loss_6 + 30 * loss_2 + 30 * loss_1  - 0.2 * loss_7 

    # loss = loss_5 +  5 * controller.loss + 5 * loss_2 + 5 * loss_1 - 0.1 * loss_3 # 1e-1 20iter train_new_cont

    # loss = 10 * loss_2 + 5 * loss_MSE + loss_1 + loss_6 - 0.1 * loss_7 + 30 * controller.loss

    # loss = loss_5 + 5 * controller.loss + 3 * loss_2 + 2 * loss_1 # 1e-1 30iter train_cont

    # loss_4 = 1.5 * loss_5 + loss_6 + 10 * controller.loss + 10 * loss_2 + 10 * loss_1 - 0.15 * loss_3 # 1e-1 20iter train_loss4.pth market

    # loss_4 = 1.5 * loss_5 + loss_6 + 10 * controller.loss + 10 * loss_2 + 10 * loss_1 - 0.15 * loss_3 # 1e-1 20iter train_loss4.pth msmt

    # loss_5 = 1.5 * loss_5 + loss_11 + 10 * controller.loss + 10 * loss_2 + 10 * loss_1 - 0.15 * loss_3 # 1e-1 20iter train_loss5.pth

    # loss_6 = 1.5 * loss_5 + loss_clip + 10 * controller.loss + 10 * loss_2 + 10 * loss_1 - 0.15 * loss_3 # 1e-1 20iter train_loss6.pth

    # loss_7 = 1.5 * loss_5 + 0.5 * loss_6 + 0.5 * loss_clip + 10 * controller.loss + 15 * loss_2 + 5 * loss_1 - 0.1 * loss_3 # 1e-1 20iter train_loss7.pth

    # loss_8 = 1.5 * loss_5 + 0.5 * loss_6 + 0.5 * loss_clip + 0.5 * loss_clip_rn + 10 * controller.loss + 15 * loss_2 + 5 * loss_1 - 0.1 * loss_3 # 1e-1 20iter train_loss8.pth

    # loss_9 = 1.5 * loss_5 + loss_6 + loss_clip + 10 * controller.loss + 15 * loss_2 + 10 * loss_1 - 0.1 * loss_3 # 1e-1 20iter train_loss9.pth

    # loss_10 = loss_5 + 0.5 * loss_6 + 7.5 * controller.loss + 7.5 * loss_2 + 7.5 * loss_MSE + 7.5 * loss_1- 0.1 * loss_3 # 1e-1 20iter train_loss10.pth loss_5 + loss_6 + 0.5 * loss_clip + 10 * controller.loss + 5 * loss_2 + 5 * loss_MSE + 15 * loss_1- 0.1 * loss_3

    # loss_11 = loss_5 + 0.5 * loss_6 + 0.5*loss_clip + 7.5 * controller.loss + 7.5 * loss_2 + 7.5 * loss_MSE + 7.5 * loss_1- 0.1 * loss_3 # 1e-1 20iter train_loss11.pth 

    # loss_12 = 2 * loss_5 + 0.5 * loss_6 + loss_clip + 10 * controller.loss + 10 * loss_2 + 5 * loss_MSE + 10 * loss_1 # 1e-1 30iter train_loss12.pth 

    # loss_13 = 2 * loss_5 + 0.5 * loss_6 + loss_clip + 10 * controller.loss + 10 * loss_2 + 5 * loss_MSE + 10 * loss_1 # 1e-1 20iter train_loss13.pth 

    # loss_14 = 1.5 * loss_5 + 0.5 * loss_6 + 0.5 * loss_clip + 10 * controller.loss + 15 * loss_2 + 5 * loss_1 - 0.1 * loss_3 # 1e-1 30iter train_loss14.pth

    # loss_15 = 2 * loss_5 + 0.5 * loss_6 + 5 * controller.loss + 3 * loss_2 # 3e-2 15iter train_loss15.pth 2 * loss_5 + 0.5 * loss_6 + loss_clip + 20 * loss_2 + 5 * loss_MSE + 10 * loss_1

    # loss_16 = 2 * loss_5 + loss_1 + loss_2 + loss_MSE + 0.1 * loss_clip # 5e-2 10iter train_loss16.pth  备用 2.5 * loss_5 + 1.5*loss_1 + 1.5*loss_2 + loss_MSE + 0.1 * loss_clip

    # loss_17 = 5 * loss_5 + loss_1 + loss_2 + 0.1*loss_clip# 2e-2 10iter train_loss17.pth  

    loss_18 = 5 * loss_5 + loss_1 + loss_2  # 2e-2 10iter train_loss18.pth  

    # loss_19 = 5 * loss_5 + loss_1 + loss_2 + 0.1 * loss_clip# 2e-2 10iter train_loss18.pth  
    logger.info("loss_18 = 5 * loss_5 + loss_1 + loss_2")

    return loss_18

def norm_img(cfg, pro_img):

    out_image = (pro_img / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=out_image.dtype, device=out_image.device)
    std = torch.as_tensor(cfg.INPUT.PIXEL_STD, dtype=out_image.dtype, device=out_image.device)
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)

    return out_image

def psnr_ssim(pro_img, img, all_SSIM, all_PSNR):
    ssim = SSIM()((pro_img*0.5+0.5).detach().cpu(), (img*0.5+0.5).detach().cpu())
    print(ssim)
    all_SSIM.append(ssim)

    a = (pro_img*0.5+0.5).detach().cpu()
    b = (img*0.5+0.5).detach().cpu()
    psnr_value = 10 * torch.log10((1.0 ** 2) / torch.nn.functional.mse_loss(a,b))
    print(psnr_value)
    all_PSNR.append(psnr_value)

def generate_trainset(cfg,
             model_reid,
             model_supervisor,
             train_loader,
            local_rank):
             
    device = "cuda"
    
    # diffusion & CLIP init     
    ldm_stable = StableDiffusionPipeline.from_pretrained(cfg.MODEL.PRETRAINED_DIFFUSION_PATH).to(local_rank)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    ldm_stable.scheduler.set_timesteps(cfg.DIFFUSION.NUM_INFERENCE_STEPS)
    
    CLIP_model = CLIPModel.from_pretrained("/data1/mqx_log/clip-vit-large-patch14").to(local_rank)
    CLIP_processor = CLIPProcessor.from_pretrained("/data1/mqx_log/clip-vit-large-patch14")

    # hyper-param setting
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS
    num_gpus = torch.cuda.device_count()
    batch_size_per_gpu = cfg.SOLVER.IMS_PER_BATCH // num_gpus
    
    # logger 
    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    # deploy model to GPU
    if device:
        model_reid.to(device=local_rank)
        model_reid.eval()
        model_supervisor.to(device=local_rank)
        if cfg.MODEL.DIST_TRAIN:
            model_supervisor = torch.nn.parallel.DistributedDataParallel(model_supervisor, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
        model_supervisor.eval()
        if cfg.MODEL.DIST_TRAIN:
            ldm_stable.unet = torch.nn.parallel.DistributedDataParallel(ldm_stable.unet, device_ids=[local_rank], find_unused_parameters=False)
            ldm_stable.vae = torch.nn.parallel.DistributedDataParallel(ldm_stable.vae, device_ids=[local_rank], find_unused_parameters=False)
            ldm_stable.text_encoder = torch.nn.parallel.DistributedDataParallel(ldm_stable.text_encoder, device_ids=[local_rank], find_unused_parameters=False)
        ldm_stable.unet.requires_grad_(False)
        ldm_stable.vae.requires_grad_(False)
        ldm_stable.text_encoder.requires_grad_(False)
        # CLIP_model.eval()
    
    # list for adv_train imgs
    all_pro_image = {}

    all_PSNR = []
    all_SSIM = []

    # stage 1
    # Adv_train set generation
    for n_iter, (img, vid, target_cam, target_view, img_path, latent_pre, description) in enumerate(train_loader) :
        
        print('\n')
        print('--------------------------------------------------------------------------------')
        print('STAGE 1 : ADV-train set generation of BATCH {}'.format(n_iter))
        logger.info('STAGE 1 : ADV-train set generation of BATCH {}'.format(n_iter))
        print('--------------------------------------------------------------------------------')
        print('\n')

        # tensor to GPU
        img = img.to(device)
        target = vid.to(device)
        target_cam = target_cam.to(device)
        target_view = target_view.to(device)
        
        # img 2 latent
        batch_prompt = list(description)
        batch_latent = latent_pre.to(device)
        # batch_latent = get_batch_latent(cfg, img, batch_prompt, ldm_stable)
        ori_latent = latent_pre.to(device).requires_grad_(True)
        batch_latent.requires_grad_(True)

        optimizer_latent = torch.optim.AdamW(params=[{"params": [batch_latent]}], lr=2e-2, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        logger.info("optim is {}".format(optimizer_latent))
        # Controller
        # controller = AttentionControlEdit(cfg, cfg.DIFFUSION.NUM_INFERENCE_STEPS, 1., cfg.INPUT.SIZE_TRAIN)
        # register_attention_control(ldm_stable, controller)

        # text embedding & uncond embedding & DDIM context
        max_length = 77
        batch_uncond_input = ldm_stable.tokenizer(
            [""] * batch_size_per_gpu, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        bathc_uncond_embeddings = ldm_stable.text_encoder(batch_uncond_input.input_ids.to(ldm_stable.device))[0].detach()
        batch_text_input = ldm_stable.tokenizer(
            batch_prompt,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        bacth_text_embeddings = ldm_stable.text_encoder(batch_text_input.input_ids.to(ldm_stable.device))[0].detach()
        batch_uncond_text = torch.cat((bathc_uncond_embeddings,bacth_text_embeddings), dim=0)
        batch_context = batch_uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)

        # for self attetion control
        # uncond_text = torch.cat((torch.cat((bathc_uncond_embeddings, bathc_uncond_embeddings), dim=0) , torch.cat((bacth_text_embeddings, bacth_text_embeddings), dim=0)), dim=0)
        # batch_uncond_text = uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)
        
        # Strat optimizing latent
        iterations = 10
        logger.info("iterations = {}".format(iterations))
        pbar = tqdm(range(iterations), desc="Iterations")
        start_time = time.time()

        for step, _ in enumerate(pbar):

            # for self-attention control
            latent = torch.cat((batch_latent, ori_latent), dim=0)

            optimizer_latent.zero_grad()
            
            pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)
            # pro_img = latent_2_image(cfg, ldm_stable, latent, batch_uncond_text)[:batch_size_per_gpu]
            # pro_img, mid_feat = latent_2_image(cfg, ldm_stable, latent, batch_uncond_text)
            # pro_img = pro_img[:batch_size_per_gpu]

            save_image(img, pro_img, n_iter, cfg, step, False, 'train')

            out_image = norm_img(cfg, pro_img)

            pro_feat_bn, pro_feat, _ = model_supervisor(out_image, target, cam_label=target_cam, view_label=target_view)
            ori_feat_bn, ori_feat, _ = model_supervisor(img, target, cam_label=target_cam, view_label=target_view)

            loss = compute_loss(logger, batch_latent, ori_latent, out_image, img, pro_feat, ori_feat)

            pbar.set_postfix_str(
                f"loss: {loss.item():.5f}")
            
            loss.backward()
            
            optimizer_latent.step()

        with torch.no_grad():
            pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)
            # pro_img = latent_2_image(cfg, ldm_stable, latent, batch_uncond_text)[:batch_size_per_gpu]
            
            # pro_img, mid_feat = latent_2_image(cfg, ldm_stable, latent, batch_uncond_text)
            # pro_img = pro_img[:batch_size_per_gpu]
            save_image(img, pro_img, n_iter, cfg, step+1, False, 'train')

            psnr_ssim(pro_img, img, all_SSIM, all_PSNR)
            
        for i in range(img.shape[0]):
            all_pro_image[img_path[i]]=pro_img[i].detach().cpu()

        del pro_img
            
        end_time = time.time()
        print('\n')
        print("Each batch need {:.4f}s".format(end_time-start_time))
        print('\n')

        # if n_iter % 1000 == 0:
        #     if cfg.MODEL.DIST_TRAIN:
        #         path = '/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/pro_train_loss18_iter_' + str(n_iter) + '_' + str(dist.get_rank()) + '.pth'
        #         torch.save(all_pro_image, path)
        #         logger.info("save pth in {}".format(path))
        #     else:
        #         path = '/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/pro_train_loss18_iter_' + str(n_iter) + '.pth'
        #         torch.save(all_pro_image, path)
        #         logger.info("save pth in {}".format(path))

    logger.info('--------------------------------------------------------------------------------')
    logger.info("Finish ADV-train set generation")
    logger.info('--------------------------------------------------------------------------------')
    logger.info("mean SSIM is {:.4f}".format(torch.mean(torch.stack(all_SSIM, dim=0))))
    logger.info("mean PSNR is {:.4f}".format(torch.mean(torch.stack(all_PSNR, dim=0))))
    if cfg.MODEL.DIST_TRAIN:
        path = '/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW0.5_pro_train_loss18_' + str(dist.get_rank()) + '.pth'
        torch.save(all_pro_image, path)
        logger.info("save pth in {}".format(path))
    else:
        path = '/data1/mqx_log/Privacy/PPReID/market/dataset_generation/boost/train'  + '.pth'
        torch.save(all_pro_image, path)
        logger.info("save pth in {}".format(path))


def generate_testset(cfg,
                 model_reid,
                 model_supervisor,
                 val_loader,
                 train_loader,
                 query_loader,
                 gallery_loader,
                 num_query, local_rank, model_msi, model_transformer_base, model_transreid, model_AGW, model_SOLIDER, model_PCB, model_ResNet, query_loader_st2, gallery_loader_st2):
    
    device = "cuda"
    flag = True
    ldm_stable = StableDiffusionPipeline.from_pretrained(cfg.MODEL.PRETRAINED_DIFFUSION_PATH).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    ldm_stable.scheduler.set_timesteps(cfg.DIFFUSION.NUM_INFERENCE_STEPS)
    CLIP_model = CLIPModel.from_pretrained("/data1/mqx_log/clip-vit-large-patch14").to(device)
    CLIP_processor = CLIPProcessor.from_pretrained("/data1/mqx_log/clip-vit-large-patch14")
    CLIP_model.eval()
    
    model_name = "RN50"
    # CLIP_rn, _ = clip.load(model_name, device="cuda" if torch.cuda.is_available() else "cpu")


    num_gpus = torch.cuda.device_count()
    batch_size_per_gpu = cfg.TEST.IMS_PER_BATCH // num_gpus

    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator_1 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_2 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_3 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_4 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_5 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_6 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_7 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_8 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator_1.reset()
    evaluator_2.reset()
    evaluator_3.reset()
    evaluator_4.reset()
    evaluator_5.reset()
    evaluator_6.reset()
    evaluator_7.reset()
    evaluator_8.reset()

    if device:
        model_reid.to(local_rank)
        model_reid.eval()
        model_supervisor.to(local_rank)
        model_msi.to(local_rank)
        model_msi.eval()
        model_transformer_base.to(local_rank)
        model_transreid.to(local_rank)
        model_AGW.to(local_rank)
        model_SOLIDER.to(local_rank)
        model_PCB.to(local_rank)
        model_ResNet.to(local_rank)
        model_transformer_base.eval()
        model_transreid.eval()
        model_AGW.eval()
        model_SOLIDER.eval()
        model_PCB.eval()
        model_ResNet.eval()
        if cfg.MODEL.DIST_TRAIN:
            model_supervisor = torch.nn.parallel.DistributedDataParallel(model_supervisor, device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
        model_supervisor.eval()
        if cfg.MODEL.DIST_TRAIN:
            ldm_stable.unet = torch.nn.parallel.DistributedDataParallel(ldm_stable.unet, device_ids=[local_rank], find_unused_parameters=False)
            ldm_stable.vae = torch.nn.parallel.DistributedDataParallel(ldm_stable.vae, device_ids=[local_rank], find_unused_parameters=False)
            ldm_stable.text_encoder = torch.nn.parallel.DistributedDataParallel(ldm_stable.text_encoder, device_ids=[local_rank], find_unused_parameters=False)
        ldm_stable.unet.requires_grad_(False)
        ldm_stable.vae.requires_grad_(False)
        ldm_stable.text_encoder.requires_grad_(False)

    all_pro_query = {}
    # all_pro_gallery = {}
    
    all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/boost/gallery.pth")

    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_1/gallery.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/gallery/pro_gallery_loss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/pro_gallery_msmtloss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/dataset_generation/ablation_new/no_IQ/gallery.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/dataset_generation/ablation_new/only_IQ/gallery.pth")
    all_PSNR = []
    all_SSIM = []
    all_SIM = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath, latent_pre, description) in enumerate(query_loader):

        print('\n')
        print('--------------------------------------------------------------------------------')
        print('ADV-test set generation of BATCH {}'.format(n_iter))
        logger.info('ADV-query set generation of BATCH {}'.format(n_iter))
        print('--------------------------------------------------------------------------------')
        print('\n')
        start_time = time.time()
        print(imgpath)
        img = img.to(device)
        vid = torch.tensor(pid, dtype=torch.int64).to(device)
        camids = camids.to(device)
        target_view = target_view.to(device)

        # img 2 latent
        batch_prompt = list(description)
        batch_latent = latent_pre.to(device)
        # batch_latent = get_batch_latent(cfg, img, batch_prompt, ldm_stable)
        ori_latent = latent_pre.to(device).requires_grad_(True)
        batch_latent.requires_grad_(True)

        optimizer_latent = torch.optim.AdamW(params=[{"params": [batch_latent]}], lr=2e-2, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        logger.info("optim is {}".format(optimizer_latent))
        # Controller
       
        # register_attention_control(ldm_stable, controller)
        # if cfg.MODEL.DIST_TRAIN:
        #     controller = AttentionControlEdit_DDP(cfg, cfg.DIFFUSION.NUM_INFERENCE_STEPS, 1., cfg.INPUT.SIZE_TRAIN)
        #     register_attention_control_DDP(ldm_stable, controller)

        # text embedding & uncond embedding & DDIM context
        max_length = 77
        batch_uncond_input = ldm_stable.tokenizer(
            [""] * batch_size_per_gpu, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        bathc_uncond_embeddings = ldm_stable.text_encoder(batch_uncond_input.input_ids.to(ldm_stable.device))[0].detach()
        batch_text_input = ldm_stable.tokenizer(
            batch_prompt,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        bacth_text_embeddings = ldm_stable.text_encoder(batch_text_input.input_ids.to(ldm_stable.device))[0].detach()
        batch_uncond_text = torch.cat((bathc_uncond_embeddings,bacth_text_embeddings), dim=0)
        batch_context = batch_uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)

        # for self attetion control
        # uncond_text = torch.cat((torch.cat((bathc_uncond_embeddings, bathc_uncond_embeddings), dim=0) , torch.cat((bacth_text_embeddings, bacth_text_embeddings), dim=0)), dim=0)
        # batch_uncond_text = uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)
        
        # Strat optimizing latent
        iterations = 10
        logger.info("iterations = {}".format(iterations))
        pbar = tqdm(range(iterations), desc="Iterations")
        start_time = time.time()

        for step, _ in enumerate(pbar):



            # for self-attention control
            latent = torch.cat((batch_latent, ori_latent), dim=0)

            optimizer_latent.zero_grad()

            pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)
            # pro_img = latent_2_image(cfg, ldm_stable, latent, batch_uncond_text)[:batch_size_per_gpu]

            save_image(img, pro_img, n_iter, cfg, step, True, 'query')

            out_image = norm_img(cfg, pro_img)

            pro_feat_bn, pro_feat, _ = model_supervisor(out_image, vid, cam_label=camids, view_label=target_view)
            ori_feat_bn, ori_feat, _ = model_supervisor(img, vid, cam_label=camids, view_label=target_view)

            # pro_feat = model_msi(out_image)
            # ori_feat = model_msi(img)

            # pro_feat= model_transreid(out_image, vid, cam_label=camids, view_label=target_view)
            # ori_feat= model_transreid(img, vid, cam_label=camids, view_label=target_view)

            # pro_feat = model_transformer_base(out_image, vid, cam_label=camids, view_label=target_view)
            # ori_feat = model_transformer_base(img, vid, cam_label=camids, view_label=target_view)

            # pro_feat = model_AGW(out_image)
            # ori_feat = model_AGW(img)

            # pro_feat, _ = model_SOLIDER(out_image, vid, cam_label=camids, view_label=target_view)
            # ori_feat, _ = model_SOLIDER(img, vid, cam_label=camids, view_label=target_view)

            # text_input = CLIP_processor(text=description, return_tensors="pt", padding=True, truncation=True).to(device)
            # CLIP_ori_output = CLIP_model(input_ids=text_input.input_ids, pixel_values=img, interpolate_pos_encoding=True, output_hidden_states=True)
            # CLIP_pro_output = CLIP_model(input_ids=text_input.input_ids, pixel_values=out_image, interpolate_pos_encoding=True, output_hidden_states=True)

            # clip_ori_feat = CLIP_ori_output.vision_model_output.pooler_output
            # clip_pro_feat = CLIP_pro_output.vision_model_output.pooler_output

            loss = compute_loss(logger, batch_latent, ori_latent, out_image, img, pro_feat, ori_feat)

            pbar.set_postfix_str(
                f"loss: {loss.item():.5f}")
            
            loss.backward()
            
            optimizer_latent.step()

        with torch.no_grad():
            pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)
            # pro_img = latent_2_image(cfg, ldm_stable, latent, batch_uncond_text)[:batch_size_per_gpu]

            save_image(img, pro_img, n_iter, cfg, step+1, True, 'query')

            psnr_ssim(pro_img, img, all_SSIM, all_PSNR)
            
        for i in range(img.shape[0]):
            all_pro_query[imgpath[i]]=pro_img[i].detach().cpu()
            
        end_time = time.time()
        print('\n')
        print("Each batch need {:.4f}s".format(end_time-start_time))
        print('\n')

        # feat_pro, _, _ = model_supervisor(img, cam_label=camids, view_label=target_view)
        
        # feat_ori, _, _ = model_supervisor(img, cam_label=camids, view_label=target_view)
            
        # feat_sup, _, _ = model_supervisor(pro_img, cam_label=camids, view_label=target_view)
        
        # evaluator_1.update((feat_pro, pid, camid))
        # evaluator_2.update((feat_ori, pid, camid))
        # evaluator_3.update((feat_sup, pid, camid))

        del pro_img

        end_time = time.time()
        all_time = end_time - start_time
        print('\n')
        print("Each batch need {:.4f}s".format(end_time-start_time))
        print('\n')

        # if n_iter % 1000 == 0:
        #     if cfg.MODEL.DIST_TRAIN:
        #         path = '/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/pro_query_loss18_iter_' + str(n_iter) + '_' + str(dist.get_rank()) + '.pth'
        #         torch.save(all_pro_query, path)
        #         logger.info("save pth in {}".format(path))
        #     else:
        #         path = '/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/pro_query_loss18_iter_' + str(n_iter) + '.pth'
        #         torch.save(all_pro_query, path)
        #         logger.info("save pth in {}".format(path))

    logger.info('--------------------------------------------------------------------------------')
    logger.info("Finish ADV-query set generation")
    logger.info('--------------------------------------------------------------------------------')
    logger.info("mean SSIM is {:.4f}".format(torch.mean(torch.stack(all_SSIM, dim=0))))
    logger.info("mean PSNR is {:.4f}".format(torch.mean(torch.stack(all_PSNR, dim=0))))

    if cfg.MODEL.DIST_TRAIN:
        path = '/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW0.5_pro_query_loss18_' + str(dist.get_rank()) + '.pth'
        torch.save(all_pro_query, path)
        logger.info("save pth in {}".format(path))
    else:
        path = '/data1/mqx_log/Privacy/PPReID/market/dataset_generation/boost/query'  + '.pth'
        torch.save(all_pro_query, path)
        logger.info("save pth in {}".format(path))

    for n_iter, (img, pid, camid, camids, target_view, imgpath, latent_pre, description) in enumerate(gallery_loader):
        # if n_iter > 4637:
        print(imgpath)
        print('\n')
        print('--------------------------------------------------------------------------------')
        print('ADV-test set generation of BATCH {}'.format(n_iter))
        logger.info('ADV-gallery set generation of BATCH {}'.format(n_iter))
        print('--------------------------------------------------------------------------------')
        print('\n')
        time_1 = time.time()
        
        img = img.to(device)
        vid = torch.tensor(pid, dtype=torch.int64).to(device)
        camids = camids.to(device)
        target_view = target_view.to(device)

        
        # img 2 latent
        batch_prompt = list(description)
        batch_latent = latent_pre.to(device)
        # batch_latent, inversion_latents = get_batch_latent(cfg, img, batch_prompt, ldm_stable)
        ori_latent = latent_pre.to(device).requires_grad_(True)
        latent = batch_latent.clone()
        batch_latent.requires_grad_(True)
        

        optimizer_latent = torch.optim.AdamW(params=[{"params": [batch_latent]}], lr=2e-2, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        logger.info("optim is {}".format(optimizer_latent))
        # Controller
        
        # if cfg.MODEL.DIST_TRAIN:
        #     controller = AttentionControlEdit_DDP(cfg, cfg.DIFFUSION.NUM_INFERENCE_STEPS, 1., cfg.INPUT.SIZE_TRAIN)
        #     register_attention_control_DDP(ldm_stable, controller)

        # text embedding & uncond embedding & DDIM context
        max_length = 77
        batch_uncond_input = ldm_stable.tokenizer(
            [""] * batch_size_per_gpu, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        bathc_uncond_embeddings = ldm_stable.text_encoder(batch_uncond_input.input_ids.to(ldm_stable.device))[0].detach()
        batch_text_input = ldm_stable.tokenizer(
            batch_prompt,
            padding="max_length",
            max_length=ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        bacth_text_embeddings = ldm_stable.text_encoder(batch_text_input.input_ids.to(ldm_stable.device))[0].detach()
        batch_uncond_text = torch.cat((bathc_uncond_embeddings,bacth_text_embeddings), dim=0)
        batch_context = batch_uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)

        ''' IF NOT optimizie unconditional embedding'''
        # # for self attetion control
        # uncond_text = torch.cat((torch.cat((bathc_uncond_embeddings, bathc_uncond_embeddings), dim=0) , torch.cat((bacth_text_embeddings, bacth_text_embeddings), dim=0)), dim=0)
        # batch_context = uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)
        # register_attention_control(ldm_stable, controller)



        ''' IF optimizie unconditional embedding'''
        # all_uncond_emb = []
        # bathc_uncond_embeddings.requires_grad_(True)
        # optimizer = optim.AdamW([bathc_uncond_embeddings], lr=1e-1)
        # loss_func = torch.nn.MSELoss()

        # batch_uncond_text = torch.cat((bathc_uncond_embeddings,bacth_text_embeddings), dim=0)

        # for ind, t in enumerate(tqdm(ldm_stable.scheduler.timesteps[1 + cfg.DIFFUSION.START_STEP - 1:], desc="Optimize_uncond_embed")):
        #     # t: [201, 151, 101, 51, 1]
        #     for _ in range(10 + 2 * ind):
        #         out_latents = diffusion_step(ldm_stable, latent, batch_uncond_text, t, 0)
        #         optimizer.zero_grad()
        #         loss = loss_func(out_latents, inversion_latents[cfg.DIFFUSION.START_STEP - 1 + ind + 1])
        #         loss.backward()
        #         print(loss)
        #         optimizer.step()
        #         batch_uncond_text = torch.cat((bathc_uncond_embeddings,bacth_text_embeddings), dim=0)
        #     with torch.no_grad():
        #         latent = diffusion_step(ldm_stable, latent, batch_uncond_text, t, 0).detach()
        #         all_uncond_emb.append(bathc_uncond_embeddings.detach().clone())
    
        # # for self attetion control
        # batch_context = []
        # for i in range(5):
        #     batch_context.append(torch.cat((torch.cat((all_uncond_emb[i], all_uncond_emb[i]), dim=0) , torch.cat((bacth_text_embeddings, bacth_text_embeddings), dim=0)), dim=0)) 
        # batch_context = torch.stack(batch_context, dim=0)

        # register_attention_control(ldm_stable, controller)

        # bathc_uncond_embeddings.requires_grad_(False)

        # Strat optimizing latent
        iterations = 10
        logger.info("iterations = {}".format(iterations))
        pbar = tqdm(range(iterations), desc="Iterations")
        start_time = time.time()

        for step, _ in enumerate(pbar):
            

            # for self-attention control
            latent = torch.cat((batch_latent, ori_latent), dim=0)

            optimizer_latent.zero_grad()
            
            pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)
            # pro_img = latent_2_image(cfg, ldm_stable, latent, batch_context)[:batch_size_per_gpu]

            # # Mid feat loss
            # pro_img, mid_feat = latent_2_image(cfg, ldm_stable, latent, batch_context)
            # pro_img = pro_img[:batch_size_per_gpu]

            save_image(img, pro_img, n_iter, cfg, step, True, 'gallery')

            out_image = norm_img(cfg, pro_img)

            pro_feat_bn, pro_feat, _ = model_supervisor(out_image, vid, cam_label=camids, view_label=target_view)
            ori_feat_bn, ori_feat, _ = model_supervisor(img, vid, cam_label=camids, view_label=target_view)

            # pro_feat = model_msi(out_image)
            # ori_feat = model_msi(img)

            # pro_feat= model_transreid(out_image, vid, cam_label=camids, view_label=target_view)
            # ori_feat= model_transreid(img, vid, cam_label=camids, view_label=target_view)

            # pro_feat = model_transformer_base(out_image, vid, cam_label=camids, view_label=target_view)
            # ori_feat = model_transformer_base(img, vid, cam_label=camids, view_label=target_view)

            # pro_feat = model_AGW(out_image)
            # ori_feat = model_AGW(img)

            # pro_feat, _ = model_SOLIDER(out_image, vid, cam_label=camids, view_label=target_view)
            # ori_feat, _ = model_SOLIDER(img, vid, cam_label=camids, view_label=target_view)
            
            # text_input = CLIP_processor(text=description, return_tensors="pt", padding=True, truncation=True).to(device)
            # CLIP_ori_output = CLIP_model(input_ids=text_input.input_ids, pixel_values=img, interpolate_pos_encoding=True, output_hidden_states=True)
            # CLIP_pro_output = CLIP_model(input_ids=text_input.input_ids, pixel_values=out_image, interpolate_pos_encoding=True, output_hidden_states=True)

            # clip_ori_feat = CLIP_ori_output.vision_model_output.pooler_output
            # clip_pro_feat = CLIP_pro_output.vision_model_output.pooler_output

            # pro_reidfeat_bn, pro_reidfeat, _ = model_reid(out_image, vid, cam_label=camids, view_label=target_view)
            # ori_reidfeat_bn, ori_reidfeat, _ = model_reid(img, vid, cam_label=camids, view_label=target_view)

            # img_224 = T.Resize((224,224))(img)
            # out_img_224 = T.Resize((224,224))(out_image)
            # clip_ori_feat_rn = CLIP_rn.encode_image(img_224)
            # clip_pro_feat_rn = CLIP_rn.encode_image(out_img_224)

            # # Mid feat loss
            # loss = compute_loss(logger, batch_latent, ori_latent, out_image, img, pro_feat, ori_feat, pro_feat_bn, ori_feat_bn, mid_feat)

            # # CLIP loss
            loss = compute_loss(logger, batch_latent, ori_latent, out_image, img, pro_feat, ori_feat)

            pbar.set_postfix_str(
                f"loss: {loss.item():.5f}")
            
            loss.backward()
            
            optimizer_latent.step()

        with torch.no_grad():
            pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)
            # pro_img = latent_2_image(cfg, ldm_stable, latent, batch_context)[:batch_size_per_gpu]

            # pro_img, mid_feat = latent_2_image(cfg, ldm_stable, latent, batch_context)
            # pro_img = pro_img[:batch_size_per_gpu]

            save_image(img, pro_img, n_iter, cfg, step+1, True, 'gallery')
            feat = model_msi(pro_img)
            feat_ori = model_msi(img)
            all_SIM.append(torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(feat, feat_ori, (-1 * torch.ones(feat.shape[0])).to('cuda')))
            logger.info(torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(feat, feat_ori, (-1 * torch.ones(feat.shape[0])).to('cuda')))
            logger.info(torch.mean(torch.stack(all_SIM, dim=0)))
            psnr_ssim(pro_img, img, all_SSIM, all_PSNR)
            
        for i in range(img.shape[0]):
            all_pro_gallery[imgpath[i]]=pro_img[i].detach().cpu()
            
        end_time = time.time()
        print('\n')
        print("Each batch need {:.4f}s".format(end_time-time_1))
        print('\n')

        del pro_img, feat_ori, feat

        end_time = time.time()
        all_time = end_time - start_time
        print('\n')
        print("Each batch need {:.4f}s".format(end_time-start_time))
        print('\n')

        # if n_iter % 10000 == 0:
        #     if cfg.MODEL.DIST_TRAIN:
        #         path = '/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW0.5_gallery_10_loss18_iter_' + str(n_iter) + '_' + str(dist.get_rank()) + '.pth'
        #         torch.save(all_pro_gallery, path)
        #         logger.info("save pth in {}".format(path))
        #     else:
        #         path = '/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW0.5_gallery_10_loss18_iter_' + str(n_iter) + '.pth'
        #         torch.save(all_pro_gallery, path)
        #         logger.info("save pth in {}".format(path))

    # reset_attention_control(ldm_stable)


    logger.info('--------------------------------------------------------------------------------')
    logger.info("Finish ADV-gallery set generation")
    logger.info('--------------------------------------------------------------------------------')
    logger.info("mean SSIM is {:.4f}".format(torch.mean(torch.stack(all_SSIM, dim=0))))
    logger.info("mean PSNR is {:.4f}".format(torch.mean(torch.stack(all_PSNR, dim=0))))

    if cfg.MODEL.DIST_TRAIN:
        path = '/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW0.5_gallery_10_loss18' + str(dist.get_rank()) + '.pth'
        torch.save(all_pro_gallery, path)
        logger.info("save pth in {}".format(path))
    else:
        path = '/data1/mqx_log/Privacy/PPReID/market/dataset_generation/boost/gallery'  + '.pth'
        torch.save(all_pro_gallery, path)
        logger.info("save pth in {}".format(path))

    for n_iter, (img, pid, camid, camids, target_view, imgpath, latent_pre, description) in enumerate(query_loader_st2):

        start_time = time.time()

        img = img.to(device)
        
        # img_AGW = img / 2 + 0.5
        # img_AGW = img_AGW.permute(0, 2, 3, 1)
        # mean_AGW = torch.as_tensor( [0.485, 0.456, 0.406], dtype=img.dtype, device=img.device)
        # std_AGW = torch.as_tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device)
        # img_AGW = img_AGW[:, :, :].sub(mean_AGW).div(std_AGW)
        # img_AGW = img_AGW.permute(0, 3, 1, 2)

        vid = torch.tensor(pid, dtype=torch.int64).to(device)
        camids = camids.to(device)
        target_view = target_view.to(device)
        batch_size_per_gpu = img.shape[0]

        with torch.no_grad():

            feat_sup, feat_sup_bfbn, score_pro = model_supervisor(img, cam_label=camids, view_label=target_view)
            
            feat_msi = model_msi(img)

            feat_transreid= model_transreid(img, cam_label=camids, view_label=target_view)
            
            feat_AGW = model_AGW(img)

            feat_transformer_base = model_transformer_base(img, cam_label=camids, view_label=target_view)

            feat_solider, _ = model_SOLIDER(img, cam_label=camids, view_label=target_view)

            feat_PCB = model_PCB(img)

            feat_resnet = model_ResNet(img)

            evaluator_1.update((feat_sup, pid, camid))
            evaluator_2.update((feat_msi, pid, camid))
            evaluator_3.update((feat_transreid, pid, camid))
            evaluator_4.update((feat_AGW, pid, camid))
            evaluator_5.update((feat_transformer_base, pid, camid))
            evaluator_6.update((feat_solider, pid, camid))
            evaluator_7.update((feat_PCB, pid, camid))
            evaluator_8.update((feat_resnet, pid, camid))

    for n_iter, (img, pid, camid, camids, target_view, imgpath, latent_pre, description) in enumerate(gallery_loader_st2):

        print('\n')
        print('--------------------------------------------------------------------------------')
        print('ADV-test set generation of BATCH {}'.format(n_iter))
        logger.info('ADV-gallery set generation of BATCH {}'.format(n_iter))
        print('--------------------------------------------------------------------------------')
        print('\n')
        start_time = time.time()

        pro_img_list = []
        for i in range(img.shape[0]):
            if imgpath[i] in all_pro_gallery.keys():
                pro_img_list.append(all_pro_gallery[imgpath[i]])
            else:
                pro_img_list.append(img[i])
                logger.info("LACK OF PRO_IMG in {}".format(imgpath[i]))
        pro_img = torch.stack(pro_img_list, dim=0).to(device)

        pro_img = (pro_img / 2 + 0.5).clamp(0, 1)
        pro_img = pro_img.permute(0, 2, 3, 1)

        mean = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=pro_img.dtype, device=pro_img.device)
        std = torch.as_tensor(cfg.INPUT.PIXEL_STD, dtype=pro_img.dtype, device=pro_img.device)
        out_img = pro_img[:, :, :].sub(mean).div(std)
        out_img = out_img.permute(0, 3, 1, 2)

        # mean_AGW = torch.as_tensor( [0.485, 0.456, 0.406], dtype=pro_img.dtype, device=pro_img.device)
        # std_AGW = torch.as_tensor([0.229, 0.224, 0.225], dtype=pro_img.dtype, device=pro_img.device)
        # pro_img_AGW = pro_img[:, :, :].sub(mean_AGW).div(std_AGW)
        # pro_img_AGW = pro_img_AGW.permute(0, 3, 1, 2)

        img = img.to(device)
        vid = torch.tensor(pid, dtype=torch.int64).to(device)
        camids = camids.to(device)
        target_view = target_view.to(device)
        batch_size_per_gpu = img.shape[0]

        with torch.no_grad():

            feat_sup, feat_sup_bfbn, score_pro = model_supervisor(out_img, cam_label=camids, view_label=target_view)
            
            feat_msi = model_msi(out_img)

            feat_transreid= model_transreid(out_img, cam_label=camids, view_label=target_view)
            
            feat_AGW = model_AGW(out_img)

            feat_transformer_base = model_transformer_base(out_img, cam_label=camids, view_label=target_view)

            feat_solider, _ = model_SOLIDER(out_img, cam_label=camids, view_label=target_view)

            feat_PCB = model_PCB(out_img)

            feat_resnet = model_ResNet(out_img)

            # feat_sup, feat_sup_bfbn, score_pro = model_supervisor(img, cam_label=camids, view_label=target_view)
            
            # feat_msi = model_msi(img)

            # feat_transreid= model_transreid(img, cam_label=camids, view_label=target_view)
            
            # feat_AGW = model_AGW(img)

            # feat_transformer_base = model_transformer_base(img, cam_label=camids, view_label=target_view)

            # feat_solider, _ = model_SOLIDER(img, cam_label=camids, view_label=target_view)

            # feat_PCB = model_PCB(img)

            # feat_resnet = model_ResNet(img)

            evaluator_1.update((feat_sup, pid, camid))
            evaluator_2.update((feat_msi, pid, camid))
            evaluator_3.update((feat_transreid, pid, camid))
            evaluator_4.update((feat_AGW, pid, camid))
            evaluator_5.update((feat_transformer_base, pid, camid))
            evaluator_6.update((feat_solider, pid, camid))
            evaluator_7.update((feat_PCB, pid, camid))
            evaluator_8.update((feat_resnet, pid, camid))


    cmc, mAP, _, _, _, _, _ = evaluator_1.compute()
    cmc2, mAP2, _, _, _, _, _ = evaluator_2.compute()
    cmc3, mAP3, _, _, _, _, _ = evaluator_3.compute()
    cmc4, mAP4, _, _, _, _, _ = evaluator_4.compute()
    cmc5, mAP5, _, _, _, _, _ = evaluator_5.compute()
    cmc6, mAP6, _, _, _, _, _ = evaluator_6.compute()
    cmc7, mAP7, _, _, _, _, _ = evaluator_7.compute()
    cmc8, mAP8, _, _, _, _, _ = evaluator_8.compute()

    logger.info("Validation Results ")
        
    logger.info("SSLpro2pro_mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("pro2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    # return cmc2[0], cmc2[4]

    logger.info("MSIori2pro_mAP: {:.1%}".format(mAP2))
    for r in [1, 5, 10]:
        logger.info("ori2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
    # return cmc2[0], cmc2[4]

    logger.info("TransReIDori2pro_mAP: {:.1%}".format(mAP3))
    for r in [1, 5, 10]:
        logger.info("ori2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc3[r - 1]))
    
    logger.info("AGWori2pro_mAP: {:.1%}".format(mAP4))
    for r in [1, 5, 10]:
        logger.info("ori2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc4[r - 1]))

    logger.info("ViTBori2pro_mAP: {:.1%}".format(mAP5))
    for r in [1, 5, 10]:
        logger.info("ori2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc5[r - 1]))

    logger.info("SOLIDERori2pro_mAP: {:.1%}".format(mAP6))
    for r in [1, 5, 10]:
        logger.info("ori2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc6[r - 1]))

    logger.info("PCBori2pro_mAP: {:.1%}".format(mAP7))
    for r in [1, 5, 10]:
        logger.info("ori2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc7[r - 1]))

    logger.info("PCBori2pro_mAP: {:.1%}".format(mAP8))
    for r in [1, 5, 10]:
        logger.info("ori2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc8[r - 1]))


