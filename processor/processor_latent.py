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
from diffattack.diff_latent_attack import preprocess, encoder, ddim_reverse_sample, diffusion_step, register_attention_control, reset_attention_control
from diffattack.distances import LpDistance
from diffattack.attentionControl import AttentionControlEdit
from diffattack.utils import aggregate_attention
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import re
from sklearn.manifold import TSNE
import  matplotlib.pyplot as plt
import torchvision.transforms as T
from timm.data.random_erasing import RandomErasing
import random
from torch.optim.optimizer import Optimizer, required

CHECK = 1e-5
SAT_MIN = 0.5

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import seaborn as sns

def visualize_tsne(orimodel_ori_query_feature, orimodel_pro_gallery_feature, sysmodel_ori_query_feature, sysmodel_ori_gallery_feature, sysmodel_pro_query_feature, sysmodel_pro_gallery_feature, query_label, gallery_label, output_path='/data1/mqx_log/'):
    # 选择10个唯一的标签
    unique_labels = np.unique(query_label)
    selected_labels = np.random.choice(unique_labels, 10, replace=False)
    
    selected_query_indices = []
    selected_gallery_indices = []
    
    # 对每个选定的标签选择5个query样本和15个gallery样本
    for label in selected_labels:
        query_indices = np.where(query_label == label)[0]
        gallery_indices = np.where(gallery_label == label)[0]
        
        # 从query中选择5个样本
        selected_query_indices.extend(np.random.choice(query_indices, min(5, len(query_indices)), replace=False))
        
        # 从gallery中选择15个样本
        selected_gallery_indices.extend(np.random.choice(gallery_indices, min(15, len(gallery_indices)), replace=False))
    
    # 过滤数据
    selected_query_indices = np.array(selected_query_indices)
    selected_gallery_indices = np.array(selected_gallery_indices)

    # 过滤特征和标签
    orimodel_ori_query_feature = orimodel_ori_query_feature[selected_query_indices]
    orimodel_pro_gallery_feature = orimodel_pro_gallery_feature[selected_gallery_indices]
    sysmodel_ori_query_feature = sysmodel_ori_query_feature[selected_query_indices]
    sysmodel_ori_gallery_feature = sysmodel_ori_gallery_feature[selected_gallery_indices]
    sysmodel_pro_query_feature = sysmodel_pro_query_feature[selected_query_indices]
    sysmodel_pro_gallery_feature = sysmodel_pro_gallery_feature[selected_gallery_indices]
    query_label = query_label[selected_query_indices]
    gallery_label = gallery_label[selected_gallery_indices]
    
    # 生成更有区分度的颜色
    palette = sns.color_palette("tab10", 10)
    label_to_color = {label: palette[i] for i, label in enumerate(selected_labels)}
    
    # 计算每一类的颜色
    query_colors = np.array([label_to_color[label] for label in query_label])
    gallery_colors = np.array([label_to_color[label] for label in gallery_label])

    # 定义一个绘图函数，绘制t-SNE结果
    def plot_tsne(tsne_result, query_feature, gallery_feature, query_label, gallery_label, query_marker, gallery_marker, output_filename, flag=0):
        plt.figure(figsize=(10, 10))
        # 绘制查询样本
        plt.scatter(tsne_result[0:len(query_feature), 0], tsne_result[0:len(query_feature), 1], c=query_colors, marker=query_marker, s=300, alpha=0.9, label='Query')
        # 绘制画廊样本
        plt.scatter(tsne_result[len(query_feature):, 0], tsne_result[len(query_feature):, 1], c=gallery_colors, marker=gallery_marker, s=400, alpha=0.9, label='Gallery')

        plt.axis('off')  # 隐藏坐标轴
        plt.legend(fontsize=18, loc='upper right', frameon=False)


        plt.savefig(f'{output_path}/{output_filename}', dpi=450)
        plt.show()

    # 处理 orimodel_ori_query_feature 和 orimodel_pro_gallery_feature 的 t-SNE
    orimodel_features = np.vstack([orimodel_ori_query_feature, orimodel_pro_gallery_feature])
    tsne = TSNE(n_components=2, random_state=0)
    tsne_result = tsne.fit_transform(orimodel_features)
    plot_tsne(tsne_result, orimodel_ori_query_feature, orimodel_pro_gallery_feature, query_label, gallery_label, query_marker='^', gallery_marker='*', output_filename='tsne_orimodel_ori_query_vs_pro_gallery.png', flag=1)

    # 处理 sysmodel_ori_query_feature 和 sysmodel_ori_gallery_feature 的 t-SNE
    sysmodel_features = np.vstack([sysmodel_ori_query_feature, sysmodel_ori_gallery_feature])
    tsne_result = tsne.fit_transform(sysmodel_features)
    plot_tsne(tsne_result, sysmodel_ori_query_feature, sysmodel_ori_gallery_feature, query_label, gallery_label, query_marker='^', gallery_marker='*', output_filename='tsne_sysmodel_ori_query_vs_ori_gallery.png')

    # 处理 sysmodel_ori_query_feature 和 sysmodel_pro_gallery_feature 的 t-SNE
    sysmodel_features = np.vstack([sysmodel_ori_query_feature, sysmodel_pro_gallery_feature])
    tsne_result = tsne.fit_transform(sysmodel_features)
    plot_tsne(tsne_result, sysmodel_ori_query_feature, sysmodel_pro_gallery_feature, query_label, gallery_label, query_marker='^', gallery_marker='*', output_filename='tsne_sysmodel_ori_query_vs_pro_gallery.png')

    # 处理 sysmodel_pro_query_feature 和 sysmodel_pro_gallery_feature 的 t-SNE
    sysmodel_features = np.vstack([sysmodel_pro_query_feature, sysmodel_pro_gallery_feature])
    tsne_result = tsne.fit_transform(sysmodel_features)
    plot_tsne(tsne_result, sysmodel_pro_query_feature, sysmodel_pro_gallery_feature, query_label, gallery_label, query_marker='^', gallery_marker='*', output_filename='tsne_sysmodel_pro_query_vs_pro_gallery.png')

    # features = features[:100, ]
    # labels = labels[:100]

    # tsne = TSNE(n_components=2, random_state=0)
    # tsne_result = tsne.fit_transform(features)

    # # 可视化并保存图像
    # plt.figure(figsize=(20, 20))
    # scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    # plt.colorbar(scatter)
    # for i in range(len(tsne_result)):
    #     plt.text(tsne_result[i, 0], tsne_result[i, 1], str(labels[i]), fontsize=12, alpha=0.7)
    # plt.savefig(output_path) 

def rescale_check(check, sat, sat_change, sat_min):
    return sat_change < check and sat > sat_min

class MI_SGD(Optimizer):
    def __init__(
            self,
            params,
            lr=required,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            max_eps=10 / 255,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign=False,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MI_SGD, self).__init__(params, defaults)
        self.sat = 0
        self.sat_prev = 0
        self.max_eps = max_eps

    def __setstate__(self, state):
        super(MI_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def rescale(self, ):
        for group in self.param_groups:
            if not group["sign"]:
                continue
            for p in group["params"]:
                self.sat_prev = self.sat
                self.sat = (p.data.abs() >= self.max_eps).sum().item() / p.data.numel()
                sat_change = abs(self.sat - self.sat_prev)
                if rescale_check(CHECK, self.sat, sat_change, SAT_MIN):
                    print('rescaled')
                    p.data = p.data / 2

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group["sign"]:
                    d_p = d_p / (d_p.norm(1) + 1e-12)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if group["sign"]:
                    p.data.add_(-group["lr"], d_p.sign())
                    # p.data = torch.clamp(p.data, -self.max_eps, self.max_eps)
                else:
                    p.data.add_(-group["lr"], d_p)

        return loss

def save_image(img, pro_img, batch_num, cfg, step, is_test, dataset):

    if is_test :
        img_dir = cfg.IMG_TEST_DIR
    else:
        img_dir = cfg.IMG_DIR

    img_array = img[0].detach().permute(1, 2, 0).cpu().numpy()  # 从 [1, 3, 256, 128] 到 [256, 128, 3]
    ori_img = (((img_array*0.5)+0.5)*255).astype(np.uint8)
    ori_image = Image.fromarray(ori_img)
    if is_test :
        ori_path = img_dir + dataset + '_bs_' + str(batch_num) + 'No_0' + 'ori.jpg'
    else:
        ori_path = img_dir + 'bs_' + str(batch_num) + 'No_0' + 'ori.jpg'

    proimg_array = pro_img[0].detach().permute(1, 2, 0).cpu().numpy()  # 从 [1, 3, 256, 128] 到 [256, 128, 3]
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
    
    if batch_num % 1==0 and step == 0 :
        ori_image.save(ori_path)
        # ori_image1.save(ori_path1)
    if batch_num % 1==0 and step != 0:
        pro_image.save(pro_path)
        diff_image.save(diff_path)
        # pro_image1.save(pro_path1)
        # diff_image1.save(diff_path1)

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))
    a = ((img[0].permute(1, 2, 0).detach().unsqueeze(0))*0.5)+0.5
    a = a.cpu().numpy()
    b = ((pro_img[0].permute(1, 2, 0).detach().unsqueeze(0))*0.5)+0.5
    b = b.cpu().numpy()
    print("L1: {}\tL2: {}\tLinf: {}".format(L1(a, b), L2(a,b), Linf(a,b)))

def get_batch_latent(cfg, img, prompt, model, guidance_scale=0):
    batch_size = img.shape[0]
    latent, inversion_latents = ddim_reverse_sample(img, prompt, model, batch_size=batch_size, 
            num_inference_steps=cfg.DIFFUSION.NUM_INFERENCE_STEPS, guidance_scale=guidance_scale, res=cfg.INPUT.SIZE_TRAIN)
    inversion_latents = inversion_latents[::-1]
    latent = inversion_latents[cfg.DIFFUSION.START_STEP - 1]
    print(f"Totally get {batch_size} latents, with shape of {latent[0].shape}")
    
    return latent

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
    for ind, t in enumerate(model.scheduler.timesteps[1 + cfg.DIFFUSION.START_STEP - 1:]):
        latents = diffusion_step(model, latents, contexts[ind], t, guiding_scale)
    img = model.vae.decode(1 / 0.18215 * latents)['sample']

    return img

def compute_loss(logger, controller, batch_latent, ori_latent, out_image, img, pro_feat, ori_feat, pro_feat_bn, ori_feat_bn):
    loss_1 = torch.nn.L1Loss(reduction='mean')(batch_latent, ori_latent)
    print("loss_1: MSELoss between ori_lt & pro_lt: {:.4f}".format(loss_1))

    loss_2 = torch.nn.L1Loss(reduction='mean')(out_image, img)
    loss_MSE = torch.nn.MSELoss(reduction='mean')(out_image, img)
    print("loss_2: MSELoss between out_image & img: {:.4f}".format(loss_2))

    loss_3 = torch.nn.MSELoss(reduction='mean')(pro_feat, ori_feat)
    logger.info("loss_3: MSELoss between pro_feat & ori_feat: {:.4f}".format(loss_3))

    loss_7 = torch.nn.MSELoss(reduction='mean')(pro_feat_bn, ori_feat_bn)
    logger.info("loss_3: MSELoss between pro_feat_bn & ori_feat_bn: {:.4f}".format(loss_7))
    
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

    loss_6 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(pro_feat_bn, ori_feat_bn, (-1 * torch.ones(pro_feat.shape[0])).to('cuda'))
    print("loss_6: After BN Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_6))
    logger.info("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_6))

    loss_10 = torch.max(torch.abs(batch_latent - ori_latent))
    print("loss_10: MaxABSLoss between ori_lt & pro_lt: {:.4f}".format(loss_10))

    # loss = loss_5 + 10 * loss_2 + 10 * loss_1 + 2 * loss_6 - 0.1 * loss_3 # running 1e-1 30iter
    # loss = 3 * loss_5 + 5 * loss_6 + 30 * loss_2 + 30 * loss_1  - 0.2 * loss_7 
    print(controller.loss)
    # loss = 10 * loss_2 + 5 * loss_MSE + loss_1 + loss_6 - 0.1 * loss_7 + 30 * controller.loss
    loss = loss_5 + 5 * controller.loss + 3 * loss_2 + 2 * loss_1 

    return loss

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

def do_train_dif_latent(cfg,
             model_reid,
             center_criterion,
             train_loader,
             val_loader,
             train_loader_st2_pro,
             train_loader_st2,
             val_loader_st2,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             reverse_tri_loss,
             tri_loss,
             num_query, local_rank):
             
    device = "cuda"
    
    # diffusion & CLIP init     
    # ldm_stable = StableDiffusionPipeline.from_pretrained(cfg.MODEL.PRETRAINED_DIFFUSION_PATH).to(local_rank)
    # ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    # ldm_stable.scheduler.set_timesteps(cfg.DIFFUSION.NUM_INFERENCE_STEPS)
    # CLIP_model = CLIPModel.from_pretrained("/data1/mqx_log/clip-vit-large-patch14").to(local_rank)
    # CLIP_processor = CLIPProcessor.from_pretrained("/data1/mqx_log/clip-vit-large-patch14")

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
        model_reid.to(local_rank)
        # model_reid.eval()
        # model_supervisor.to(local_rank)
        # model_supervisor.eval()
        # ldm_stable.unet.requires_grad_(False)
        # ldm_stable.vae.requires_grad_(False)
        # ldm_stable.text_encoder.requires_grad_(False)
        # CLIP_model.eval()
        
    # Loss & ACC meter 
    loss_meter = AverageMeter()
    reverse_loss_meter = AverageMeter()
    cos_loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()

    acc_meter = AverageMeter()  
    acc_meter_ori = AverageMeter()

    # evaluator for Rank1 & mAP 
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_pro = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    # scaler for amp.autocast 
    scaler = amp.GradScaler()

    # train_transform     
    train_transforms = T.Compose([
            # T.ToTensor(),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING, -1),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
        ])
    
    # list for adv_train imgs
    all_pro_image = {}

    all_PSNR = []
    all_SSIM = []

    # stage 1
    # Adv_train set generation
    # for n_iter, (img, vid, target_cam, target_view, img_path, latent_pre, description) in enumerate(train_loader) :
        
    #     print('\n')
    #     print('--------------------------------------------------------------------------------')
    #     print('STAGE 1 : ADV-train set generation of BATCH {}'.format(n_iter))
    #     logger.info('STAGE 1 : ADV-train set generation of BATCH {}'.format(n_iter))
    #     print('--------------------------------------------------------------------------------')
    #     print('\n')

    #     # tensor to GPU
    #     img = img.to(device)
    #     target = vid.to(device)
    #     target_cam = target_cam.to(device)
    #     target_view = target_view.to(device)
        
    #     # img 2 latent
    #     batch_prompt = list(description)
    #     batch_latent = latent_pre.to(device)
    #     # batch_latent = get_batch_latent(cfg, img, batch_prompt, ldm_stable)
    #     ori_latent = latent_pre.to(device).requires_grad_(True)
    #     batch_latent.requires_grad_(True)

    #     # latent_optimizer
    #     # optimizer_latent = MI_SGD(
    #     #     [
    #     #         {"params": [batch_latent], "lr": 3e-2, "momentum": cfg.SOLVER.MOMENTUM, "sign": True}
    #     #     ],
    #     # )
    #     optimizer_latent = torch.optim.AdamW(params=[{"params": [batch_latent]}], lr=1e-1, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    #     # Controller
    #     controller = AttentionControlEdit(cfg, cfg.DIFFUSION.NUM_INFERENCE_STEPS, 1., cfg.INPUT.SIZE_TRAIN)
    #     register_attention_control(ldm_stable, controller)

    #     # text embedding & uncond embedding & DDIM context
    #     max_length = 77
    #     batch_uncond_input = ldm_stable.tokenizer(
    #         [""] * batch_size_per_gpu, padding="max_length", max_length=max_length, return_tensors="pt"
    #     )
    #     bathc_uncond_embeddings = ldm_stable.text_encoder(batch_uncond_input.input_ids.to(ldm_stable.device))[0].detach()
    #     batch_text_input = ldm_stable.tokenizer(
    #         batch_prompt,
    #         padding="max_length",
    #         max_length=ldm_stable.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     bacth_text_embeddings = ldm_stable.text_encoder(batch_text_input.input_ids.to(ldm_stable.device))[0].detach()
    #     batch_uncond_text = torch.cat((bathc_uncond_embeddings,bacth_text_embeddings), dim=0)
    #     batch_context = batch_uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)

    #     # for self attetion control
    #     uncond_text = torch.cat((torch.cat((bathc_uncond_embeddings, bathc_uncond_embeddings), dim=0) , torch.cat((bacth_text_embeddings, bacth_text_embeddings), dim=0)), dim=0)
    #     batch_uncond_text = uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)
        
    #     # Strat optimizing latent
    #     iterations = 30
    #     pbar = tqdm(range(iterations), desc="Iterations")
    #     start_time = time.time()

    #     for step, _ in enumerate(pbar):
    #         controller.loss = 0
    #         controller.reset()

    #         # for self-attention control
    #         latent = torch.cat((batch_latent, ori_latent), dim=0)

    #         optimizer_latent.zero_grad()
            
    #         pro_img = latent_2_image(cfg, ldm_stable, latent, batch_uncond_text)[:cfg.SOLVER.IMS_PER_BATCH]

    #         # img1_attention_map = aggregate_attention(batch_prompt, controller, [cfg.INPUT.SIZE_TRAIN[0]//32, cfg.INPUT.SIZE_TRAIN[1]//32], ("up", "down"), True, 0, is_cpu=False)
    #         # img2_attention_map = aggregate_attention(batch_prompt, controller, [cfg.INPUT.SIZE_TRAIN[0]//32, cfg.INPUT.SIZE_TRAIN[1]//32], ("up", "down"), True, 1, is_cpu=False)

    #         # img1_pmt = len(re.findall(r"\w+|\S", batch_prompt[0]))
    #         # img2_pmt = len(re.findall(r"\w+|\S", batch_prompt[1]))

    #         # img1_true_label_attention_map = img1_attention_map[:, :, 1: img1_pmt + 1]

    #         # img2_true_label_attention_map = img2_attention_map[:, :, 1: img2_pmt + 1]

    #         # save_image(img, pro_img, n_iter, cfg, step, False, 'train')

    #         out_image = norm_img(cfg, pro_img)

    #         # text_input = CLIP_processor(text=description, return_tensors="pt", padding=True).to(device)

    #         # CLIP_ori_output = CLIP_model(input_ids=text_input.input_ids, pixel_values=img, interpolate_pos_encoding=True, output_hidden_states=True)
    #         # CLIP_pro_output = CLIP_model(input_ids=text_input.input_ids, pixel_values=out_image, interpolate_pos_encoding=True, output_hidden_states=True)
            
    #         # project_ori_feature = CLIP_ori_output.vision_model_output.pooler_output
    #         # project_text_feature = CLIP_ori_output.text_embeds
    #         # logits_ori_image = CLIP_ori_output.logits_per_image
    #         # logits_ori_text = CLIP_ori_output.logits_per_text

    #         # project_pro_feature = CLIP_pro_output.vision_model_output.pooler_output
    #         # logits_pro_image = CLIP_pro_output.logits_per_image
    #         # logits_pro_text = CLIP_pro_output.logits_per_text
            
    #         pro_feat_bn, pro_feat = model_supervisor(out_image, target, cam_label=target_cam, view_label=target_view)
    #         ori_feat_bn, ori_feat = model_supervisor(img, target, cam_label=target_cam, view_label=target_view)

    #         loss = compute_loss(logger, controller, batch_latent, ori_latent, out_image, img, pro_feat, ori_feat, pro_feat_bn, ori_feat_bn)

    #         pbar.set_postfix_str(
    #             f"loss: {loss.item():.5f}")
            
    #         loss.backward()
            
    #         optimizer_latent.step()

    #     with torch.no_grad():
    #         pro_img = latent_2_image(cfg, ldm_stable, latent, batch_uncond_text)[:cfg.SOLVER.IMS_PER_BATCH]
    #         # save_image(img, pro_img, n_iter, cfg, step+1, False, 'train')

    #         psnr_ssim(pro_img, img, all_SSIM, all_PSNR)
            
    #     for i in range(img.shape[0]):
    #         all_pro_image[img_path[i]]=pro_img[i].detach().cpu()

    #     del pro_img
            
    #         # pro.append(pro_img.detach())
    #         # tar.append(target.detach())

    #     # if n_iter % 100 == 0:
    #     #     pro_i = torch.cat(pro, dim=0)
    #     #     tar_i = torch.cat(tar, dim=0)
    #     #     feat_sup, _ = model_supervisor(pro_i, tar_i, cam_label=target_cam, view_label=target_view)
    #     #     loss_4 = reverse_tri_loss(feat_sup, tar_i)
    #     #     print(loss_4)
    #     end_time = time.time()
    #     print('\n')
    #     print("Each batch need {:.4f}s".format(end_time-start_time))
    #     print('\n')

    # logger.info('--------------------------------------------------------------------------------')
    # logger.info("Finish ADV-train set generation")
    # logger.info('--------------------------------------------------------------------------------')
    # logger.info("mean SSIM is {:.4f}".format(torch.mean(torch.stack(all_SSIM, dim=0))))
    # logger.info("mean PSNR is {:.4f}".format(torch.mean(torch.stack(all_PSNR, dim=0))))
    # torch.save(all_pro_image, '/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/pro_train_controller.pth')

    # stage 2
    # Training on Adv_train set 
    all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/train/pro_train_loss18.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/pro_train_msmtloss18.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/train/pro_train_5loss18.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/train/pro_train_15loss18.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/train/pro_train_20loss18.pth")

    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW0.5_pro_train_loss18.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/MSI0.5_pro_train_loss18.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/SOLID_pro_train_loss18.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/Trans_pro_train_loss18.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_1/train.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_2/train.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_3/train.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/no_L_IQ/train.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/only_IQ/train.pth")
    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/dataset_generation/ablation_new/only_IQ/train.pth")


    # all_pro_image = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/boost/train.pth")

    for epoch in range(1, epochs + 1):

        print('\n')
        print('--------------------------------------------------------------------------------')
        print('STAGE 2 : System model training on ADV-train set of EPOCH {}'.format(epoch))
        logger.info('STAGE 2 : System model training on ADV-train set of EPOCH {}'.format(epoch))
        print('--------------------------------------------------------------------------------')
        print('\n')

        start_time = time.time()
        loss_meter.reset()
        reverse_loss_meter.reset()
        cos_loss_meter.reset()
        ce_loss_meter.reset()
        tri_loss_meter.reset()

        acc_meter.reset()
        acc_meter_ori.reset()
        evaluator.reset()
        evaluator_pro.reset()

        model_reid.train()
        # for module in model_reid.modules():
        #     if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d)):
        #         module.track_running_stats = False
        
        for n_iter, (img, vid, target_cam, target_view, img_path, latent_pre, description) in enumerate(train_loader_st2) :
        # for n_iter, ((img, vid, target_cam, target_view, img_path, latent_pre, description), (pro_img, vid_pro, target_cam_pro, target_view_pro, img_path_pro, latent_pre_pro, description_pro)) in enumerate(zip(train_loader_st2, train_loader_st2_pro)):        
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            # print('\n')
            # print('--------------------------------------------------------------------------------')
            # print('STAGE 2 : System model training on ADV-train set of EPOCH {} BATCH {}'.format(epoch, n_iter))
            # logger.info('STAGE 2 : System model training on ADV-train set of EPOCH {} BATCH {}'.format(epoch, n_iter))
            # print('--------------------------------------------------------------------------------')
            # print('\n')

            pro_img_list = []
            for i in range(img.shape[0]):
                if img_path[i] in all_pro_image.keys():
                    pro_img_list.append(all_pro_image[img_path[i]])
                else:
                    pro_img_list.append(img[i])
                    # logger.info("LACK OF PRO_IMG in {}".format(img_path[i]))

            # key = random.sample(list(all_pro_image.keys()), img.shape[0])
            # pro_img_list = [all_pro_image[key] for key in key]
                    
            pro_img = torch.stack(pro_img_list, dim=0).to(device)

            pro_img = (pro_img / 2 + 0.5).clamp(0, 1)
            pro_img = pro_img.permute(0, 2, 3, 1)
            mean = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=pro_img.dtype, device=pro_img.device)
            std = torch.as_tensor(cfg.INPUT.PIXEL_STD, dtype=pro_img.dtype, device=pro_img.device)
            pro_img = pro_img[:, :, :].sub(mean).div(std)
            pro_img = pro_img.permute(0, 3, 1, 2)

            
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            pro_img = pro_img.to(device)
            # img_trans = train_transforms(img)
            pro_img_trans = train_transforms(pro_img)

            with amp.autocast(enabled=True):
                                    
                score_ori, feat_ori, feat_ori_afbn = model_reid(img, target, cam_label=target_cam, view_label=target_view)

                score_pro, feat_pro, feat_pro_afbn = model_reid(pro_img, target, cam_label=target_cam, view_label=target_view)

                '' 'MSI '''
                # feat_ori, score_ori, feats_ori, scores_ori, sam_logits = model_reid(img)

                # feat_pro, score_pro, feats_pro, scores_pro, sam_logits = model_reid(pro_img)

                '' 'AGW '''
                # score_ori, feat_ori = model_reid(img)

                # score_pro, feat_pro = model_reid(pro_img)

                '''SOLID'''
                # score_ori, feat_ori, feat_ori_afbn = model_reid(img, target, cam_label=target_cam, view_label=target_view)
                # score_pro, feat_pro, feat_pro_afbn = model_reid(pro_img, target, cam_label=target_cam, view_label=target_view)


                '''Trans'''
                # score_ori, feat_ori = model_reid(img, target, cam_label=target_cam, view_label=target_view)
                # score_pro, feat_pro = model_reid(pro_img, target, cam_label=target_cam, view_label=target_view)

                # model_reid.eval()
                # feat_ori, feat_ori_afbn, score_ori = model_reid(img, target, cam_label=target_cam, view_label=target_view)
                # model_reid.train()

                loss_1, loss_ce, loss_tri = loss_fn(score_pro, feat_pro, target, target_cam, False)
                # print("ce loss + tri_loss: adv_image in sys_model: {:.4f}".format(loss_1))
                # if (n_iter ) % log_period == 0:
                #     logger.info("ce loss + tri_loss: adv_image in sys_model: {:.4f}".format(loss_1))

                loss_2 = reverse_tri_loss(feat_ori, target)
                '''MSI'''
                # feat_cat = torch.cat((feat_ori, feats_ori), dim=1)
                # loss_2= reverse_tri_loss(feat_cat, target)
                '''MSI'''
                # loss_4, loss_4ce, loss_4tri = loss_fn(scores_pro, feats_pro, target, target_cam, False)
            # print("reverse tri_loss: ori_img in sys_model: {:.4f}".format(loss_2))
            # if (n_iter ) % log_period == 0:
            #     logger.info("reverse tri_loss: ori_img in sys_model: {:.4f}".format(loss_2))

            # loss_3 = loss_fn(score_ori, feat_ori, target, target_cam, True)
            # if (n_iter ) % log_period == 0:
            #     logger.info("ce loss + tri_loss: ori_img in sys_model: {:.4f}".format(loss_3))

                # loss_4 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(feat_ori, feat_pro, -1 * torch.ones(feat_ori.shape[0]).to('cuda'))
            # if (n_iter ) % log_period == 0:
            #     logger.info("CosineEmbeddingLoss between feat_ori & feat_pro: {:.4f}".format(loss_4))

                # loss_5 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(feat_ori_afbn, feat_pro_afbn, -1 * torch.ones(feat_ori.shape[0]).to('cuda'))
            # if (n_iter ) % log_period == 0:
            #     logger.info("CosineEmbeddingLoss between feat_ori & feat_pro: {:.4f}".format(loss_5))
            
            # loss_6 = tri_loss(feat_ori, target)
            # if (n_iter ) % log_period == 0:
            #     logger.info("tri_loss: ori_img in sys_model: {:.4f}".format(loss_6))
            # if epoch < 30:
            #     loss = loss_4 
            # else:
            # loss = loss_1 + 5 * loss_4 + 3 * loss_5

            # loss = loss_1 + 5 * loss_4 + 5 * loss_5 + loss_2

                loss = loss_1 + loss_2 
                # if epoch < 20 :
                #     loss = loss_1
                # else:
                #     loss = loss_2

                # if epoch < 20 :
                #     loss = loss_2
                # else:
                #     loss = loss_1

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            
            if isinstance(score_pro, list):
                acc = (score_pro[0].max(1)[1] == target).float().mean()
            else:
                acc = (score_pro.max(1)[1] == target).float().mean()

            if isinstance(score_ori, list):
                acc_ori = (score_ori[0].max(1)[1] == target).float().mean()
            else:
                acc_ori = (score_ori.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            reverse_loss_meter.update(loss_2.item(), img.shape[0])
            # cos_loss_meter.update(loss_4.item(), img.shape[0])
            ce_loss_meter.update(loss_ce.item(), img.shape[0])
            tri_loss_meter.update(loss_tri.item(), img.shape[0])

            acc_meter.update(acc, 1)
            acc_meter_ori.update(acc_ori,1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter ) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, CELoss: {:.3f}, TriLoss: {:.3f}, RevLoss: {:.3f}, CosLoss: {:.3f}, Acc: {:.3f}, Acc_ori: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader_st2), loss_meter.avg, ce_loss_meter.avg, tri_loss_meter.avg, reverse_loss_meter.avg, cos_loss_meter.avg, acc_meter.avg, acc_meter_ori.avg, base_lr))
            else:
                if (n_iter ) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, CELoss: {:.3f}, TriLoss: {:.3f}, RevLoss: {:.3f}, CosLoss: {:.3f}, Acc: {:.3f}, Acc_ori: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader_st2), loss_meter.avg, ce_loss_meter.avg, tri_loss_meter.avg, reverse_loss_meter.avg, cos_loss_meter.avg, acc_meter.avg, acc_meter_ori.avg, base_lr))
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader_st2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model_reid.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'model_reid_{}.pth'.format(epoch)))
            else:
                torch.save(model_reid.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'model_reid_{}.pth'.format(epoch)))


        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model_reid.eval()
                    for n_iter, (img, vid, camid, camids, target_view, imgpath, latent_pre, description) in enumerate(val_loader_st2):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, _, _ = model_reid(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/query/pro_query_loss18.pth")
                all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/gallery/pro_gallery_loss18.pth")
                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/pro_querytrue_msmtloss18.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/pro_gallery_msmtloss18.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/query/pro_query_5loss18.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/gallery/pro_gallery_loss18_5.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/query/pro_query_15loss18.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/gallery/pro_gallery_loss18_15.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/query/pro_query_20loss18.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/gallery/pro_gallery_loss18_20.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW0.5_pro_query_loss18.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW0.5_gallery_10_loss18.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/MSI0.5_pro_query_loss18.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/MSI0.5_gallery_10_loss18.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/SOLID_pro_query_loss18.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/SOLID_gallery_12_loss18.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/Trans_pro_query_loss18.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/Trans_gallery_12_loss18.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_1/query.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_1/gallery.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_2/query.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_2/gallery.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_3/query.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_3/gallery.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/only_IQ/query.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/only_IQ/gallery.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/dataset_generation/ablation_new/only_IQ/query.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/dataset_generation/ablation_new/only_IQ/gallery.pth")

                # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/boost/query.pth")
                # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/boost/gallery.pth")

                model_reid.eval()
                for n_iter, (img, vid, camid, camids, target_view, imgpath, latent_pre, description) in enumerate(val_loader_st2):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)

                        pro_img_list = []
                        for i in range(img.shape[0]):
                            if imgpath[i] in all_pro_query.keys():
                                pro_img_list.append(all_pro_query[imgpath[i]].to(device))
                            elif imgpath[i] in all_pro_gallery.keys():
                                pro_img_list.append(all_pro_gallery[imgpath[i]].to(device))
                            else:
                                pro_img_list.append(img[i])
                                # logger.info("LACK OF PRO_IMG in {}".format(imgpath[i]))
                        pro_img = torch.stack(pro_img_list, dim=0)

                        pro_img = (pro_img / 2 + 0.5).clamp(0, 1)
                        pro_img = pro_img.permute(0, 2, 3, 1)
                        mean = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=pro_img.dtype, device=pro_img.device)
                        std = torch.as_tensor(cfg.INPUT.PIXEL_STD, dtype=pro_img.dtype, device=pro_img.device)
                        pro_img = pro_img[:, :, :].sub(mean).div(std)
                        pro_img = pro_img.permute(0, 3, 1, 2)

                        feat, _, _ = model_reid(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                        feat_pro, _, _ = model_reid(pro_img, cam_label=camids, view_label=target_view)
                        evaluator_pro.update((feat_pro, vid, camid))

                        '''Trans'''
                        # feat = model_reid(img, cam_label=camids, view_label=target_view)
                        # evaluator.update((feat, vid, camid))
                        # feat_pro= model_reid(pro_img, cam_label=camids, view_label=target_view)
                        # evaluator_pro.update((feat_pro, vid, camid))
                        '''SOLID'''
                        # feat, _ = model_reid(img, cam_label=camids, view_label=target_view)
                        # evaluator.update((feat, vid, camid))
                        # feat_pro, _ = model_reid(pro_img, cam_label=camids, view_label=target_view)
                        # evaluator_pro.update((feat_pro, vid, camid))
                        '''MSI'''
                        # feat= model_reid(img)
                        # evaluator.update((feat, vid, camid))
                        # feat_pro = model_reid(pro_img)
                        # evaluator_pro.update((feat_pro, vid, camid))
                        '''AGW'''
                        # feat= model_reid(img)
                        # evaluator.update((feat, vid, camid))
                        # feat_pro = model_reid(pro_img)
                        # evaluator_pro.update((feat_pro, vid, camid))

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results (System model on original query & gallery) - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                cmc_pro, mAP_pro, _, _, _, _, _ = evaluator_pro.compute()
                logger.info("Validation Results (System model on protected query & gallery) - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP_pro))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_pro[r - 1]))
                    
                torch.cuda.empty_cache()

def do_inference_dif_latent(cfg,
                 model_reid,
                 model_supervisor,
                 val_loader,
                 train_loader,
                 query_loader,
                 gallery_loader,
                 num_query,
                 tri_loss):
    
    device = "cuda"

    ldm_stable = StableDiffusionPipeline.from_pretrained(cfg.MODEL.PRETRAINED_DIFFUSION_PATH).to(device)
    ldm_stable.scheduler = DDIMScheduler.from_config(ldm_stable.scheduler.config)
    ldm_stable.scheduler.set_timesteps(cfg.DIFFUSION.NUM_INFERENCE_STEPS)
    # CLIP_model = CLIPModel.from_pretrained("/data1/mqx_log/clip-vit-large-patch14").to(device)
    # CLIP_processor = CLIPProcessor.from_pretrained("/data1/mqx_log/clip-vit-large-patch14")

    ldm_stable.unet.requires_grad_(False)
    ldm_stable.vae.requires_grad_(False)
    ldm_stable.text_encoder.requires_grad_(False)
    # CLIP_model.eval()
    
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator_1 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_2 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_3 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator_4 = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator_1.reset()
    evaluator_2.reset()
    evaluator_3.reset()
    evaluator_4.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model_reid = nn.DataParallel(model_reid)
        model_reid.to(device)
        model_supervisor.to(device)

    model_reid.eval()

    # for module in model_reid.modules():
    #     if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
    #         module.train()
    train_transforms = T.Compose([
            # T.ToTensor(),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING, -1),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu')
        ])
    
    model_supervisor.eval()


    all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/query/pro_query_loss18.pth")
    all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/gallery/pro_gallery_loss18.pth")


    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_1/query.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_1/gallery.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_2/query.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_2/gallery.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_3/query.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/different_lambda_3/gallery.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/no_L_IQ/query.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/no_L_IQ/gallery.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/only_IQ/query.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/ablation_new/only_IQ/gallery.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/dataset_generation/ablation_new/only_IQ/query.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/dataset_generation/ablation_new/only_IQ/gallery.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/boost/query.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/boost/gallery.pth")
    
    orimodel_ori_query_feature = []
    orimodel_pro_gallery_feature = []

    sysmodel_ori_query_feature = []
    sysmodel_ori_gallery_feature = []

    sysmodel_ori_query_feature = []
    sysmodel_pro_gallery_feature = []

    sysmodel_pro_query_feature = []
    sysmodel_pro_gallery_feature = []

    query_labels = []
    gallery_labels = []
    


    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/query/pro_query_5loss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/gallery/pro_gallery_loss18_5.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/query/pro_query_15loss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/gallery/pro_gallery_loss18_15.pth")
    
    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/query/pro_query_20loss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/market/dataset_generation/pth/gallery/pro_gallery_loss18_20.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/pro_querytrue_msmtloss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/pro_gallery_msmtloss18.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/SOLID_pro_query_loss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/SOLID_gallery_12_loss18.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW_pro_query_loss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/AGW_gallery_12_loss18.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/MSI_pro_query_loss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/MSI_gallery_12_loss18.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/SOLID_pro_query_loss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/SOLID_gallery_12_loss18.pth")

    # all_pro_query = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/Trans_pro_query_loss18.pth")
    # all_pro_gallery = torch.load("/data1/mqx_log/Privacy/PPReID/msmt/pth/Trans_gallery_12_loss18.pth")

    
    
    # all_pro_query = {}
    # all_pro_gallery = {}
    for n_iter, (img, pid, camid, camids, target_view, imgpath, latent_pre, description) in enumerate(query_loader):
        
        print('\n')
        print('--------------------------------------------------------------------------------')
        print('ADV-test set generation of BATCH {}'.format(n_iter))
        logger.info('ADV-query set generation of BATCH {}'.format(n_iter))
        print('--------------------------------------------------------------------------------')
        print('\n')
        start_time = time.time()

        pro_img_list = []
        for i in range(img.shape[0]):
            if imgpath[i] in all_pro_query.keys():
                pro_img_list.append(all_pro_query[imgpath[i]])
            else:
                pro_img_list.append(img[i])
                logger.info("LACK OF PRO_IMG in {}".format(imgpath[i]))
        pro_img = torch.stack(pro_img_list, dim=0).to(device)

        pro_img = (pro_img / 2 + 0.5).clamp(0, 1)
        pro_img = pro_img.permute(0, 2, 3, 1)
        mean = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=pro_img.dtype, device=pro_img.device)
        std = torch.as_tensor(cfg.INPUT.PIXEL_STD, dtype=pro_img.dtype, device=pro_img.device)
        pro_img = pro_img[:, :, :].sub(mean).div(std)
        pro_img = pro_img.permute(0, 3, 1, 2)

        img = img.to(device)
        vid = torch.tensor(pid, dtype=torch.int64).to(device)
        camids = camids.to(device)
        target_view = target_view.to(device)
        batch_size_per_gpu = img.shape[0]

        # img = train_transforms(img)
        # pro_img = train_transforms(pro_img)

        

        # img 2 latent
        # batch_prompt = list(description)
        # batch_latent = latent_pre.to(device)
        # batch_latent.requires_grad_(True)
        # # batch_latent = get_batch_latent(cfg, img, batch_prompt, ldm_stable)
        # ori_latent = batch_latent.clone()

        # optimizer_latent = torch.optim.AdamW(params=[{"params": [batch_latent]}], lr=1e-1, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

        # # text embedding & uncond embedding & DDIM context
        # max_length = 77
        # batch_uncond_input = ldm_stable.tokenizer(
        #     [""] * batch_size_per_gpu, padding="max_length", max_length=max_length, return_tensors="pt"
        # )
        # bathc_uncond_embeddings = ldm_stable.text_encoder(batch_uncond_input.input_ids.to(ldm_stable.device))[0].detach()
        # batch_text_input = ldm_stable.tokenizer(
        #     batch_prompt,
        #     padding="max_length",
        #     max_length=ldm_stable.tokenizer.model_max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        # bacth_text_embeddings = ldm_stable.text_encoder(batch_text_input.input_ids.to(ldm_stable.device))[0].detach()
        # batch_uncond_text = torch.cat((bathc_uncond_embeddings,bacth_text_embeddings), dim=0)
        # batch_context = batch_uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)
        
        # # Strat optimizing latent
        # iterations = 30
        # pbar = tqdm(range(iterations), desc="Iterations")

        # for step, _ in enumerate(pbar):

        #     optimizer_latent.zero_grad()
            
        #     pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)

        #     # save_image(img, pro_img, n_iter, cfg, step, True, 'query')

        #     out_image = (pro_img / 2 + 0.5).clamp(0, 1)
        #     out_image = out_image.permute(0, 2, 3, 1)
        #     mean = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=out_image.dtype, device=out_image.device)
        #     std = torch.as_tensor(cfg.INPUT.PIXEL_STD, dtype=out_image.dtype, device=out_image.device)
        #     out_image = out_image[:, :, :].sub(mean).div(std)
        #     out_image = out_image.permute(0, 3, 1, 2)

        #     # _, pro_feat = model_supervisor(out_image, vid, cam_label=camids, view_label=target_view)
        #     # _, ori_feat = model_supervisor(img, vid, cam_label=camids, view_label=target_view)

        #     # loss_5 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(pro_feat, ori_feat,  (-1 * torch.ones(pro_feat.shape[0])).to(device))
        #     # print("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_5))
        #     # logger.info("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_5))
            
        #     # loss_1 = torch.nn.MSELoss(reduction='mean')(ori_latent, batch_latent)

        #     pro_feat_bn, pro_feat = model_supervisor(out_image, vid, cam_label=camids, view_label=target_view)
        #     ori_feat_bn, ori_feat = model_supervisor(img, vid, cam_label=camids, view_label=target_view)

        #     loss_1 = torch.nn.L1Loss(reduction='mean')(batch_latent, ori_latent)
        #     print("loss_1: MSELoss between ori_lt & pro_lt: {:.4f}".format(loss_1))

        #     loss_2 = torch.nn.L1Loss(reduction='mean')(out_image, img)
        #     print("loss_2: MSELoss between out_image & img: {:.4f}".format(loss_2))

        #     loss_3 = torch.nn.MSELoss(reduction='mean')(pro_feat, ori_feat)
        #     logger.info("loss_3: MSELoss between pro_feat & ori_feat: {:.4f}".format(loss_3))

        #     loss_7 = torch.nn.MSELoss(reduction='mean')(pro_feat_bn, ori_feat_bn)
        #     logger.info("loss_3: MSELoss between pro_feat_bn & ori_feat_bn: {:.4f}".format(loss_7))

        #     loss_5 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(pro_feat, ori_feat, (-1 * torch.ones(pro_feat.shape[0])).to(device))
        #     print("loss_5: Before BN Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_5))
        #     logger.info("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_5))

        #     loss_6 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(pro_feat_bn, ori_feat_bn, (-1 * torch.ones(pro_feat.shape[0])).to(device))
        #     print("loss_6: After BN Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_6))
        #     logger.info("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_6))
        
        #     loss = loss_5 + 10 * loss_2 + 5 * loss_1 + 2 * loss_6 - 0.1 * loss_3

        #     pbar.set_postfix_str(
        #         f"loss_1: {loss_1.item():.5f}"
        #         f"loss_5: {loss_5.item():.5f}"
        #         f"loss: {loss.item():.5f}")
            
        #     loss.backward()
            
        #     optimizer_latent.step()

        with torch.no_grad():

            # pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)
            # for i in range(img.shape[0]):
            #     all_pro_query[imgpath[i]]=pro_img[i].detach().cpu()
            # save_image(img, pro_img, n_iter, cfg, step+1, True, 'query')

            feat_pro, _, score_ori = model_reid(img, cam_label=camids, view_label=target_view)
            
            feat_ori, _, _ = model_reid(img, cam_label=camids, view_label=target_view)

            feat_sup, feat_sup_bfbn, score_pro = model_reid(pro_img, cam_label=camids, view_label=target_view)

            feat_reid, feat_reid_bfbn, score_reid = model_supervisor(pro_img, cam_label=camids, view_label=target_view)

            feat_reid_ori, feat_reid_bfbn, score_reid = model_supervisor(img, cam_label=camids, view_label=target_view)

            '''SOLID'''
            # feat_pro, _ = model_reid(img, cam_label=camids, view_label=target_view)
            # feat_ori, _ = model_reid(img, cam_label=camids, view_label=target_view)
            # feat_sup, _ = model_reid(pro_img, cam_label=camids, view_label=target_view)
            '''Trans'''
            # feat_pro= model_reid(img, cam_label=camids, view_label=target_view)
            # feat_ori= model_reid(img, cam_label=camids, view_label=target_view)
            # feat_sup= model_reid(pro_img, cam_label=camids, view_label=target_view)

            '''MSI'''
            # feat_pro = model_reid(img)
            # feat_ori = model_reid(img)
            # feat_sup = model_reid(pro_img)
            
            '''AGW'''
            # feat_pro = model_reid(img)
            # feat_ori = model_reid(img)
            # feat_sup = model_reid(pro_img)


            # acc = (score_ori.max(1)[1] == vid).float().mean()
            # acc_pro = (score_pro.max(1)[1] == vid).float().mean()
            # print(acc)
            # print(acc_pro)
            # a = tri_loss(feat_sup_bfbn, vid)
            # print(a)

            # feat_msi = model_msi(img)
            
            evaluator_1.update((feat_pro, pid, camid))
            evaluator_2.update((feat_ori, pid, camid))
            evaluator_3.update((feat_sup, pid, camid))
            # evaluator_4.update((feat_msi, pid, camid))

            orimodel_ori_query_feature.append(feat_reid_ori.cpu().numpy())

            sysmodel_ori_query_feature.append(feat_ori.cpu().numpy())

            sysmodel_pro_query_feature.append(feat_sup.cpu().numpy())

            query_labels.append(np.asarray(pid))

   
        end_time = time.time()
        all_time = end_time - start_time
        print('\n')
        print("Each batch need {:.4f}s".format(end_time-start_time))
        print('\n')
    

    for n_iter, (img, pid, camid, camids, target_view, imgpath, latent_pre, description) in enumerate(gallery_loader):

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
        pro_img = pro_img[:, :, :].sub(mean).div(std)
        pro_img = pro_img.permute(0, 3, 1, 2)
        
        img = img.to(device)
        vid = torch.tensor(pid, dtype=torch.int64).to(device)
        camids = camids.to(device)
        target_view = target_view.to(device)
        batch_size_per_gpu = img.shape[0]

        # img = train_transforms(img)
        # pro_img = train_transforms(pro_img)


        # img 2 latent
        # batch_prompt = list(description)
        # batch_latent = latent_pre.to(device)
        # batch_latent.requires_grad_(True)
        # # batch_latent = get_batch_latent(cfg, img, batch_prompt, ldm_stable)
        # ori_latent = batch_latent.clone()

        # optimizer_latent = torch.optim.AdamW(params=[{"params": [batch_latent]}], lr=1e-1, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

        # # text embedding & uncond embedding & DDIM context
        # max_length = 77
        # batch_uncond_input = ldm_stable.tokenizer(
        #     [""] * batch_size_per_gpu, padding="max_length", max_length=max_length, return_tensors="pt"
        # )
        # bathc_uncond_embeddings = ldm_stable.text_encoder(batch_uncond_input.input_ids.to(ldm_stable.device))[0].detach()
        # batch_text_input = ldm_stable.tokenizer(
        #     batch_prompt,
        #     padding="max_length",
        #     max_length=ldm_stable.tokenizer.model_max_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        # bacth_text_embeddings = ldm_stable.text_encoder(batch_text_input.input_ids.to(ldm_stable.device))[0].detach()
        # batch_uncond_text = torch.cat((bathc_uncond_embeddings,bacth_text_embeddings), dim=0)
        # batch_context = batch_uncond_text.unsqueeze(0).expand((cfg.DIFFUSION.NUM_INFERENCE_STEPS - cfg.DIFFUSION.START_STEP), -1, -1, -1)
        
        # # Strat optimizing latent
        # iterations = 30
        # pbar = tqdm(range(iterations), desc="Iterations")

        # for step, _ in enumerate(pbar):

        #     optimizer_latent.zero_grad()
            
        #     pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)

        #     # save_image(img, pro_img, n_iter, cfg, step, True, 'gallery')

        #     out_image = (pro_img / 2 + 0.5).clamp(0, 1)
        #     out_image = out_image.permute(0, 2, 3, 1)
        #     mean = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=out_image.dtype, device=out_image.device)
        #     std = torch.as_tensor(cfg.INPUT.PIXEL_STD, dtype=out_image.dtype, device=out_image.device)
        #     out_image = out_image[:, :, :].sub(mean).div(std)
        #     out_image = out_image.permute(0, 3, 1, 2)

        #     # _, pro_feat = model_supervisor(out_image, vid, cam_label=camids, view_label=target_view)
        #     # _, ori_feat = model_supervisor(img, vid, cam_label=camids, view_label=target_view)

        #     # loss_5 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(pro_feat, ori_feat,  (-1 * torch.ones(pro_feat.shape[0])).to(device))
        #     # print("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_5))
        #     # logger.info("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_5))

        #     # loss_1 = torch.nn.MSELoss(reduction='mean')(ori_latent, batch_latent)

        #     # loss = loss_1 + loss_5

        #     pro_feat_bn, pro_feat = model_supervisor(out_image, vid, cam_label=camids, view_label=target_view)
        #     ori_feat_bn, ori_feat = model_supervisor(img, vid, cam_label=camids, view_label=target_view)

        #     loss_1 = torch.nn.L1Loss(reduction='mean')(batch_latent, ori_latent)
        #     print("loss_1: MSELoss between ori_lt & pro_lt: {:.4f}".format(loss_1))

        #     loss_2 = torch.nn.L1Loss(reduction='mean')(out_image, img)
        #     print("loss_2: MSELoss between out_image & img: {:.4f}".format(loss_2))

        #     loss_3 = torch.nn.MSELoss(reduction='mean')(pro_feat, ori_feat)
        #     logger.info("loss_3: MSELoss between pro_feat & ori_feat: {:.4f}".format(loss_3))

        #     loss_7 = torch.nn.MSELoss(reduction='mean')(pro_feat_bn, ori_feat_bn)
        #     logger.info("loss_3: MSELoss between pro_feat_bn & ori_feat_bn: {:.4f}".format(loss_7))

        #     loss_5 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(pro_feat, ori_feat, (-1 * torch.ones(pro_feat.shape[0])).to(device))
        #     print("loss_5: Before BN Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_5))
        #     logger.info("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_5))

        #     loss_6 = torch.nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)(pro_feat_bn, ori_feat_bn, (-1 * torch.ones(pro_feat.shape[0])).to(device))
        #     print("loss_6: After BN Feature CosineSimilarity between ori & pro: {:.4f}".format(loss_6))
        #     logger.info("loss_5: CosineSimilarity between vis_output: {:.4f}".format(loss_6))
        
        #     loss = loss_5 + 10 * loss_2 + 5 * loss_1 + 2 * loss_6 - 0.1 * loss_3

        #     pbar.set_postfix_str(
        #         f"loss_1: {loss_1.item():.5f}"
        #         f"loss_5: {loss_5.item():.5f}"
        #         f"loss: {loss.item():.5f}")
            
        #     loss.backward()
            
        #     optimizer_latent.step()

        with torch.no_grad():

        #     pro_img = latent_2_image(cfg, ldm_stable, batch_latent, batch_context)
        #     for i in range(img.shape[0]):
        #         all_pro_gallery[imgpath[i]]=pro_img[i].detach().cpu()
            # save_image(img, pro_img, n_iter, cfg, step+1, True, 'gallery')

            feat_pro, _, _ = model_reid(img, cam_label=camids, view_label=target_view)
            
            feat_ori, _, _ = model_reid(pro_img, cam_label=camids, view_label=target_view)
            # for module in model_reid.modules():
            #     if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.InstanceNorm2d)):
            #         module.train()
            feat_sup, feat_sup_bfbn, score_pro = model_reid(pro_img, cam_label=camids, view_label=target_view)

            feat_reid, feat_reid_bfbn, score_reid = model_supervisor(pro_img, cam_label=camids, view_label=target_view)

            feat_reid_ori, feat_reid_bfbn, score_reid = model_supervisor(img, cam_label=camids, view_label=target_view)
            

            '''SOLID'''
            # feat_pro, _ = model_reid(img, cam_label=camids, view_label=target_view)
            # feat_ori, _ = model_reid(pro_img, cam_label=camids, view_label=target_view)
            # feat_sup, _ = model_reid(pro_img, cam_label=camids, view_label=target_view)

            '''Trans'''
            # feat_pro= model_reid(img, cam_label=camids, view_label=target_view)
            # feat_ori= model_reid(pro_img, cam_label=camids, view_label=target_view)
            # feat_sup= model_reid(pro_img, cam_label=camids, view_label=target_view)

            '''MSI'''
            # feat_pro = model_reid(img)
            # feat_ori = model_reid(pro_img)
            # feat_sup = model_reid(pro_img)
            
            '''AGW'''
            # feat_pro = model_reid(img)
            # feat_ori = model_reid(pro_img)
            # feat_sup = model_reid(pro_img)

            # feat_msi = model_msi(pro_img)

            # acc = (score_pro.max(1)[1] == vid).float().mean()

            # print(acc)
            
            evaluator_1.update((feat_pro, pid, camid))
            evaluator_2.update((feat_ori, pid, camid))
            evaluator_3.update((feat_sup, pid, camid))
            # evaluator_4.update((feat_msi, pid, camid))


            orimodel_pro_gallery_feature.append(feat_reid.cpu().numpy())

            sysmodel_ori_gallery_feature.append(feat_pro.cpu().numpy())

            sysmodel_pro_gallery_feature.append(feat_sup.cpu().numpy())

            gallery_labels.append(np.asarray(pid))

    orimodel_ori_query_feature = np.concatenate(orimodel_ori_query_feature, axis=0)
    orimodel_pro_gallery_feature = np.concatenate(orimodel_pro_gallery_feature, axis=0)
    sysmodel_ori_query_feature = np.concatenate(sysmodel_ori_query_feature, axis=0)
    sysmodel_ori_gallery_feature = np.concatenate(sysmodel_ori_gallery_feature, axis=0)
    sysmodel_pro_query_feature = np.concatenate(sysmodel_pro_query_feature, axis=0)
    sysmodel_pro_gallery_feature = np.concatenate(sysmodel_pro_gallery_feature, axis=0)
    query_label = np.concatenate(query_labels, axis=0)
    gallery_label = np.concatenate(gallery_labels, axis=0)

    visualize_tsne(orimodel_ori_query_feature, orimodel_pro_gallery_feature, sysmodel_ori_query_feature, sysmodel_ori_gallery_feature, sysmodel_pro_query_feature, sysmodel_pro_gallery_feature, query_label, gallery_label, output_path=cfg.OUTPUT_DIR)

    # torch.save(all_pro_query, '/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/pro_query_2.pth')
    # torch.save(all_pro_gallery, '/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/pro_gallery_2.pth')
    cmc, mAP, _, _, _, _, _ = evaluator_1.compute()
    cmc2, mAP2, _, _, _, _, _ = evaluator_2.compute()
    cmc3, mAP3, _, _, _, _, _ = evaluator_3.compute()
    # cmc4, mAP4, _, _, _, _, _ = evaluator_4.compute()
    logger.info("Validation Results ")
    logger.info("ori2ori_mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("ori2ori_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    # return cmc[0], cmc[4]

    logger.info("ori2pro_mAP: {:.1%}".format(mAP2))
    for r in [1, 5, 10]:
        logger.info("ori2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
    # return cmc2[0], cmc2[4]
        
    logger.info("pro2pro_mAP: {:.1%}".format(mAP3))
    for r in [1, 5, 10]:
        logger.info("pro2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc3[r - 1]))
    # return cmc2[0], cmc2[4]

    # logger.info("MSIori2pro_mAP: {:.1%}".format(mAP4))
    # for r in [1, 5, 10]:
    #     logger.info("ori2pro_CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc4[r - 1]))

