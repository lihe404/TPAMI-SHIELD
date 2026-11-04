import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import torch
import numpy as np
import argparse
from config import cfg
import torch.distributed as dist

from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer, WarmupMultiStepLR
from solver.scheduler_factory import create_scheduler
from loss import make_loss

from processor import do_train_dif_latent

# from MSINet_in.reid.models import msinet_x1_0
# from TransReID_in.model import make_model as make_transreid
# from modeling import build_model as make_AGW
# from SOLIDER_in.model import make_model as make_SOLIDER
# from TransReID_in.config import cfg as transreid_config
# from AGW_in.config import cfg as AGW_config
# from SOLIDER_in.config import cfg as SOLIDER_config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="/home/mengqianxiang/code/TPAMI_PPPR/configs/market1501/vit_small_ics.yml", help="path to config file", type=str
    )
    # parser.add_argument(
    #     "--mamaba", default=False, help="path to config file", type=str
    # )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    # # if args.mamba:
    # if 'mamba' in cfg.MODEL.NAME:
    #     cfg.merge_from_file(cfg.MODEL.CONFIG)

    cfg.merge_from_list(args.opts)
    
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    try:
        os.makedirs(output_dir)
    except:
        pass

    logger = setup_logger(cfg, "transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    #  logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            #  logger.info(config_str)

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, query_loader, gallery_loader, train_loader_st2, val_loader_st2, query_loader_st2, gallery_loader_st2, train_loader_st2_pro = make_dataloader(cfg)

    # ReID model & Loss_func & Optimizer & Scheduler
    if cfg.MODEL.NAME =='transformer':
        model_reid = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    
    # model_reid = msinet_x1_0(cfg, num_classes)

    # SOLIDER_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/SOLIDER_in/configs/market/swin_base.yml")
    # model_reid = make_SOLIDER(SOLIDER_config, num_class=num_classes, camera_num=camera_num, view_num = view_num, semantic_weight = SOLIDER_config.MODEL.SEMANTIC_WEIGHT)
    # model_reid.load_param(SOLIDER_config.TEST.WEIGHT)

    # transreid_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/TransReID_in/configs/Market/vit_transreid_stride.yml")
    # model_reid = make_transreid(transreid_config, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    # model_reid.load_param(transreid_config.TEST.WEIGHT)

    # AGW_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/AGW_in/configs/AGW_baseline.yml")
    # model_reid = make_AGW(AGW_config, num_classes)
    # model_reid.load_param(AGW_config.TEST.WEIGHT)

    # elif 'mamba' in cfg.MODEL.NAME:
    #     model_reid = make_mamba(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func_reid, center_criterion_reid, reverse_tri_loss, reverse_loss_func, tri_loss = make_loss(cfg, num_classes=num_classes)
    optimizer_reid, optimizer_center_reid = make_optimizer(cfg, model_reid, center_criterion_reid)
    

    if cfg.SOLVER.WARMUP_METHOD == 'cosine':
        logger.info('===========using cosine learning rate=======')
        scheduler = create_scheduler(cfg, optimizer_reid)
    else:
        logger.info('===========using normal learning rate=======')
        scheduler = WarmupMultiStepLR(optimizer_reid, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)

    model_supervisor = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, supervisor=True)
    optimizer_sup, optimizer_center_sup = make_optimizer(cfg, model_supervisor, center_criterion_reid)
  
    do_train_dif_latent(
        cfg,
        model_reid,
        center_criterion_reid,
        train_loader,
        val_loader,
        train_loader_st2_pro,
        train_loader_st2,
        val_loader_st2,
        optimizer_reid,
        optimizer_center_reid,
        scheduler,
        loss_func_reid,
        reverse_tri_loss,
        tri_loss,
        num_query, args.local_rank
    )

