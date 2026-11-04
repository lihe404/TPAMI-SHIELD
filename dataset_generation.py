import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

from processor import generate_trainset, generate_testset

# from MSINet_in.reid.models import msinet_x1_0
# from TransReID_in.model import make_model as make_transreid
# from modeling import build_model as make_AGW
# from SOLIDER_in.model import make_model as make_SOLIDER
# from TransReID_in.config import cfg as transreid_config
# from AGW_in.config import cfg as AGW_config
# from SOLIDER_in.config import cfg as SOLIDER_config
# # from OSNet.torchreid.models import osnet_x1_0
# # from PCB.bpm.model.PCBModel import PCBModel as PCB
# from PCB_2.core.model import Model

def load_PCB(model, src_state_dict):
  """Copy parameters and buffers from `src_state_dict` into `model` and its 
  descendants. The `src_state_dict.keys()` NEED NOT exactly match 
  `model.state_dict().keys()`. For dict key mismatch, just
  skip it; for copying error, just output warnings and proceed.

  Arguments:
    model: A torch.nn.Module object. 
    src_state_dict (dict): A dict containing parameters and persistent buffers.
  Note:
    This is modified from torch.nn.modules.module.load_state_dict(), to make
    the warnings and errors more detailed.
  """
  from torch.nn import Parameter

  dest_state_dict = model.state_dict()
  for name, param in src_state_dict.items():
    if name not in dest_state_dict:
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    try:
      dest_state_dict[name].copy_(param)
    except Exception as msg:
      print("Warning: Error occurs when copying '{}': {}"
            .format(name, str(msg)))

  src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
  if len(src_missing) > 0:
    print("Keys not found in source state_dict: ")
    for n in src_missing:
      print('\t', n)

  dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
  if len(dest_missing) > 0:
    print("Keys not found in destination state_dict: ")
    for n in dest_missing:
      print('\t', n)

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
        "--config_file", default="/home/mengqianxiang/code/TPAMI_PPPR/configs/market1501/vit_small_ice_generation.yml", help="path to config file", type=str
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

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, query_loader, gallery_loader, train_loader_st2, val_loader_st2, query_loader_st2, gallery_loader_st2, _ = make_dataloader(cfg)
    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, query_loader, gallery_loader = make_dataloader(cfg)

    # ReID model & Loss_func & Optimizer & Scheduler
    if cfg.MODEL.NAME =='transformer':
        model_reid = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
   
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
    optimizer_reid, optimizer_center_reid = make_optimizer(cfg, model_reid, center_criterion_reid)

    # print("--------------------MSINet-----------------------")
    # print('\n')
    # model_msi = msinet_x1_0(cfg, num_classes)


    # # print("--------------------OSNet-----------------------")
    # # print('\n')

    # # model_osnet=osnet_x1_0(num_classes)

    # print("--------------------transfromer_base-----------------------")
    # print('\n')

    # transfromer_base_config = transreid_config.clone()
    # transfromer_base_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/TransReID_in/configs/Market/vit_base.yml")
    # model_transformer_base = make_transreid(transfromer_base_config, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    # model_transformer_base.load_param(transfromer_base_config.TEST.WEIGHT)

    # print('\n')
    # print("--------------------transreid_base-----------------------")
    # print('\n')

    # transreid_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/TransReID_in/configs/Market/vit_transreid_stride.yml")
    # model_transreid = make_transreid(transreid_config, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    # model_transreid.load_param(transreid_config.TEST.WEIGHT)

    # print('\n')
    # print("--------------------AGW-----------------------")
    # print('\n')

    # AGW_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/AGW_in/configs/AGW_baseline.yml")
    # model_AGW = make_AGW(AGW_config, num_classes)
    # model_AGW.load_param(AGW_config.TEST.WEIGHT)

    # print('\n')
    # print("--------------------SOLIDER-----------------------")
    # print('\n')
    
    # SOLIDER_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/SOLIDER_in/configs/market/swin_base.yml")
    # model_SOLIDER = make_SOLIDER(SOLIDER_config, num_class=num_classes, camera_num=camera_num, view_num = view_num, semantic_weight = SOLIDER_config.MODEL.SEMANTIC_WEIGHT)
    # model_SOLIDER.load_param(SOLIDER_config.TEST.WEIGHT)


    # print("--------------------PCB-----------------------")
    # print('\n')
    # model_PCB = Model(part_num=6, class_num=751)
    # a = torch.load("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/PCB_2/results/model.pth")
    # # a = torch.load("/data1/mqx_log/Pretrained_Weight/PCB/pcb_market.pth")
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in a.items():
    #     name = k.replace("module.", "")  # 去掉 "module."
    #     new_state_dict[name] = v
    # model_PCB.load_state_dict(new_state_dict)

    # print("--------------------ResNet-----------------------")
    # print('\n')
    # AGW_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/AGW_in/configs/resnet50.yml")
    # model_ResNet = make_AGW(AGW_config, num_classes)
    # model_ResNet.load_param(AGW_config.TEST.WEIGHT)
    generate_trainset(
        cfg,
        model_reid,
        model_supervisor,
        train_loader,
        args.local_rank
    )

    # generate_testset(
    #     cfg,
    #     model_reid,
    #     model_supervisor,
    #     val_loader,
    #     train_loader,
    #     query_loader,
    #     gallery_loader,
    #     num_query, args.local_rank, model_msi, model_transformer_base, model_transreid, model_AGW, model_SOLIDER, model_PCB, model_ResNet, query_loader_st2, gallery_loader_st2
    # )

