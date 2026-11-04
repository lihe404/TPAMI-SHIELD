import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model, make_latent
# from MSINet_in.reid.models import msinet_x1_0
from processor import do_inference_dif_latent
from loss import make_loss
from utils.logger import setup_logger
import torch

# from TransReID_in.model import make_model as make_transreid
# from modeling import build_model as make_AGW
# from SOLIDER_in.model import make_model as make_SOLIDER
# from TransReID_in.config import cfg as transreid_config
# from AGW_in.config import cfg as AGW_config
# from SOLIDER_in.config import cfg as SOLIDER_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="/home/mengqianxiang/code/TPAMI_PPPR/configs/market1501/vit_small_ics.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.TEST.EVAL = True
    
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger(cfg, "transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num, query_loader, gallery_loader, train_loader_st2, val_loader_st2, query_loader_st2, gallery_loader_st2, _ = make_dataloader(cfg)

    loss_func_reid, center_criterion_reid, reverse_tri_loss, reverse_loss_func, tri_loss = make_loss(cfg, num_classes=num_classes)

    model_reid = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model_supervisor = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, supervisor=True)
    
    
    # model_msi = msinet_x1_0(cfg, num_classes)

    # SOLIDER_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/SOLIDER_in/configs/market/swin_base.yml")
    # model_reid = make_SOLIDER(SOLIDER_config, num_class=num_classes, camera_num=camera_num, view_num = view_num, semantic_weight = SOLIDER_config.MODEL.SEMANTIC_WEIGHT)
    # model_reid.load_param("/data1/mqx_log/Privacy/PPReID/msmt/attention_cosine/towardsperfect_fromsup/SOLID/transformermodel_reid_40.pth")

    # transreid_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/TransReID_in/configs/Market/vit_transreid_stride.yml")
    # model_reid = make_transreid(transreid_config, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    # model_reid.load_param(transreid_config.TEST.WEIGHT)

    # AGW_config.merge_from_file("/home/mengqianxiang/code/Test/TransReID-SSL/transreid_pytorch/AGW_in/configs/AGW_baseline.yml")
    # model_reid = make_AGW(AGW_config, num_classes)
    # model_reid.load_param(AGW_config.TEST.WEIGHT)


    do_inference_dif_latent(cfg,
                model_reid,
                model_supervisor,
                val_loader,
                train_loader_st2,
                query_loader_st2,
                gallery_loader_st2,
                num_query,
                tri_loss)

