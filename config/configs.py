import yaml
from easydict import EasyDict as edict

cfg = edict()

cfg.DATASET = edict()
cfg.DATASET.NAME = ''
cfg.DATASET.NUM_CLASSES = 0
cfg.DATASET.REDUCE_ZERO_LABEL = True
cfg.DATASET.DATAROOT = ''
cfg.DATASET.SCALE = []
cfg.DATASET.RATIO_RANGE = []
cfg.DATASET.CROP_SIZE = []
cfg.DATASET.CAT_MAX_RATIO = 0
cfg.DATASET.TEXT_WEIGHT = ''
cfg.DATASET.IMG_NORM_CFG = edict()
cfg.DATASET.IMG_NORM_CFG.MEAN = []
cfg.DATASET.IMG_NORM_CFG.STD = []
cfg.DATASET.IMG_NORM_CFG.RGB = True
cfg.DATASET.K = 0
cfg.DATASET.DISTILL_K = 0
cfg.DATASET.THRESHOLD = 0
cfg.DATASET.IGNORE_INDEX = 255
cfg.DATASET.PALETTE = []

cfg.MODEL = edict()
cfg.MODEL.FEATURE_EXTRACTOR = ''
cfg.MODEL.TEXT_CHANNEL = 0
cfg.MODEL.VISUAL_CHANNEL = 0
cfg.MODEL.TRAINING = False
cfg.MODEL.FEATURE_FUSION = edict()
cfg.MODEL.FEATURE_FUSION.ENABLE = False
cfg.MODEL.FEATURE_FUSION.LAYERS = [6, 9, 12]
cfg.MODEL.FEATURE_FUSION.INIT_GAMMA = 0.0
cfg.MODEL.FEATURE_FUSION.MODE = 'l12_only'
cfg.MODEL.FEATURE_FUSION.GAMMA9 = 0.20
cfg.MODEL.FEATURE_FUSION.GAMMA6 = 0.05
cfg.MODEL.FEATURE_FUSION.GATE_TEMP = 10.0
cfg.MODEL.FEATURE_FUSION.PRESERVE_LOSS_WEIGHT = 0.0
cfg.MODEL.CLASS_GATE = edict()
cfg.MODEL.CLASS_GATE.ENABLE = False
cfg.MODEL.CLASS_GATE.THRESHOLD = 0.20
cfg.MODEL.CLASS_GATE.TEMP = 10.0
cfg.MODEL.CLASS_GATE.LOG_BIAS_SCALE = 1.0

cfg.MODEL.SFP_DTLR = edict()
cfg.MODEL.SFP_DTLR.TOPK = 800
cfg.MODEL.SFP_DTLR.TOP_FRACTION = -1.0
cfg.MODEL.SFP_DTLR.CONF_THD = 0.97
cfg.MODEL.SFP_DTLR.CONF_SCALE = 10.0
cfg.MODEL.SFP_DTLR.LOGIT_BETA = 0.55
cfg.MODEL.SFP_DTLR.PROXY_LAMBDA = 2.0
cfg.MODEL.SFP_DTLR.PROXY_CONF_THD = 0.95
cfg.MODEL.SFP_DTLR.PROXY_KERNEL = 5
cfg.MODEL.SFP_DTLR.DTLR_BETA = 1.20
cfg.MODEL.SFP_DTLR.DTLR_SIGMA_S = 70.0
cfg.MODEL.SFP_DTLR.DTLR_SIGMA_S_REL = -1.0
cfg.MODEL.SFP_DTLR.DTLR_SIGMA_R = 1.50
cfg.MODEL.SFP_DTLR.DTLR_NUM_ITER = 1
cfg.MODEL.SFP_DTLR.DTLR_STRUCTURE_GAIN_THD = 0.00
cfg.MODEL.SFP_DTLR.DTLR_STRUCTURE_CLASSES = [4, 8, 10]
# Dataset-agnostic entropy-normalized reliability gate (de-VOC code fix 2026-07-08).
# ENTROPY_GATE off => original absolute max-prob CONF_THD/PROXY_CONF_THD gates (VOC-original).
cfg.MODEL.SFP_DTLR.ENTROPY_GATE = False
cfg.MODEL.SFP_DTLR.ENTROPY_TAU_UNREL = 0.0745
cfg.MODEL.SFP_DTLR.ENTROPY_TAU_REL = 0.1154

cfg.MODEL.PAMR = edict()
cfg.MODEL.PAMR.NUM_ITER = 10
cfg.MODEL.PAMR.DILATIONS = [1, 2, 4, 8, 12, 24]
# "token": refine at the baseline token-grid resolution (same hook as SFP+DTLR).
# "image": upsample logits to the original image resolution, run PAMR there
#          (where PAMR was designed to operate), then downsample back.
cfg.MODEL.PAMR.RESOLUTION = "token"

# Method A: trainable soft presence-calibration head (model.model_presence).
cfg.MODEL.PRESENCE = edict()
cfg.MODEL.PRESENCE.MODE = "zglobal"       # "zglobal" | "zglobal_dense"
cfg.MODEL.PRESENCE.INIT_FROM = ""         # official baseline ckpt to load+freeze before training
cfg.MODEL.PRESENCE.BCE_W = 1.0            # image-level presence BCE weight
cfg.MODEL.PRESENCE.NEG_POS_W = 0.2       # <1 => suppressing a present class costs 1/NEG_POS_W more
cfg.MODEL.PRESENCE.REG_W = 0.1           # baseline-preserving regularization weight

cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 1
cfg.TRAIN.MAX_EPOCH = 50
cfg.TRAIN.EPOCH = 0
cfg.TRAIN.MAX_ITER = 0
cfg.TRAIN.LR = 0
cfg.TRAIN.LOG = ''

cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 0
cfg.TEST.PD = 0
cfg.TEST.ReCLIP_PD = 0.5

cfg.EVAL_METRIC = ''
cfg.SAVE_DIR = ''
cfg.NUM_WORKERS = 0
cfg.LOAD_PATH = ''
cfg.LOAD_DISTILL_PATH = ''

def merge_a_to_b(a, b):
    if type(a) is not edict:
        return
    for k in a:
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        if type(a[k]) is edict:
            merge_a_to_b(a[k], b[k])
        else:
            b[k] = a[k]
    return cfg


def cfg_from_file(filename):

    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    merge_a_to_b(yaml_cfg, cfg)
    return cfg

