import argparse
import torch
import clip
from torch import optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import ssl
import importlib

ssl._create_default_https_context = ssl._create_unverified_context

import torch.distributed as dist
import torch.multiprocessing as mp

import sys
import os

cur = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(cur, ".."))
if root not in sys.path:
    sys.path.insert(0, root)


from config.configs import cfg_from_file
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens

# Fixed training seed (GENERALIZATION_PROTOCOL.md section 8.4, 2026-07-11): covers
# Python/NumPy/PyTorch/CUDA; DataLoader workers reseed via _seed_worker + generator.
SEED = 0


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def custom_collate_fn(batch):
    imgs, labels, metas, filenames, pseudo_classes = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels, metas, filenames, pseudo_classes
    

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/voc_train_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='RECLIPPP', type=str)
    parser.add_argument('--model_module', default='model.model',
                        help='Python module that provides RECLIPPP/ReCLIP classes.')
    parser.add_argument('--distributed', action='store_true',
                        help='Enable DistributedDataParallel. Default is local single-process training.')
    args = parser.parse_args()
    return args


def load_model_classes(module_name):
    module = importlib.import_module(module_name)
    return getattr(module, 'RECLIPPP'), getattr(module, 'ReCLIP', None)


def build_model(model_cls, cfg, clip_model, rank, text_weight):
    try:
        return model_cls(cfg=cfg, clip_model=clip_model, rank=rank, zeroshot_weights=text_weight)
    except TypeError:
        return model_cls(cfg=cfg, clip_model=clip_model, rank=rank)


class Train(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_filenames, _, self.train_images, self.train_labels, _, _, _, self.pseudo_classes = read_file_list(cfg)

    def __getitem__(self, idx):
        with open(self.train_images[idx], 'rb') as f:
            value_buf = f.read()
        with open(self.train_labels[idx], 'rb') as f:
            label_buf = f.read()
        img, label, img_metas = preprocess(self.cfg, value_buf, label_buf, return_meta=True, unlabeled=False)
        return img, label, img_metas, self.train_images[idx], self.pseudo_classes[idx]

    def __len__(self):
        return len(self.train_images)


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(rank, world_size):
    args = get_parser()
    distributed = bool(args.distributed and world_size > 1)
    if distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)
    clip_model = clip_model.to(device)

    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    RECLIPPP, ReCLIP = load_model_classes(args.model_module)
    log = open('experiments/log_voc_rectification.txt', mode='a')
    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(cfg)
    cls_name_token, classes = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)

    train_data = Train(cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if distributed else None
    train_loader = DataLoader(dataset=train_data, shuffle=train_sampler is None, num_workers=cfg.NUM_WORKERS,
                              pin_memory=torch.cuda.is_available(), sampler=train_sampler,
                              batch_size=cfg.TRAIN.BATCH_SIZE, collate_fn=custom_collate_fn,
                              generator=torch.Generator().manual_seed(SEED),
                              worker_init_fn=_seed_worker)
    
    if args.model_name == 'RECLIPPP':
        model = build_model(RECLIPPP, cfg, clip_model, device, text_weight).to(device)
    else:
        model = build_model(ReCLIP, cfg, clip_model, device, text_weight).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank,
                                                          find_unused_parameters=True)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR, momentum=0.9,
                          weight_decay=0.0005)
    max_epoch = cfg.TRAIN.MAX_EPOCH
    if cfg.TRAIN.EPOCH >= 0:
        stop_epoch = cfg.TRAIN.EPOCH
    else:
        stop_epoch = max_epoch
    c_num = cfg.DATASET.NUM_CLASSES
    best_iou = 0.0
    for epoch in range(max_epoch):
        idx = 0
        model.train()
        running_loss = 0.0

        lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch, cfg.TRAIN.LR, power=0.9)
        loop = tqdm(train_loader)

        for img, label, img_metas, filenames, pseudo_class in loop:
            gt_cls = []
            batch_size = img.shape[0]
            for i in range(batch_size):
                temp = [int(tensor) if isinstance(tensor, int) else int(tensor.item()) for tensor in pseudo_class[i]]
                gt_cls.append(temp)

                if len(temp) == 0:
                    continue
            if len(gt_cls[0]) == 0:
                continue
            output, loss = model(img.to(device), gt_cls, text_weight, cls_name_token, training=True, img_metas=img_metas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            loop.set_postfix(epoch=epoch, img_loss=loss.item(), avg_loss=running_loss / (idx + 1))
            print('filenames:{}, img_idx:{}, img_loss:{:.5f}, avg_loss:{:.5f}'.format(filenames, idx, loss.item(), running_loss / (idx + 1)), file=log)
            idx += 1
        print('epoch {} finish, lr:{}'.format(epoch, lr), file=log)

        if rank == 0:
            model.eval()
            success_num = 0
            with torch.no_grad():
                for idx in range(len(val_images)):
                    with open(val_images[idx], 'rb') as f:
                        value_buf = f.read()
                    img = val_preprocess(cfg, value_buf).unsqueeze(dim=0)
                    label = Image.open(val_labels[idx])
                    ori_shape = tuple((label.size[1], label.size[0]))
                    label = np.asarray(label)
                    gt_cls = []
                    label_cls = set(label.flatten().tolist()[1:])
                    for cls in label_cls:
                        if cls != 0 and cls != 255:
                            gt_cls.append(cls - 1)
                    shape = img.shape[2:]
                    output = model(img.to(device), gt_cls, text_weight, cls_name_token, training=False)

                    output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
                    output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0], ori_shape[1])

                    output = F.softmax(output, dim=1)
                    output = torch.argmax(output, dim=1).squeeze(dim=0)
                    torch.save(output, cfg.SAVE_DIR + val_filenames[idx] + '.pt')
                    success_num += 1

                    if (idx + 1) % 100 == 0 or idx + 1 == len(val_images):
                        print('validation progress: {}/{}'.format(idx + 1, len(val_images)))

                iou = mean_iou(results_iou, val_labels, num_classes=c_num + 1, ignore_index=255, nan_to_num=0, reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
                print(iou['IoU'])
                avg = iou['IoU'].sum() / c_num
                print('avg:%.4f' % (avg))
                print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (success_num, len(val_images), avg))
                print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (success_num, len(val_images), avg), file=log)
                log.write('miou:{}'.format(avg))
                if avg > best_iou:
                    best_iou = avg
                    torch.save(model.state_dict(), cfg.SAVE_DIR + 'best_weight.pth')
        if epoch == stop_epoch:
            break
    log.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_parser()
    if args.distributed:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            raise RuntimeError("--distributed requires at least 2 CUDA devices.")
        mp.spawn(train,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        train(rank=0, world_size=1)
