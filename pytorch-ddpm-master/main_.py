import copy
import json
import os
import warnings

import torch
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
import torchvision

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
from score.both import get_inception_and_fid_score

from utils.utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Resize, CenterCrop
from tqdm import tqdm
from dataset import dataset_norm

from torchvision import transforms


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 490000+500000, help='total training steps')
flags.DEFINE_integer('img_size', 128, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 1, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', '/content/drive/MyDrive/IITD/I3FDM_Distance_db1', help='log directory')
flags.DEFINE_integer('sample_size', 36, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
# Valid
flags.DEFINE_integer('val_step', 5000,
                     help='frequency of computing validation loss, 0 to disable')

flags.DEFINE_string('resume', "/content/drive/MyDrive/IITD/ckpt_last.pt", help='path to checkpoint to resume training')
flags.DEFINE_integer('accum_steps', 6, help='gradient accumulation steps')

device = torch.device('cuda:0')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]   # (img, label, ...)이면 img만 사용
            else:
                x = batch      # img만 나오는 경우
            yield x

def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

@torch.no_grad()
def generate_and_save_images(sampler, num_images, save_dir):
    """
    DDPM sampler로부터 num_images 장을 생성해서
    save_dir에 000000.png, 000001.png ... 형태로 저장
    """
    os.makedirs(save_dir, exist_ok=True)

    model_device = next(sampler.parameters()).device  # sampler가 올라간 device
    img_idx = 0

    for i in trange(0, num_images, FLAGS.batch_size, desc="saving images"):
        batch_size = min(FLAGS.batch_size, num_images - i)
        # noise x_T
        x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size), device=model_device)
        # 생성
        batch_images = sampler(x_T)          # [-1, 1] 범위라고 가정
        batch_images = (batch_images + 1) / 2  # [0, 1] 범위로 변환

        # 한 장씩 저장
        for b in range(batch_size):
            img = batch_images[b].clamp(0.0, 1.0)
            filename = os.path.join(save_dir, f"{img_idx:06d}.png")
            save_image(img, filename)
            img_idx += 1

    print(f"Saved {img_idx} images to {save_dir}")


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():
    # dataset
    tds = glob("/content/dataset/db1_224_for_gt_inpainting_trainset", '*', True)

    train_ls = tds[:int(len(tds) * 0.8)]
    valid_ls = tds[int(len(tds) * 0.8):]

    # target train list
    ttls = []
    for path in train_ls:
        ttls += glob(path, '*', True)

    # target valid list
    tvls = []
    for path in valid_ls:
        tvls += glob(path, '*', True)

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    valid_transformations = transforms.Compose([
        transforms.Resize((128, 128)),
        ToTensor(),
        Normalize(mean, std)
    ])

    transformations = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean, std)
    ])  # augmentation

    train_dataset = dataset_norm(
        root='', transforms=transformations, imglist=ttls)
    train_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True,
                              num_workers=FLAGS.num_workers, drop_last=True)
    print('train data: %d images' % (len(train_loader.dataset)))

    valid_dataset = dataset_norm(
        root='', transforms=valid_transformations, imglist=tvls)
    valid_loader = DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False,
                              num_workers=FLAGS.num_workers, drop_last=False)
    print('valid data: %d images' % (len(valid_loader.dataset)))

    train_datalooper = infiniteloop(train_loader)

    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # ✅ 추가: resume 로직
    if FLAGS.resume is not None:
        print(f"Resuming training from checkpoint: {FLAGS.resume}")
        ckpt = torch.load(FLAGS.resume, map_location=device)
        net_model.load_state_dict(ckpt['net_model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        start_step = ckpt.get('step', 0)
    else:
        start_step = 0
    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)

    # x_T: sampling용 noise
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)

    first_batch = next(iter(train_loader))
    if isinstance(first_batch, (tuple, list)):
        real_imgs = first_batch[0]
    else:
        real_imgs = first_batch
    grid = (make_grid(real_imgs[:FLAGS.sample_size]) + 1) / 2
    # grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2

    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()

    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    best_val_loss = float('inf')
    last_val_loss = None

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()

            total_loss = 0.0

            for accum_i in range(FLAGS.accum_steps):
                x_0 = next(train_datalooper).to(device)

                loss = trainer(x_0).mean()
                loss = loss / FLAGS.accum_steps  # ⭐ 매우 중요
                loss.backward()

                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip
            )

            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # validation loss (주기적으로)
            if FLAGS.val_step > 0 and step % FLAGS.val_step == 0 and step > 0:
                val_loss = validate(trainer, valid_loader)
                last_val_loss = val_loss
                writer.add_scalar('loss/val', val_loss, step)
                pbar.write(f"[step {step}] val_loss: {val_loss:.4f}")

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }

                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt_last.pt'))

                if last_val_loss is not None and last_val_loss < best_val_loss:
                    best_val_loss = last_val_loss
                    torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt_best.pt'))

                # torch.save(ckpt, os.path.join(FLAGS.logdir, f'ckpt_{step}.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()

@torch.no_grad()
def validate(trainer, valid_loader):
    """validation 데이터에서 DDPM loss 평균을 계산"""
    trainer.eval()
    total_loss = 0.0
    total_count = 0

    for batch in valid_loader:
        # dataset_norm 이 (img, label, ...) 형태라면 첫 번째만 사용
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device)
        loss = trainer(x).mean()
        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    trainer.train()
    if total_count == 0:
        return 0.0
    return total_loss / total_count

def eval():
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt_best.pt'))  # or ckpt_last.pt
    model.load_state_dict(ckpt['ema_model'])

    # (IS, IS_std), FID, samples = evaluate(sampler, model)
    # print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    # save_image(
    #     torch.tensor(samples[:256]),
    #     os.path.join(FLAGS.logdir, 'samples_ema.png'),
    #     nrow=16)

    save_dir = os.path.join(FLAGS.logdir, "generated_images")
    generate_and_save_images(sampler, num_images=FLAGS.num_images, save_dir=save_dir)


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
