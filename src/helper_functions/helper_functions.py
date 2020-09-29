import time
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def create_dataloader(args):
    val_bs = args.batch_size
    val_tfms = transforms.Compose(
        [transforms.Resize(int(args.input_size ))]) # suish
    val_tfms.transforms.append(transforms.ToTensor())
    val_dataset = ImageFolder(args.val_dir, val_tfms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    return val_loader


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self): self.reset()

    def reset(self): self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(model, val_loader):
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()
    last_idx = len(val_loader) - 1

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(val_loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()
            output = model(input)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            prec1_m.update(prec1.item(), output.size(0))
            prec5_m.update(prec5.item(), output.size(0))

            if (last_batch or batch_idx % 100 == 0):
                log_name = 'MS-COCO Test'
                print(
                    '{0}: [{1:>4d}/{2}]  '
                    'Prec@1: {top1.val:>7.2f} ({top1.avg:>7.2f})  '
                    'Prec@5: {top5.val:>7.2f} ({top5.avg:>7.2f})'.format(
                        log_name, batch_idx, last_idx,
                        top1=prec1_m, top5=prec5_m))
    return prec1_m, prec5_m
