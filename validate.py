# Adopted from: https://github.com/allenai/elastic/blob/master/multilabel_classify.py
# special thanks to @hellbell

import argparse
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os

from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from src.models import create_model
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--model-name', default='tresnet_l')
parser.add_argument('--model-path', default='./TRresNet_L_448_86.6.pth', type=str)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')


def main():
    args = parser.parse_args()
    args.batch_size = args.batch_size

    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    args.do_bottleneck_head = False
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    # Data loading code
    normalize = transforms.Normalize(mean=[0, 0, 0],
                                     std=[1, 1, 1])

    instances_path = os.path.join(args.data, 'annotations/instances_val2014.json')
    data_path = os.path.join(args.data, 'val2014')
    val_dataset = CocoDetection(data_path,
                                instances_path,
                                transforms.Compose([
                                    transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))

    print("len(val_dataset)): ", len(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate_multi(val_loader, model, args)


def validate_multi(val_loader, model, args):
    print("starting actuall validation")
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    mAP_meter = AverageMeter()

    Sig = torch.nn.Sigmoid()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        pred = output.data.gt(args.thre).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time,
                prec=prec, rec=rec))
            print(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                    .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    print(
        '--------------------------------------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("mAP score:", mAP_score)

    return


if __name__ == '__main__':
    main()
