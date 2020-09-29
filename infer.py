import torch
from src.helper_functions.helper_functions import validate, create_dataloader
from src.models import create_model
import argparse

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference')
parser.add_argument('--val_dir')
parser.add_argument('--model_path')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--num_classes', type=int, default=81)
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_workers', type=int, default=8)


def main():
    # parsing args
    args = parser.parse_args()

    # setup model
    print('creating model...')
    model = create_model(args).cuda()
    state = torch.load(args.model_path, map_location='cpu')['model']
    model.load_state_dict(state, strict=True)
    model.eval()
    print('done\n')

    # setup data loader
    print('creating data loader...')
    val_loader = create_dataloader(args)
    print('done\n')

    print('doing validation...')
    map_score, _ = validate(model, val_loader)
    print("Validation mAP score: {:.2f}".format(map_score.avg))


if __name__ == '__main__':
    main()
