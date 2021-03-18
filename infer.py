import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.models import create_model
import argparse
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='ASL MS-COCO Inference on a single image')

parser.add_argument('--model_path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
parser.add_argument('--pic_path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model_name', type=str, default='tresnet_l')
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--dataset_type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=None)


def main():
    print('ASL Example Inference code on a single image')

    # parsing args
    args = parse_args(parser)

    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    # doing inference
    print('loading image and doing inference...')
    im = Image.open(args.pic_path)
    im_resize = im.resize((args.input_size, args.input_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda()
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    detected_classes = classes_list[np_output > args.th]
    print('done\n')

    # example loss calculation
    output = model(tensor_batch)
    loss_func1 = AsymmetricLoss()
    loss_func2 = AsymmetricLossOptimized()
    target = output.clone()
    target[output < 0] = 0  # mockup target
    target[output >= 0] = 1
    loss1 = loss_func1(output, target)
    loss2 = loss_func2(output, target)
    assert abs((loss1.item() - loss2.item())) < 1e-6

    # displaying image
    print('showing image on screen...')
    fig = plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    # plt.rcParams["axes.titlesize"] = 10
    plt.title("detected classes: {}".format(detected_classes))

    plt.show()
    print('done\n')


if __name__ == '__main__':
    main()
