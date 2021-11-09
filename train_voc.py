import os
import argparse
import numpy as np
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import TensorDatasetVOC, mAP, CocoDetection, CutoutPIL, ModelEma, add_weight_decay, \
    TensorDataset
from src.models import create_model
from src.loss_functions.losses import AsymmetricLossOptimized
from src.helper_functions.voc import Voc2007Classification
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('data', metavar='DIR', help='path to dataset', default='../VOC2007')
parser.add_argument('dataset', type=str, choices=['ms-coco', 'nus-wide', 'pascal-voc'], help="Choose between ms-coco (that's it for now).")

# For noise injection
parser.add_argument('--gold_fraction', '-gf', type=float, default=1,
                    help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif',
                    help='Type of corruption ("unif" or "flip").')
# random seed
parser.add_argument('--seed', type=int, default=1)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--model-name', default='tresnet_xl')
parser.add_argument('--model-path', default='./tresnet_l.pth', type=str)
parser.add_argument('--num-classes', default=20)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--thre', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')


def plot_corruption_matrix(C, title, run_dir):
    fig, ax = plt.subplots()

    # ax.imshow(C, cmap='Blues', vmin=0, vmax=0.6)
    ax.imshow(C, cmap='Blues', vmin=0, vmax=1)
    # ax.imshow(C, cmap='Blues')

    # for i in range(len(C)):
    #     for j in range(len(C)):
    #         ax.text(i, j, str(C[i][j]), va='center', ha='center')

    # labels = ['label_1', 'label_2', 'label_3', 'label_4', 'label_5']
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)

    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    ax.set_title(title, y=-0.1)

    plt.savefig(os.path.join(run_dir, title + '.png'))

    # plt.show()


def main():
    # np.set_printoptions(threshold=np.inf)
    args = parser.parse_args()
    run_dir = os.path.join('runs_latest/train_voc/{}/15% gold/phase1/{}/'.format(args.corruption_type, args.corruption_prob))
    #run_dir = os.path.join('runs_latest/train_sgalc/') #{}/10% gold/phase2/{}/'.format(args.corruption_type, args.corruption_prob))
    args.do_bottleneck_head = False

    #############
    # FOR LOADING OUR SAVED SILVER MODELS (when we do train_phase2)

    # state = torch.load(args.model_path, map_location='cpu')
    # model = create_model(args).cuda()
    # model.load_state_dict(state, strict=True)

    #############

    #############
    # FOR LOADING pretrained ImageNet model from MODEL_ZOO.md (when we do train_phase1)

    print('creating model...')
    model = create_model(args).cuda()
    tresnet_path = './PASCAL_VOC_TResNet_xl_448_96.0.pth'
    if tresnet_path:  # make sure to load pretrained ImageNet model
        state = torch.load(tresnet_path, map_location='cpu')
        filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
    print('done\n')

    ###############
    if args.dataset == 'ms-coco':

        # COCO Data loading
        instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
        data_path_val = f'{args.data}/val2014'  # args.data
        data_path_train = f'{args.data}/train2014'  # args.data

        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])

        train_data_gold = CocoDetection(root=data_path_train,
                                        annFile=instances_path_train,
                                        transform=train_transform,
                                        train=True,
                                        gold=True,
                                        gold_fraction=args.gold_fraction,
                                        corruption_prob=args.corruption_prob,
                                        corruption_type=args.corruption_type,
                                        seed=args.seed)

        train_data_silver = CocoDetection(root=data_path_train,
                                        annFile=instances_path_train,
                                        transform=train_transform,
                                        train=True,
                                        gold=False,
                                        gold_fraction=args.gold_fraction,
                                        corruption_prob=args.corruption_prob,
                                        corruption_type=args.corruption_type,
                                        shuffle_indices=train_data_gold.shuffle_indices,
                                        seed=args.seed)

        with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
            f.write('\n############################\n')
            f.write("TRUE Corruption Matrix:\n\n{}\n".format(train_data_silver.C))
        plot_corruption_matrix(train_data_silver.C, 'TRUE Corruption Matrix', run_dir)
        plot_corruption_matrix(train_data_silver.C_true, '(VERY) TRUE Corruption Matrix', run_dir)

        train_data_gold_deterministic = CocoDetection(root=data_path_train,
                                                    annFile=instances_path_train,
                                                    transform=test_transform,
                                                    train=True,
                                                    gold=True,
                                                    gold_fraction=args.gold_fraction,
                                                    corruption_prob=args.corruption_prob,
                                                    corruption_type=args.corruption_type,
                                                    shuffle_indices=train_data_gold.shuffle_indices,
                                                    seed=args.seed)

        val_dataset = CocoDetection(root=data_path_val,
                                    annFile=instances_path_val,
                                    transform=test_transform,
                                    train=False)


        print("len(train_dataset_gold)): ", len(train_data_gold))
        print("len(train_dataset_silver)): ", len(train_data_silver))
        print("len(val_dataset)): ", len(val_dataset))
        print("Creating loaders...")

        # Pytorch Data loader
        train_silver_loader = torch.utils.data.DataLoader(
            train_data_silver, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        train_gold_deterministic_loader = torch.utils.data.DataLoader(
            train_data_gold_deterministic, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        train_all_merged = TensorDataset(data_path_train,
                                        np.concatenate((train_data_gold.train_data, train_data_silver.train_data)),
                                        np.concatenate((train_data_gold.train_labels, train_data_silver.train_labels)),
                                        train_transform)
        train_all_loader = torch.utils.data.DataLoader(train_all_merged, batch_size=args.batch_size, shuffle=True,
                                                    num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)


        print("Loading done!")
        
        # Training phase 1
        train_phase1(model, train_silver_loader, val_loader, args.lr, run_dir)

        # Estimating corruption matrix C_hat
        C_hat_transpose = get_C_hat_transpose(model, single_representatives_loader, train_gold_deterministic_loader,
                                            train_data_gold, run_dir)
        #C_hat_transpose = train_data_silver.C
        C_hat_transpose = C_hat_transpose.cuda()

        print('Creating model for Training Phase 2...')
        model_phase2 = create_model(args).cuda()
        tresnet_path = './MS_COCO_TRresNet_M_224_81.8.pth'
        if tresnet_path:  # make sure to load pretrained ImageNet model
            state = torch.load(tresnet_path, map_location='cpu')
            filtered_dict = {k: v for k, v in state['model'].items() if
                            (k in model_phase2.state_dict() and 'head.fc' not in k)}
            model_phase2.load_state_dict(filtered_dict, strict=False)
        print('done\n')

        train_phase2(model_phase2, train_all_loader, val_loader, args.lr, run_dir, C_hat_transpose)

        print("DONE!")
        exit()
    
    elif args.dataset == 'pascal-voc':

        # normalize = transforms.Normalize(mean=[0, 0, 0],
        #                              std=[1, 1, 1])
        # transform = transforms.Compose([
        #                         transforms.Resize((args.image_size, args.image_size)),
        #                         transforms.ToTensor(),
        #                         normalize,
        #                     ])
        


        # train_dataset = Voc2007Classification('test')
        
        # COCO Data loading
        # instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
        # instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
        # data_path_val = f'{args.data}/val2014'  # args.data
        # data_path_train = f'{args.data}/train2014'  # args.data

        # train_transform = transforms.Compose([
        #     transforms.Resize((args.image_size, args.image_size)),
        #     CutoutPIL(cutout_factor=0.5),
        #     RandAugment(),
        #     transforms.ToTensor(),
        # ])
        # test_transform = transforms.Compose([
        #     transforms.Resize((args.image_size, args.image_size)),
        #     transforms.ToTensor(),
        # ])
        # transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.ToTensor(),
                            ])
        train_data_gold = Voc2007Classification('trainval',
                                        transform=transform,
                                        gold=True,
                                        gold_fraction=args.gold_fraction,
                                        corruption_prob=args.corruption_prob,
                                        corruption_type=args.corruption_type,
                                        train=True)

        train_data_silver = Voc2007Classification('trainval',
                                        transform=transform,
                                        gold=False,
                                        gold_fraction=args.gold_fraction,
                                        corruption_prob=args.corruption_prob,
                                        corruption_type=args.corruption_type,
                                        shuffle_indices=train_data_gold.shuffle_indices,
                                        train=True)

        
        
        # train_data_gold = CocoDetection(root=data_path_train,
        #                                 annFile=instances_path_train,
        #                                 transform=train_transform,
        #                                 train=True,
        #                                 gold=True,
        #                                 gold_fraction=args.gold_fraction,
        #                                 corruption_prob=args.corruption_prob,
        #                                 corruption_type=args.corruption_type,
        #                                 seed=args.seed)

        # train_data_silver = CocoDetection(root=data_path_train,
        #                                 annFile=instances_path_train,
        #                                 transform=train_transform,
        #                                 train=True,
        #                                 gold=False,
        #                                 gold_fraction=args.gold_fraction,
        #                                 corruption_prob=args.corruption_prob,
        #                                 corruption_type=args.corruption_type,
        #                                 shuffle_indices=train_data_gold.shuffle_indices,
        #                                 seed=args.seed)


        train_data_gold_deterministic = Voc2007Classification('trainval',
                                                    transform=transform,
                                                    gold=True,
                                                    gold_fraction=args.gold_fraction,
                                                    corruption_prob=args.corruption_prob,
                                                    corruption_type=args.corruption_type,
                                                    shuffle_indices=train_data_gold.shuffle_indices,
                                                    train=True)

        # val_dataset = CocoDetection(root=data_path_val,
        #                             annFile=instances_path_val,
        #                             transform=test_transform,
        #                             train=False)

        val_dataset = Voc2007Classification('test', transform=transform, train=False)
        

        print("len(train_dataset_gold)): ", len(train_data_gold))
        print("len(train_dataset_silver)): ", len(train_data_silver))
        print("len(val_dataset)): ", len(val_dataset))
        print("Creating loaders...")

        # Pytorch Data loader
        # train_silver_loader = torch.utils.data.DataLoader(
        #     train_data_silver, batch_size=args.batch_size, shuffle=True,
        #     num_workers=args.workers, pin_memory=True)

        train_gold_deterministic_loader = torch.utils.data.DataLoader(
            train_data_gold_deterministic, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        # train_all_merged = TensorDatasetVOC(np.concatenate((train_data_gold.train_data, train_data_silver.train_data)),
        #                                 np.concatenate((train_data_gold.train_labels, train_data_silver.train_labels)),transform=transform)
        # train_all_loader = torch.utils.data.DataLoader(train_all_merged, batch_size=args.batch_size, shuffle=True,
        #                                             num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

        print("Loading done!")

        
        # C_hat_transpose = train_data_silver.C
        
        #C_hat = torch.zeros((args.num_classes, args.num_classes))

        # C_hat_transpose = torch.from_numpy(C_hat_transpose).to('cuda:0')
        # plot_corruption_matrix(C_hat_transpose.cpu().numpy(), "CM-true-training", run_dir)

        # Training phase 1
        train_phase1(model, train_gold_deterministic_loader, val_loader, args.lr, run_dir)

        # Estimating corruption matrix C_hat
        # C_hat_transpose = get_C_hat_transpose(model, single_representatives_loader, train_gold_deterministic_loader,
        #                                      train_data_gold, run_dir)
        # C_hat_transpose = C_hat_transpose.cuda()
        # # C_hat_transpose = train_data_silver.C
        
        # #C_hat = torch.zeros((args.num_classes, args.num_classes))

        # # C_hat_transpose = torch.from_numpy(C_hat_transpose).to('cuda:0')
        # #C_hat_transpose = C_hat_transpose.cuda()
        # del model
        # print('Model phase 1 terminated ...')
        # print('Creating model for Training Phase 2...')
        # model_phase2 = create_model(args).cuda()
        # tresnet_path = './PASCAL_VOC_TResNet_xl_448_96.0.pth'
        # if tresnet_path:  # make sure to load pretrained ImageNet model
        #     state = torch.load(tresnet_path, map_location='cpu')
        #     filtered_dict = {k: v for k, v in state['model'].items() if
        #                     (k in model_phase2.state_dict() and 'head.fc' not in k)}
        #     model_phase2.load_state_dict(filtered_dict, strict=False)
        # print('done\n')

        # train_phase2(model_phase2, train_all_loader, val_loader, args.lr, run_dir, C_hat_transpose)

        # print("DONE!")
        #exit()



def train_phase2(model, train_loader, val_loader, lr, run_dir, C_hat_transpose):
    print("Start Training Phase 2...\n\n")
    with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
        f.write("Start Training Phase 2...\n\n")

    start = time.process_time()

    Epochs = 80
    Stop_epoch = 80
    weight_decay = 2e-4 #2e-4
    criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    Sig = torch.nn.Sigmoid()
    num_classes = len(C_hat_transpose)
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (inputData, target) in enumerate(train_loader):
            
            inputData, target = inputData.numpy(), target.numpy()
            # inputData, target = inputData.numpy(), target

            # we subtract num_classes because we added num_classes to allow us to identify gold examples
            gold_indices = np.all(target > (num_classes - 1), axis=1)
            gold_len = np.sum(gold_indices)
            if gold_len > 0:
                data_g, target_g = inputData[gold_indices], target[gold_indices] - num_classes

            silver_indices = np.all(target < num_classes, axis=1)
            silver_len = np.sum(silver_indices)
            if silver_len > 0:
                data_s, target_s = inputData[silver_indices], target[silver_indices]

            loss_s = 0
            if silver_len > 0:
                with autocast():
                    output_s = model(torch.from_numpy(data_s).cuda()).float()

                # Inefficient way; MUST be improved
                pre1 = torch.empty(0, num_classes).cuda()
                for j in range(silver_len):
                    pos_labels = np.nonzero(target_s[j])[0]
                    # THIS COULD BE A NICE EXPERIMENT! Maybe taking the max is better than the mean
                    # max_col_C_hat_transpose = torch.max(C_hat_transpose[pos_labels], 0)[0]
                    mean_col_C_hat_transpose = torch.mean(C_hat_transpose[pos_labels], 0)
                    # max_col_C_hat_transpose = torch.max(C_hat_transpose[pos_labels],0)[0]
                    pre1 = torch.vstack((pre1, mean_col_C_hat_transpose))
                    # pre1 = torch.vstack((pre1, max_col_C_hat_transpose))
                corrected_output = torch.mul(Sig(output_s), pre1)
                # corrected_output = torch.mul(output_s, pre1)
                loss_s = criterion(corrected_output, torch.from_numpy(target_s).cuda())
            loss_g = 0
            if gold_len > 0:
                with autocast():
                    output_g = model(torch.from_numpy(data_g).cuda()).float()
                loss_g = criterion(output_g, torch.from_numpy(target_g).cuda())

            # We don't divide by batch_size as in the GLC paper because ASL treats scaling later, with the GradScaler
            loss = (loss_g + loss_s)
            model.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            if i % 200 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                tmp_log = 'Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'.format(epoch, Epochs, str(i).zfill(3),
                                                                                        str(steps_per_epoch).zfill(3),
                                                                                        scheduler.get_last_lr()[0],
                                                                                        loss.item())
                print(tmp_log)
                with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
                    f.write(tmp_log + '\n')

        model.eval()
        mAP_score = validate_multi(val_loader, model)
        dict_res = validate_multi_metrics(val_loader, model)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score

        tmp_log = 'current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP)
        print(tmp_log)
        with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
            f.write('epoch {}: {}'.format(epoch, tmp_log))
            f.write(str(dict_res))
            f.write('\n#############################################\n')

        torch.save(model.state_dict(), os.path.join(run_dir, 'model_checkpoint.pth'))
        

    # Save last model
    # torch.save(model.state_dict(), os.path.join(run_dir, 'model-last-phase2.pth'))
    

    end = time.process_time()
    with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
        f.write('total runtime: {}'.format(end - start))
        f.write('\n#######################\n'
                'Training phase2 done!\n\n')
    print('\n#######################\n'
          'Training phase2 done!\n\n')


def train_phase1(model, train_loader, val_loader, lr, run_dir):
    print("Start Training Phase 1...\n\n")
    # with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
    #     f.write("Start Training Phase 1...\n\n")
    start = time.process_time()

    Epochs = 80
    Stop_epoch = 80
    weight_decay = 2e-4 #2e-4
    criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)

    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    Sig = torch.nn.Sigmoid()
    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break

        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()
            with autocast():
                output = model(inputData).float()

            loss = criterion(output, target)
            model.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            if i % 200 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                tmp_log = 'Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'.format(epoch, Epochs, str(i).zfill(3),
                                                                                        str(steps_per_epoch).zfill(3),
                                                                                        scheduler.get_last_lr()[0],
                                                                                        loss.item())
                print(tmp_log)
                # with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
                #     f.write(tmp_log + '\n')

        model.eval()
        mAP_score = validate_multi(val_loader, model)
        # dict_res = validate_multi_metrics(val_loader, model)
        model.train()

        if mAP_score > highest_mAP:
            highest_mAP = mAP_score

        # tmp_log = 'current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP)
        # print(tmp_log)
        # with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
        #     f.write('epoch {}: {}'.format(epoch, tmp_log))
        #     f.write(str(dict_res))
        #     f.write('\n#############################################\n')

        model_state = {
              "state_dict": model.state_dict(),
              "optimizer": optimizer.state_dict(),
          }
        torch.save(model_state, "tresnetm-asl-voc-epoch{0}".format(epoch))

    # Save last model
    torch.save(model.state_dict(), os.path.join(run_dir, 'model-silver.pth'))
    

    end = time.process_time()
    with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
        f.write('total runtime: {}'.format(end - start))
        f.write('\n#######################\n'
                'Training phase1 done!\n\n')
    print('\n#######################\n'
          'Training phase1 done!\n\n')

def get_C_hat_transpose(model, single_representatives_loader, train_gold_deterministic_loader,
                        train_data_gold, run_dir):
    print('Estimating corruption matrix...\n')
    with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
        f.write('Estimating corruption matrix...\n')
    Sig = torch.nn.Sigmoid()
    Softmax = torch.nn.Softmax(1)
    preds = []
    single_repr_preds = []
    targets = []
    single_repr_targets = []

    model.eval()
    num_classes = train_data_gold.num_classes

    regulators = np.zeros((num_classes, num_classes))
    counts = np.zeros(num_classes)

    print("Setting up representative predictions...")
    for i, (inputData, target) in enumerate(single_representatives_loader):

        with torch.no_grad():
            output = Softmax(model(inputData.cuda())).cpu()

        output_np = output.cpu().numpy()
        target_np = target.cpu().numpy()

        for j in range(len(target_np)):
            target_labels_pos = target_np[j].nonzero()[0]
            for label in target_labels_pos:
                regulators[label] += output_np[j]
                counts[label] += 1

        # single_repr_preds.extend(output.cpu().numpy())
        # single_repr_targets.extend(target.cpu().numpy())

    for x in (counts == 0).nonzero()[0]:
        default = np.zeros(num_classes)
        default[x] = 1
        #default = Sig(torch.from_numpy(default)).numpy()
        default = torch.from_numpy(default).numpy() # Just without the sigmoid
        counts[x] = 1
        regulators[x] = default

    counts = counts.reshape((num_classes, 1))
    regulators /= counts

    print("done")
    print("Continue to estimate C_hat...")

    for i, (inputData, target) in enumerate(train_gold_deterministic_loader):
        # we subtract 80 because we added 80 to gold so we could identify which example is gold in train_phase2
        target = (target - num_classes)

        with torch.no_grad():
            output = Sig(model(inputData.cuda())).cpu()

        preds.extend(output.cpu().numpy())
        targets.extend(target.cpu().numpy())

    preds = np.array(preds)

    C_hat = torch.zeros((num_classes, num_classes))
    for label in range(num_classes):
        label_np = np.zeros(num_classes)
        label_np[label] = 1

        tmp = (train_data_gold.train_labels - num_classes) * label_np
        indices = np.arange(len(train_data_gold.train_labels))[np.isclose(tmp.sum(axis=1), 1)]

        sum_preds = np.zeros(num_classes)
        for ind in indices:
            target_tmp = train_data_gold.train_labels[ind] - num_classes - label_np
            other_labels = np.nonzero(target_tmp)[0]
            to_subtract = np.zeros(num_classes)
            to_rebalance = np.zeros(num_classes)

            for ol in other_labels:
                to_subtract += regulators[ol]
            sum_preds += preds[ind] - to_subtract + regulators[label] * len(other_labels)

        row = sum_preds / len(indices)
        C_hat[label] = torch.from_numpy(row)

    # Ignore these plots, they were just helpful long time ago
    Sig_C = Sig(C_hat)
    Soft_C = Softmax(C_hat)
    plot_corruption_matrix(Sig_C.numpy(), "Sig Corruption Matrix", run_dir)
    plot_corruption_matrix(Soft_C.numpy(), "Soft Corruption Matrix", run_dir)
    plot_corruption_matrix(Sig(Soft_C).numpy(), "Sig(Soft) Corruption Matrix", run_dir)
    plot_corruption_matrix(Softmax(Sig_C).numpy(), "Soft(Sig) Corruption Matrix", run_dir)

    C_hat = Sig(C_hat)
    # for i in range(len(C_hat)):
    #     daig = C_hat[i][i]
    #     for j in range(len(C_hat)):
    #         C_hat[i][j] = C_hat[i][j]/daig

    with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
        f.write('\n############################\n')
        f.write("ESTIMATED Corruption Matrix:\n\n{}\n".format(C_hat))
    plot_corruption_matrix(C_hat.numpy(), "ESTIMATED Corruption Matrix", run_dir)

    with open(os.path.join(run_dir, 'log_run.txt'), "a+") as f:
        f.write('C_hat estimation done!\n\n')
        f.write('\n#######################\n')
    print('C_hat estimation done!\n\n')
    print('\n#######################\n')

    return C_hat.T


def validate_multi(val_loader, model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []
    for i, (input_data, target) in enumerate(val_loader):
        target = target
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input_data.cuda())).cpu()

        preds_regular.append(output_regular.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())

    
    return mAP_score_regular

def validate_multi_metrics(val_loader, model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []
    for i, (input_data, target) in enumerate(val_loader):
        target = target
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input_data.cuda())).cpu()

        preds_regular.append(output_regular.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    dict_results = compute_metrics(torch.cat(preds_regular).numpy(), torch.cat(targets).numpy(), 0.5)
    
    return dict_results

if __name__ == '__main__':
    main()
