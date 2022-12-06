import torch
import torch
import torchvision
import torch.utils.data as data
import os
from os.path import join
import argparse
import logging
from tqdm import tqdm
#user import
from utils.io import check_dir,Img_to_zero_center

from model.faceAlexnet import AgeClassify
from data_loader.DataLoader_Pretrain_Alexnet import CACD

#step1: define argument
parser = argparse.ArgumentParser(description='pretrain age classifier')
# Optimizer
parser.add_argument('--learning_rate', '--lr', type=float, help='learning rate', default=1e-4)
parser.add_argument('--batch_size', '--bs', type=int, help='batch size', default=512)
parser.add_argument('--max_epoches', type=int, help='Number of epoches to run', default=200)
parser.add_argument('--val_interval', type=int, help='Number of steps to validate', default=20000)
parser.add_argument('--save_interval', type=int, help='Number of batches to save model', default=20000)

# Model
# Data and IO
parser.add_argument('--cuda_device', type=str, help='which device to use', default='0')
parser.add_argument('--checkpoint', type=str, help='logs and checkpoints directory', default='./checkpoint/pretrain_alexnet')
parser.add_argument('--saved_model_folder', type=str,
                    help='the path of folder which stores the parameters file',
                    default='./checkpoint/pretrain_alexnet/saved_parameters/')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device


check_dir(args.checkpoint)
check_dir(args.saved_model_folder)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#step2: define logging output
logger = logging.getLogger("Age classifer")
file_handler = logging.FileHandler(join(args.checkpoint, 'log.txt'), "w")
stdout_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.setLevel(logging.INFO)


def main():
    logger.info("Start to train:\n arguments: %s" % str(args))
    #step3: define transform
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((227, 227)),
        torchvision.transforms.ToTensor(),
        Img_to_zero_center()
    ])
    #step4: define train/test dataloader
    train_dataset = CACD("train", transforms, None)
    test_dataset = CACD("test", transforms, None)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    #step5: define model,optim
    model=AgeClassify()
    optim=model.optim

    for epoch in range(args.max_epoches):
        for train_idx, (img,label) in enumerate(train_loader):
            img=img.to(DEVICE)
            label=label.to(DEVICE)

            #train
            optim.zero_grad()
            model.train(img,label)
            loss=model.loss
            loss.backward()
            optim.step()
            format_str = ('step %d/%d, cls_loss = %.3f')
            logger.info(format_str % (train_idx, len(train_loader), loss))


            # save the parameters at the end of each save interval
            if train_idx*args.batch_size % args.save_interval == 0:
                model.save_model(dir=args.saved_model_folder,
                                 filename='epoch_%d_iter_%d.pth'%(epoch, train_idx))
                logger.info('checkpoint has been created!')

            #val step

            if train_idx % args.val_interval == 0:
                train_correct=0
                train_total=0
                with torch.no_grad():
                    for val_img,val_label in tqdm(test_loader):
                        val_img=val_img.to(DEVICE)
                        val_label=val_label.to(DEVICE)
                        output=model.val(val_img)
                        train_correct += (output == val_label).sum()
                        train_total += val_img.size()[0]

                logger.info('validate has been finished!')
                format_str = ('val_acc = %.3f')
                logger.info(format_str % (train_correct.cpu().numpy()/train_total))

if __name__ == '__main__':
#    print("25_Chris_O'Dowd_0012.jpg" in os.listdir('/home/ubuntu/AleksandrSim-COMP_SCI_496-0_final_project/cacd2000-lists'))
    main()