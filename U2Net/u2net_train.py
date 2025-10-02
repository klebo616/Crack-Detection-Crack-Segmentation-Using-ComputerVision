import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np
import glob
import multiprocessing

from data_loader import RescaleT, RandomCrop, ToTensorLab, SalObjDataset
from model import U2NET

# ------- 1. Define loss function --------

bce_loss = nn.BCELoss(reduction='mean')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print(f"l0: {loss0.item():.6f}, l1: {loss1.item():.6f}, l2: {loss2.item():.6f}, l3: {loss3.item():.6f}, l4: {loss4.item():.6f}, l5: {loss5.item():.6f}, l6: {loss6.item():.6f}")
    return loss0, loss

# ------- 2. Training function --------

def train():
    import matplotlib.pyplot as plt
    import csv

    epoch_num = 80
    batch_size_train = 4
    model_name = 'u2netv4'

    tra_image_dir = r"C:\Users\bob1k\Desktop\MASTER\Thesis\hed_project\U-2-Net\test_data_v4\images"
    tra_label_dir = r"C:\Users\bob1k\Desktop\MASTER\Thesis\hed_project\U-2-Net\test_data_v4\reconstructed_masks"
    image_ext = '.jpg'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
    os.makedirs(model_dir, exist_ok=True)

    log_path = os.path.join(model_dir, "training_loss_log.csv")

    tra_img_name_list = glob.glob(os.path.join(tra_image_dir, '*' + image_ext))

    # Updated label mapping logic
    tra_lbl_name_list = []
    valid_img_name_list = []

    for img_path in tra_img_name_list:
        base = os.path.splitext(os.path.basename(img_path))[0]  # e.g., '00002_jpg.rf.af4d2f...'
        label_name = base + "_reconstructed.png"
        label_path = os.path.join(tra_label_dir, label_name)
        if os.path.exists(label_path):
            valid_img_name_list.append(img_path)
            tra_lbl_name_list.append(label_path)


    tra_img_name_list = valid_img_name_list

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)
        ])
    )

    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=1
    )

    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.cuda()

    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000

    with open(log_path, 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(['iteration', 'train_loss', 'target_loss'])

    for epoch in range(epoch_num):
        net.train()
        for i, data in enumerate(salobj_dataloader):
            ite_num += 1
            ite_num4val += 1

            inputs, labels = data['image'].type(torch.FloatTensor), data['label'].type(torch.FloatTensor)
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            optimizer.zero_grad()
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_tar_loss += loss2.item()
            print(f"[epoch: {epoch+1}/{epoch_num}, batch: {(i+1)*batch_size_train}, ite: {ite_num}] train loss: {running_loss / ite_num4val:.6f}, tar: {running_tar_loss / ite_num4val:.6f}")

            with open(log_path, 'a', newline='') as logfile:
                writer = csv.writer(logfile)
                writer.writerow([ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val])

            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), os.path.join(model_dir, f"{model_name}_bce_itr_{ite_num}_train_{running_loss / ite_num4val:.6f}_tar_{running_tar_loss / ite_num4val:.6f}.pth"))
                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0
                net.train()
    # ------- 6. Plot training loss --------
    iterations, train_losses, target_losses = [], [], []

    with open(log_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            iterations.append(int(row['iteration']))
            train_losses.append(float(row['train_loss']))
            target_losses.append(float(row['target_loss']))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(iterations, train_losses, label='Train Loss')
    plt.plot(iterations, target_losses, label='Target Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "training_loss_plot.png"))
    plt.close()


# ------- Run --------
if __name__ == '__main__':
    multiprocessing.freeze_support()
    train()
