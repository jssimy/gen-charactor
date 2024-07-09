from data import PetDataset
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import pandas as pd
import sys
sys.path.append('/home/jss/convnext')
import convnext as cn

impl = str(5)   # 3;w.d.=0.05 4:wd=0.0001, 5; LR=0.005, gmm=0.2, step=30

anno_trn = '/home/jss/pet-breed-classification/data/myannotations_train.csv'
anno_val = '/home/jss/pet-breed-classification/data/myannotations_val.csv'
anno_test = '/home/jss/pet-breed-classification/data/myannotations_test.csv'
RD = '/home/jss/pet-breed-classification/results/'

if __name__ == '__main__':

    batch_size = 128
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    train_data = PetDataset(annotation_file=anno_trn)
    val_data = PetDataset(annotation_file=anno_val, split='val')

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = cn.ConvNeXt()
    checkp = torch.load('/home/jss/convnext/convnext_tiny_22k_1k_384.pth', map_location="cpu")
    model.load_state_dict(checkp["model"])

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, train_data.num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.head.parameters(), lr=0.005, weight_decay=0.0001)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    # train
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start = 0
    end = 0

    for epoch in range(num_epochs):

        start = time.time()
        model.train()

        running_loss = 0.0
        running_corrects = 0.0
        

        for e, train_batch in enumerate(train_loader):            
            images = train_batch['image'].to(device)
            labels = train_batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss  += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if e % 10 == 0:
                print('Phase: Train, Epoch: {}/{}, Batch: {}/{}, Loss: {}'.format(
                    epoch+1, num_epochs, e+1, len(train_data) // batch_size, loss.item()))

        exp_lr_scheduler.step()

        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.double() / len(train_data)

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0.0

        with torch.no_grad():
            for e, val_batch in enumerate(val_loader):
                images = val_batch['image'].to(device)
                labels = val_batch['label'].to(device)

                outputs = model(images)

                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

                if e % 5 == 0:
                    print('Phase: Val, Epoch: {}/{}, Batch: {}/{}, Loss: {}'.format(
                        epoch+1, num_epochs, e + 1, len(val_data) // batch_size, loss.item()
                    ))

            val_epoch_loss = val_running_loss / len(val_data)
            val_epoch_acc = val_running_corrects.double() / len(val_data)

            if val_epoch_acc >= best_acc:
                best_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('Epoch: {}/{}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}, best_acc: {}'.format(
            epoch+1, num_epochs, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, best_acc))
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.cpu().numpy())
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.cpu().numpy())
        print('types: {} {} {} {}'.format(type(epoch_loss), type(epoch_acc), type(val_epoch_loss), type(val_epoch_acc)))

        end = time.time()
        lapse = end - start
        print('lap time: {} seconds'.format(lapse))

    df = pd.DataFrame(history)
    df.to_csv(RD+impl+'_'+'history.csv', index=False)
    torch.save(best_model_wts, RD+impl+'_'+'best_model.pth')































