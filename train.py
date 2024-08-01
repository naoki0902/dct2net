import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.dct2net import DCT2Net


class ScaleTo255:

    def __call__(self, image):

        min_val = image.min()
        max_val = image.max()
        image_scaled = 255 * (image - min_val) / (max_val - min_val)

        return image_scaled
    

class CustomDataset(Dataset):

    def __init__(self, root_dir, transform):

        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        image = self.transform(image)

        return image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 前処理
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                                transforms.Resize((224, 224)), 
                                transforms.ToTensor(), 
                                ScaleTo255()])

# データセット
train_dataset_dir_path = './dataset/train'
test_dataset_dir_path = './dataset/test'
train_dataset = CustomDataset(root_dir=train_dataset_dir_path, transform=transform)
test_dataset = CustomDataset(root_dir=test_dataset_dir_path, transform=transform)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# モデル
patch_size = 13
sigma = 25
model = DCT2Net(patch_size, 3 * sigma / 255)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

# 学習
save_dir = './checkpoints'
num_epochs = 15
train_loss_curve = []
test_loss_curve = []
for epoch in range(num_epochs):

    # train
    model.train()
    train_loss = 0.0
    for _, imgs in enumerate(train_dataloader):
        imgs_noisy = imgs + sigma * torch.randn_like(imgs)
        imgs /= 255
        imgs_noisy /= 255
        optimizer.zero_grad()
        imgs_denoised = model(imgs_noisy)
        loss = criterion(imgs_denoised, imgs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('=', end='')
    print()
    train_loss /= len(train_dataloader)
    scheduler.step()
    
    # test
    model.eval()
    test_loss = 0.0
    for img in test_dataloader:
        img_noisy = img + sigma * torch.randn_like(img)
        img /= 255
        img_noisy /= 255
        img_denoised = model(img_noisy)
        loss = criterion(img_denoised, img)
        test_loss += loss.item()
    test_loss /= len(test_dataloader)

    # 途中経過の画像を保存
    plt.figure()
    plt.imshow(img[0, :, :, :].squeeze().detach().numpy()*255)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('img_epoch_' + str(epoch+1))

    plt.figure()
    plt.imshow(img_noisy[0, :, :, :].squeeze().detach().numpy()*255)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('img_noisy_epoch_' + str(epoch+1))

    plt.figure()
    plt.imshow(img_denoised[0, :, :, :].squeeze().detach().numpy()*255)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('img_denoised_epoch_' + str(epoch+1))

    # summary
    train_loss_curve.append(train_loss)
    test_loss_curve.append(test_loss)
    print('Epoch: ' + str(epoch+1) + ', Train Loss: ' + str(train_loss))
    print('Epoch: ' + str(epoch+1) + ', Test Loss: ' + str(test_loss))
    model_save_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), model_save_path)

# loss curveを図示
plt.figure()
plt.plot(train_loss_curve, label='Train')
plt.plot(test_loss_curve, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()