import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms

from models.dct2net import DCT2Net


image_path = './dataset/test/3096.jpg'
sigma = 25
patch_size = 13

class ScaleTo255:

    def __call__(self, image):

        min_val = image.min()
        max_val = image.max()
        image_scaled = 255 * (image - min_val) / (max_val - min_val)

        return image_scaled

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                                transforms.Resize((224, 224)), 
                                transforms.ToTensor(), 
                                ScaleTo255()])

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 画像の読み込み
    image = Image.open(image_path)
    image = transform(image).view(1, 1, 224, 224) / 255
    plt.figure()
    plt.imshow(image.cpu().numpy().squeeze())
    plt.axis('off')
    plt.colorbar()
    plt.savefig('dct2net_plane_original.eps', format='eps')

    # ノイズを加える
    image_noise = image + sigma * torch.randn_like(image) / 255
    plt.figure()
    plt.imshow(image_noise.cpu().numpy().squeeze())
    plt.axis('off')
    plt.colorbar()
    plt.savefig('dct2net_plane_noisy.eps', format='eps')

    # モデル
    checkpoint_path = './checkpoints/model_epoch_15.pth'
    model = DCT2Net(patch_size, 3 * sigma / 255)
    model.to(device)
    plt.figure()
    plt.imshow(model.conv1.weight.view(patch_size**2, patch_size**2).detach().numpy())
    plt.axis('off')
    plt.colorbar()
    plt.savefig('dct2net_weight_before.eps', format='eps')
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    plt.figure()
    plt.imshow(model.conv1.weight.view(patch_size**2, patch_size**2).detach().numpy())
    plt.axis('off')
    plt.colorbar()
    plt.savefig('dct2net_weight_after.eps', format='eps')
    

    # デノイズ
    with torch.no_grad():
        image_denoise = model(image_noise)
        image_denoise = image_denoise.view(224, 224).cpu().numpy()
        image_denoise = np.clip(image_denoise, 0, 1) * 255
    
    # デノイズ結果
    plt.figure()
    plt.imshow(image_denoise)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('dct2net_plane_denoise_before.eps', format='eps')

    # PSNR
    print('PSNR (noise): ' + str(round(10*np.log10(255**2 / np.mean((image.cpu().numpy().squeeze() * 255 - image_noise.cpu().numpy().squeeze() * 255)**2)), 2)) + 'dB')
    print('PSNR (denoise): ' + str(round(10*np.log10(255**2 / np.mean((image.cpu().numpy().squeeze() * 255 - image_denoise)**2)), 2)) + 'dB')