import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import feature
from scipy.ndimage import binary_dilation
import torch
import torch.nn as nn
from torchvision import transforms

from models.dct import DCTDenoiser
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
    plt.imshow(image.cpu().numpy().squeeze() * 255)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('fusion_plane_original.eps', format='eps')

    # ノイズを加える
    image_noise = image + sigma * torch.randn_like(image) / 255
    plt.figure()
    plt.imshow(image_noise.cpu().numpy().squeeze() * 255)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('fusion_plane_noisy.eps', format='eps')

    # モデル
    checkpoint_path = './checkpoints/model_epoch_15.pth'
    model_dct = DCTDenoiser(patch_size, 3 * sigma / 255)
    model_dct.to(device)
    model_dct2net = DCT2Net(patch_size, 3 * sigma / 255)
    model_dct2net.to(device)
    state_dict = torch.load(checkpoint_path)
    model_dct2net.load_state_dict(state_dict)

    # デノイズ
    with torch.no_grad():
        image_denoise_dct = model_dct(image_noise)
        image_denoise_dct = image_denoise_dct.view(224, 224).cpu().numpy()
        image_denoise_dct = np.clip(image_denoise_dct, 0, 1) * 255
        plt.figure()
        plt.imshow(image_denoise_dct)
        plt.axis('off')
        plt.colorbar()
        plt.savefig('fusion_plane_denoise_dct.eps', format='eps')

        image_denoise_dct2net = model_dct2net(image_noise)
        image_denoise_dct2net = image_denoise_dct2net.view(224, 224).cpu().numpy()
        image_denoise_dct2net = np.clip(image_denoise_dct2net, 0, 1) * 255
        plt.figure()
        plt.imshow(image_denoise_dct2net)
        plt.axis('off')
        plt.colorbar()
        plt.savefig('fusion_plane_denoise_dct2net.eps', format='eps')

        mask = feature.canny(image_denoise_dct, sigma=5)
        mask = binary_dilation(mask, structure=np.ones((3, 3)))
        plt.figure()
        plt.imshow(mask)
        plt.axis('off')
        plt.colorbar()
        plt.savefig('fusion_plane_mask.eps', format='eps')

        image_denoise_dct_masked = image_denoise_dct * (1 - mask)
        image_denoise_dct2net_masked =image_denoise_dct2net * mask
        plt.figure()
        plt.imshow(image_denoise_dct_masked)
        plt.axis('off')
        plt.colorbar()
        plt.savefig('fusion_plane_denoise_dct_masked.eps', format='eps')
        plt.figure()
        plt.imshow(image_denoise_dct2net_masked)
        plt.axis('off')
        plt.colorbar()
        plt.savefig('fusion_plane_denoise_dct2net_masked.eps', format='eps')
        image_denoise_fusion = image_denoise_dct_masked + image_denoise_dct2net_masked

    
    # デノイズ結果
    plt.figure()
    plt.imshow(image_denoise_fusion)
    plt.axis('off')
    plt.colorbar()
    plt.savefig('fusion_plane_denoise_fusion.eps', format='eps')

    # PSNR
    print('PSNR (noise): ' + str(round(10*np.log10(255**2 / np.mean((image.cpu().numpy().squeeze() * 255 - image_noise.cpu().numpy().squeeze() * 255)**2)), 2)) + 'dB')
    print('PSNR (denoise): ' + str(round(10*np.log10(255**2 / np.mean((image.cpu().numpy().squeeze() * 255 - image_denoise_fusion)**2)), 2)) + 'dB')