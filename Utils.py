import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt


def down_sample(im, factor=2):
    """
    Down sample the input tensor by a factor.
    """
    im_l = F.interpolate(im, scale_factor=1/factor, mode='bilinear',
                         recompute_scale_factor=False, align_corners=False)
    im_l = F.interpolate(im_l, scale_factor=factor, mode='bilinear',
                         recompute_scale_factor=False, align_corners=False)
    return im_l


def rgb2y(im):
    """
    Convert RGB image to Y channel.
    """
    r = im[..., 0, :, :]
    g = im[..., 1, :, :]
    b = im[..., 2, :, :]

    y = .299 * r + .587 * g + .114 * b

    return y


def psnr(img1, img2):
    """
    Calculate the PSNR between two images.
    """
    out = rgb2y(img1)
    lbl = rgb2y(img2)
    out = out.clip(0, 255).round()
    lbl = lbl.clip(0, 255).round()
    diff = out - lbl
    rmse = torch.sqrt(torch.mean(diff**2))
    psnr = 20*torch.log10(255/rmse)
    return psnr


def plot_tensor(im, title=None):
    plt.imshow(im[0].permute(1, 2, 0))
    plt.show()
