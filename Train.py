import glob

from Utils import *
import torch
import torchvision
from FEQEModel import FeqeModel
from DataSet import Div2k
from torch.utils.tensorboard import SummaryWriter
from VGGLoss import VGGPerceptualLoss
import time
from piqa import SSIM
from pthflops import count_ops
from torchvision.io import read_image
import torchvision.transforms.functional as F

RUN_NAME = "C32_N5"
LOW_RES_SCALE = 2
# --- Hyper Parameters ---
BATCH_SIZE = 8
TRAIN_IM_SIZE = (196, 196)
NUM_WORKERS = 0
RESIDUAL_BLOCKS = 20
CHANNELS = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
CHECKPOINT_EVERY = 5
VGG_LOSS_WEIGHT = 1e-3
ssim = SSIM().cuda()


def get_device():
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on device:", device)
    return device


def train(data_loader_train, data_loader_test, device, model, optimizer, vgg_loss_fn, scheduler, writer):

    for epoch in range(EPOCHS + 1):
        start = time.time()
        for i, im_batch in enumerate(data_loader_train):
            im_batch = im_batch.to(device)
            im_batch_lowres = down_sample(im_batch, LOW_RES_SCALE)

            im_batch_lowres_recon = model(im_batch_lowres)
            mse_loss = torch.nn.MSELoss()(im_batch_lowres_recon, im_batch)
            vgg_loss = vgg_loss_fn(im_batch_lowres_recon, im_batch)
            loss = (1 - VGG_LOSS_WEIGHT) * mse_loss + VGG_LOSS_WEIGHT * vgg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter = epoch * len(data_loader_train) + i
            writer.add_scalar('Loss/loss', loss.item(), iter)
            writer.add_scalar('Loss/mse_loss', mse_loss.item(), iter)
            writer.add_scalar('Loss/vgg_loss', vgg_loss.item(), iter)
            writer.add_scalar('Metrics/PSNR_Train', psnr(im_batch * 255, im_batch_lowres_recon * 255), iter)
            writer.add_scalar('Metrics/SSIM_Train', ssim(im_batch, im_batch_lowres_recon), iter)

        # evaluate on test set
        with torch.no_grad():
            test_avg_psnr = 0
            test_avg_ssim = 0
            for i, im_batch in enumerate(data_loader_test):
                im_batch = im_batch.to(device)
                im_batch_lowres = down_sample(im_batch, LOW_RES_SCALE)

                im_batch_lowres_recon = model(im_batch_lowres)

                test_avg_psnr += psnr(im_batch * 255, im_batch_lowres_recon * 255)
                test_avg_ssim += ssim(im_batch, im_batch_lowres_recon)
            test_avg_psnr /= len(data_loader_test)
            test_avg_ssim /= len(data_loader_test)
            writer.add_scalar('Metrics/PSNR_Test', test_avg_psnr, epoch * len(data_loader_test) + i)
            writer.add_scalar('Metrics/SSIM_Test', test_avg_ssim, epoch * len(data_loader_test) + i)

        if epoch % CHECKPOINT_EVERY == 0:
            torch.save(model, 'checkpoints/' + RUN_NAME + '_' + str(epoch) + '.pth')

        scheduler.step()

        print("Epoch: {}/{}".format(epoch, EPOCHS), "Time: {:.2f}".format(time.time() - start))
        print("Test Avg. PSNR: {:.2f}".format(test_avg_psnr))
        print("Test Avg. SSIM: {:.2f}".format(test_avg_ssim))

    return model


def banchmark_test(data_loader_test, device, model, writer, name):

    # evaluate on test set
    with torch.no_grad():
        test_avg_psnr = 0
        test_avg_ssim = 0
        for i, im_batch in enumerate(data_loader_test):
            im_batch = im_batch.to(device)
            im_batch_lowres = down_sample(im_batch, LOW_RES_SCALE)

            im_batch_lowres_recon = model(im_batch_lowres)

            test_avg_psnr += psnr(im_batch * 255, im_batch_lowres_recon * 255)
            test_avg_ssim += ssim(im_batch, im_batch_lowres_recon)
        test_avg_psnr /= len(data_loader_test)
        test_avg_ssim /= len(data_loader_test)
        print("Test Avg. PSNR: {:.2f}".format(test_avg_psnr))
        print("Test Avg. SSIM: {:.2f}".format(test_avg_ssim))
        writer.add_scalar(f'{name}/Metrics/PSNR_Test', test_avg_psnr)
        writer.add_scalar(f'{name}/Metrics/SSIM_Test', test_avg_ssim)


def reconstruct_image(model, device, transform):

    image_name = '0840'
    image_path = f'Data/Test/{image_name}.png'
    image = read_image(image_path)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].title.set_text('Original Image')
    ax[0, 0].imshow(F.to_pil_image(image))
    image = transform(image)
    image = (image.unsqueeze(0) / 255).to(device)
    im_l = down_sample(image, LOW_RES_SCALE)
    im_l_recon = model(im_l)
    ax[0, 1].title.set_text('Ground Truth')
    ax[0, 1].imshow(F.to_pil_image(image.squeeze(0)))
    ax[1, 0].title.set_text('Low Resolution')
    ax[1, 0].imshow(F.to_pil_image(im_l.squeeze(0)))
    ax[1, 1].title.set_text('Reconstructed')
    ax[1, 1].imshow(F.to_pil_image(im_l_recon.squeeze(0)))
    fig.show()
    fig.savefig('Results/' + image_name + '_reconstructed.png')


def main():
    device = get_device()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(TRAIN_IM_SIZE),
    ])

    # writer = SummaryWriter(comment=RUN_NAME)

    """Train Model"""
    # dataset_train = Div2k('Data/Train/', transform=transform)
    # data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE,
    #                                                 shuffle=True, num_workers=0)
    #
    # dataset_test = Div2k('Data/Test/')
    # data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
    #                                                shuffle=True, drop_last=True, num_workers=0)
    #
    # model = FeqeModel(num_residual_blocks=RESIDUAL_BLOCKS, channels=CHANNELS).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
    #                              betas=(0.9, 0.999), eps=1e-08)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2 * EPOCHS // 3, gamma=0.1)
    #
    # vgg_loss_fn = VGGPerceptualLoss(resize=False).to(device)

    # train(data_loader_train, data_loader_test, device, model, optimizer, vgg_loss_fn, scheduler, writer)

    """Load Model"""
    model = torch.load('checkpoints/' + RUN_NAME + '_' + '50' + '.pth')

    """Calculate Parameters and Flops"""
    # print(f"Parameters: {count_parameters(model)}")
    #
    # input = torch.randn(1, 3, 196, 196).to(device)
    # print(f"FLOPS: {count_ops(model, input)}")

    """Reconstruct image"""
    reconstruct_image(model, device, transform)

    """Get Banchmark Results"""
    # banchmark_test(data_loader_test, device, model, writer, 'Feqe')
    # for benchmark_folder in glob.glob('Data/Benchmark/*'):
    #     print(benchmark_folder)
    #     dataset_benchmark = Div2k(benchmark_folder)
    #     data_loader_benchmark = torch.utils.data.DataLoader(dataset_benchmark, batch_size=1,
    #                                                         shuffle=True, drop_last=True, num_workers=0)
    #     banchmark_test(data_loader_benchmark, device, model, writer, name=benchmark_folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    main()
