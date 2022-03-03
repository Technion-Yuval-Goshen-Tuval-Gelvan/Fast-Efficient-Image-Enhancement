from Utils import *
import torch
import torchvision
from FEQEModel import FeqeModel
from DataSet import Div2k
from torch.utils.tensorboard import SummaryWriter
import time


RUN_NAME = "firsttry"
LOW_RES_SCALE = 2
# --- Hyper Parameters ---
BATCH_SIZE = 8
TRAIN_IM_SIZE = (196, 196)
NUM_WORKERS = 0
RESIDUAL_BLOCKS = 5
CHANNELS = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
CHECKPOINT_EVERY = 5


if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on device:", device)


transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(TRAIN_IM_SIZE),
    ])
dataset_train = Div2k('Data/Train/', transform=transform)
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=0)

dataset_test = Div2k('Data/Test/', transform=transform)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=100,
                                               shuffle=True, drop_last=True, num_workers=0)

model = FeqeModel(num_residual_blocks=RESIDUAL_BLOCKS, channels=CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                             betas=(0.9, 0.999), eps=1e-08)

writer = SummaryWriter(comment=RUN_NAME)

for epoch in range(EPOCHS+1):
    start = time.time()
    for i, im_batch in enumerate(data_loader_train):
        im_batch = im_batch.to(device)
        im_batch_lowres = down_sample(im_batch, LOW_RES_SCALE)

        im_batch_lowres_recon = model(im_batch_lowres)
        loss = torch.nn.MSELoss()(im_batch_lowres_recon, im_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss.item(), epoch*len(data_loader_train)+i)
        writer.add_scalar('Metrics/PSNR_Train', psnr(im_batch*255, im_batch_lowres_recon*255), epoch*len(data_loader_train)+i)

    # evaluate on test set
    with torch.no_grad():
        test_avg_psnr = 0
        for i, im_batch in enumerate(data_loader_test):
            im_batch = im_batch.to(device)
            im_batch_lowres = down_sample(im_batch, LOW_RES_SCALE)

            im_batch_lowres_recon = model(im_batch_lowres)

            test_avg_psnr += psnr(im_batch*255, im_batch_lowres_recon*255)
        test_avg_psnr /= len(data_loader_test)
        writer.add_scalar('Metrics/PSNR_Test', test_avg_psnr, epoch*len(data_loader_test)+i)

    if epoch % CHECKPOINT_EVERY == 0:
        torch.save(model, 'checkpoints/' + RUN_NAME + '_' + str(epoch) + '.pth')

    print("Epoch: {}/{}".format(epoch, EPOCHS), "Time: {:.2f}".format(time.time() - start))

