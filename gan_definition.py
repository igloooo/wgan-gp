import torch
import torch.nn as nn


DIM = 64 # Model dimensionality
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

# ==================Definition Start======================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(), # don't use inplace ReLU
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        return output.view(-1, OUTPUT_DIM)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.ReLU(),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        return out.view(-1)

class Dataset_from_GAN:
    """
    create the instance after netG has been moved to device
    netG should not requires_grad

    the dataset should also be saved to checkpoint
    """
    def __init__(self, netG, batch_size, n_data=60000):
        self.netG = netG
        self.batch_size = batch_size

        self.cur = 0
        self.n_data = n_data
        # get the device that netG is on
        self.device = next(self.netG.parameters()).device
        noises = torch.randn(self.n_data, 128).to(self.device)
        images = []
        for i in range(n_data//100):
            images.append(self.netG(noises[(100*i):(100*i+100)]))
        self.images = torch.cat(images, dim=0)

    def __iter__(self):
        # initialize and return self
        self.cur = 0
        return self

    def __next__(self):
        """
        return a length-1 list to be consistent with DataLoader,
        the element is a shape (batch_size, OUTPUT_DIM) tensor
        """
        if self.cur + self.batch_size <= self.n_data:
            image_batch = self.images[self.cur:(self.cur+self.batch_size),:]
            self.cur = self.cur + self.batch_size
            return [image_batch]
        else:
            raise StopIteration

    def load_state_dict(self, state_dict):
        # load the noise
        self.images = state_dict['images'].to(self.device)
        self.n_data = self.images.shape[0]

    def state_dict(self):
        return {'images': self.images}
