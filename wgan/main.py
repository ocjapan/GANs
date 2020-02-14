import wgan
import visdom
import adabound
import torch

D = wgan.D()

# During discriminator forward-backward-update
D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
# During generator forward-backward-update
G_loss = -torch.mean(D_fake)