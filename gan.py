import torch
import torch.nn as nn

import utils
import modules


class GAN(torch.nn.Module):
    def __init__(self, trial, id, discriminator_steps=1, generator_steps=2, 
                 disc_input_dim=784, gen_input_dim=100,
                 lr_disc=.0002, lr_gen=.0002, label_smooth=False, alpha=4):
        super(GAN, self).__init__()
       
        self.trial = trial
        self.id = id
        self.discriminator_steps = discriminator_steps
        self.generator_steps = generator_steps
        self.disc_input_dim = disc_input_dim
        self.gen_input_dim = gen_input_dim
        self.label_smooth = label_smooth

        self.discriminator = modules.Discriminator(disc_input_dim, alpha = alpha)
        self.generator = modules.Generator(gen_input_dim, alpha = alpha)

        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_gen)
        self.loss_function = nn.BCELoss()
        self.constNoise = torch.randn(25, self.gen_input_dim)

        self.lst_epochs = []
        self.lst_disc_loss = []
        self.lst_gen_loss = []

    def train_discriminator(self, real_data, fake_data):
        """
        #real_data, fake_data, tensors(n examples, imageSize(784)) all entries in [-1.0, 1.0]
        :return:
        """
        self.d_optimizer.zero_grad()

        prediction_r = self.discriminator(real_data)
        target = torch.ones(real_data.size(0), 1)
        if self.label_smooth: target = .9 * target
        error_r = self.loss_function(prediction_r, target)  # real
        error_r.backward()

        prediction_f = self.discriminator(fake_data)
        error_f = self.loss_function(prediction_f, torch.zeros(fake_data.size(0), 1))  # fake
        error_f.backward()

        self.d_optimizer.step()

        return error_r + error_f

    def train_generator(self, noise_data):
        """
        :param: optimizer: Adam
        :param: noise_data: shape (gen_input_dim), filled with values randomly sampled
                            from a Gaussian distribution
        :return: can optionally return the loss for plotting or something
        """
        self.g_optimizer.zero_grad()

        synth_data = self.generator(noise_data)
        prediction_f = self.discriminator(synth_data)
        error_f = self.loss_function(prediction_f, torch.ones(synth_data.size(0), 1))
        error_f.backward()

        self.g_optimizer.step()

        return error_f

    def train(self, data_loader, num_epoch, start_epoch=0, printProgress=False, updateEvery=50):
        """
        For each epoch in num_epochs:
            Train discriminator for discriminator_step iterations.
                1. sample minibatch from data generating distribution - p_data
                2. sample minibatch from noise prior - p_g
                3. update discriminator via SGA
            Train generator for generator_step iterations.
                1. sample minibatch from noise prior - p_g
                2. update generator via SGA
        """
        for epoch in range(start_epoch, num_epoch):
            if ((epoch + 1) % updateEvery) == 0:
                print("Epoch: {}".format(epoch + 1))
                if printProgress: self.sample_images(epoch + 1)
            disc_loss, gen_loss = 0, 0
            for n_batch, real_data in enumerate(data_loader):  #real_data is of shape (batch_size, 784)
                for _ in range(self.discriminator_steps):
                    noise = torch.randn(real_data.shape[0], self.gen_input_dim)
                    fake_data = self.generator(noise).detach()
                    
                    disc_loss += self.train_discriminator(real_data, fake_data).item() / self.discriminator_steps
                for _ in range(self.generator_steps):
                    gen_noise = torch.randn(real_data.shape[0], self.gen_input_dim)
                    gen_loss += self.train_generator(gen_noise).item() / self.generator_steps

            # print("Discriminator loss: ", disc_loss)
            # print("Generator loss: ", gen_loss)

            # the three lines below are for plotting
            self.lst_epochs.append(epoch)
            self.lst_disc_loss.append(disc_loss)
            self.lst_gen_loss.append(gen_loss)
        
        print("Saving model with ID", self.id, "at epoch", num_epoch)
        utils.save_model(self, self.trial, self.id, num_epoch)
        utils.plot_loss(self.lst_epochs, self.lst_disc_loss, self.lst_gen_loss, "./outputs/trial{}/gan{}/loss{}-{}".format(self.trial, self.id, start_epoch, num_epoch))

    def sample_images(self, epoch):
        """
        Generates and displays fake images using the current model.
        """
        synth_data = self.generator(self.constNoise)
        utils.vector_to_img(synth_data, "./outputs/trial{}/gan{}/epoch{}.jpg".format(self.trial, self.id, epoch))
