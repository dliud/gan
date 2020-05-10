import torch
import torch.nn as nn
import utils
import modules


class GAN:
    def __init__(self, discriminator_steps=2, disc_input_dim=784, 
                gen_input_dim=100, batch_size=10, lr_disc=.0002, 
                lr_gen=.0002):
        self.discriminator_steps = discriminator_steps
        self.disc_input_dim = disc_input_dim
        self.gen_input_dim = gen_input_dim
        self.batch_size = batch_size

        self.discriminator = modules.Discriminator(disc_input_dim)
        self.generator = modules.Generator(gen_input_dim)
        
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr_disc)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr_gen)
        self.loss_function = nn.BCELoss()
        self.constNoise = torch.randn(10, self.gen_input_dim)

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
        error_r = self.loss_function(prediction_r, torch.ones(real_data.size(0), 1))  # real
        error_r.backward()

        prediction_f = self.discriminator(fake_data)
        error_f = self.loss_function(prediction_f, torch.zeros(fake_data.size(0), 1))  # fake
        error_f.backward()

        self.d_optimizer.step()

        """
        # FOR PLOTTING
        self.lst_disc_loss.append(error_r + error_f)
        """

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

        """
        # FOR PLOTTING
        self.lst_gen_loss.append(error_f)
        """

        return error_f

    def train(self, data_loader, num_epoch = 100):
        """
        For each epoch in num_epochs:
            Train discriminator for discriminator_step iterations.
                1. sample minibatch from data generating distribution - p_data
                2. sample minibatch from noise prior - p_g
                3. update discriminator via SGA
            Train generator for 1 iteration.
                1. sample minibatch from noise prior - p_g
                2. update generator via SGA
        """
        for epoch in range(num_epoch):
            print(epoch)
            self.lst_epochs.append(epoch) # FOR PLOTTING

            if epoch%10 == 0: self.sampleImages(epoch)
            for n_batch, real_data in enumerate(data_loader): #n_batch 0,1,2..., real_data (batch_size, 784)
                loss = 0
                for _ in range(self.discriminator_steps):
                    noise = torch.randn(self.batch_size, self.gen_input_dim)
                    fake_data = self.generator(noise).detach()
                    disc_loss = self.train_discriminator(real_data, fake_data)

                    self.lst_disc_loss.append(disc_loss.item())

                # print("Discriminator loss: ", disc_loss)
                gen_noise = torch.randn(self.batch_size, self.gen_input_dim)
                gen_loss = self.train_generator(gen_noise)
                # print("Generator loss: ", gen_loss)

                self.lst_gen_loss.append(gen_loss.item())
        
        utils.plot_loss(self.lst_epochs, self.lst_disc_loss, self.lst_gen_loss, "loss")

    def sampleImages(self, epoch):
        #generates and displays fake images using the current model
        synth_data = self.generator(self.constNoise)
        utils.vector_to_img(synth_data, "./images/{}.jpg".format(iter))
        