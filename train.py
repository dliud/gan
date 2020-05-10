
class SGD:
    def __init__(self, num_iter, discriminator_steps=1):
        self.num_iter = num_iter
        self.discriminator_steps = discriminator_steps


    def calc_discriminator_gradient(self, prior_samples, data_samples):
        """
        :param prior_samples: Shape (batch_size)
        :param data_samples: Shape (batch_size)
        :return:
        """
    

    def calc_generator_gradient(self, prior_samples):
        """
        :param prior_samples: Shape (batch_size)
        :return:
        """


    def train_discriminator(self):
        """

        :return:
        """

    def train_generator(self):
        """

        :return:
        """

    def train(self):
        for _ in range(self.num_iter):
            for _ in range(self.discriminator_steps):
                # sample minibatch from noise prior - p_g
                # sample minibatch from data generating distribution - p_data
                # update discriminator
            
            # sample minibatch from noise prior - p_g
            # update generator