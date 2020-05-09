
class SGD():
    def __init__(num_iter, discriminator_steps=1):
        self.num_iter = num_iter
        self.discriminator_steps = discriminator_steps


    def calc_discriminator_gradient():
    

    def calc_generator_gradient():


    def train_discriminator():


    def train_generator():


    def train():
        for _ in range(self.num_iter):
            for _ in range(self.discriminator_steps):
                # sample minibatch from noise prior - p_g
                # sample minibatch from data generating distribution - p_data
                # update discriminator
            
            # sample minibatch from noise prior - p_g
            # update generator