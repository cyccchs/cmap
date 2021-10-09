from .custom_function import extract_glimpse, extract_multiple_glimpse

class RetinaSensor:
    def __init__(self, img_size, glimpse_size):
        self.img_size = img_size
        self.glimpse_size = glimpse_size
    def __call__(self, img_batch, location):
        img = torch.reshape(img_batch, (img_batch[0], 3, self.img_size, self.img_size))
        img = torch.reshape(img, [location[0], self.glimpse*self.glimpse*3])
        return img

class LocationNetwork:
    def __init__(self, hidden_size, loc_dim, std=0.22, is_sampling=True):
        self.loc_dim = loc_dim
        self.std = std
        self.w = weight_variable((hidden_size, loc_dim))
        self.b = bias_variable((loc_dim,))
        self.is_sampling = is_sampling

    def __cal__(self, hidden_state):
        mean_t = 
