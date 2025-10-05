from torchvision import transforms as T

class SimCLRAugmentations:
    """
    Return two augmented views when called on a PIL image.
    """
    def __init__(self, image_size=32):
        self.train_transform = T.Compose([
            T.RandomResizedCrop(size=image_size),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8), 
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=5), 
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __call__(self, x):
        """
        Returns a tuple of two augmented images.
        """
        x1 = self.train_transform(x)
        x2 = self.train_transform(x)
        return x1, x2

def get_simclr_augment(image_size=32):
    return SimCLRAugmentations(image_size=image_size)


    