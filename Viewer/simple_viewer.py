import torch
import cv2


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def view_float_image(image):
    image = image * 255.
    image = torch.clip(image, 0, 255)
    image = image.to(torch.uint8)
    image = image.permute(1, 2, 0)
    print(image.shape)
    image = image.cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("", image)
    cv2.waitKey()


def view_imagenet_normalized_float_image(image):
