import random
import math

import torch


class RandomErase:
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.33, value='mean'):
        """
        Randomly erase the input Tensor Image with given value or random value
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        :param p: probability of undergoing random erasing
        :param sl: Minimum proportion of erased area against input image
        :param sh: Maximum proportion of erased area against input image.
        :param r1: Minimum aspect ratio of erasing rectangle region
        :param r2: Maximum aspect ratio of erasing rectangle region
        :param value: 'random', 'mean' or certain values
        """

        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2
        self.value = value

    def __call__(self, img_tensor):
        assert img_tensor.shape[0] == 3

        if random.uniform(0, 1) >= self.p:
            return img_tensor

        if self.value is 'mean':
            value = torch.tensor([img_tensor[0].mean(), img_tensor[1].mean(), img_tensor[2].mean()])
        elif self.value is 'random':
            value = torch.rand(3)
        elif isinstance(self.value, (tuple, list)):
            assert len(self.value) == 3
            value = self.value
        else:
            raise ValueError

        for attempt in range(100):
            area = img_tensor.shape[1] * img_tensor.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, self.r2)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_tensor.shape[2] and h < img_tensor.shape[1]:
                x1 = random.randint(0, img_tensor.shape[1] - h)
                y1 = random.randint(0, img_tensor.shape[2] - w)

                img_tensor[0, x1:x1 + h, y1:y1 + w] = value[0]
                img_tensor[1, x1:x1 + h, y1:y1 + w] = value[1]
                img_tensor[2, x1:x1 + h, y1:y1 + w] = value[2]

                # print(target_area / area, aspect_ratio)
                return img_tensor

        return img_tensor


if __name__ == '__main__':
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.transforms import Resize, ToTensor
    import torchvision.transforms.functional as TF
    import PIL
    from PIL import Image
    import cv2

    transform = transforms.Compose([
        Resize((256, 128)),
        ToTensor(),
        RandomErase(1)
    ])


    def tensor2np(tensor):
        if tensor.shape[0] == 4:
            tensor = tensor.squeeze(0)
        img = tensor.mul(255).byte()
        img = img.cpu().numpy().transpose((1, 2, 0))
        return img


    def show_tensor_img(img_tensor, title=None):
        img = tensor2np(img_tensor)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        title = "" if not title else title
        cv2.imshow(title, img)
        cv2.waitKey()

        return img


    for i in range(10):
        img = Image.open('../images/0002_c1s1_000551_01.jpg')
        img_tensor = transform(img)
        show_tensor_img(img_tensor)
