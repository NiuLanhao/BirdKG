import os
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from resnet_model import resnet50, resnet101
# from resnet_cbam import resnet50
# from resnet_se import resnet50
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

class_name = 'Rusty_Blackbird'
name = 'resnet101-pre'


def main():
    model = resnet101(num_classes=200)
    weights_path = "../ResNet/weights/" + name + ".pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

    target_layers = [model.layer4]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "../datasets/" + class_name + ".jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    # target_category = 20  # Eastern_Towhee
    # target_category = 117  # House_Sparrow
    target_category = 10  # Rusty_Blackbird
    # target_category = 129  # Tree_Swallow

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.title(name)
    plt.imshow(visualization)
    plt.savefig(class_name + '/' + name + '.png', bbox_inches='tight', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
