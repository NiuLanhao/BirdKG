import os
import sys
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.realpath(__file__))
# 将当前脚本所在的目录添加到 Python 的搜索路径中
sys.path.append(script_dir)
from resnet_model import resnet101


def predict(img, confidence_threshold=0.4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = os.path.join(script_dir, '..', 'datasets', 'class_indices.json')
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r", encoding='utf-8') as f:
        class_indict = json.load(f)

    # create model
    model = resnet101(num_classes=200).to(device)

    # load model weights
    # 构建 json_path
    weights_path = os.path.join(script_dir, 'weights', 'resnet101-pre.pth')
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    filtered_results = []
    # 筛选置信度大于40%的类别
    if predict[predict_cla].item() > confidence_threshold:
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        print(print_res)
        filtered_results.append({
            'class': class_indict[str(predict_cla)],
            'prob': float(predict[predict_cla].numpy())
        })

    return filtered_results


if __name__ == '__main__':
    img_path = "../datasets/Black_Footed_Albatross.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    predict(img)
