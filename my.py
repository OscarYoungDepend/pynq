import torch
import torch.nn as nn
import torchvision
from Prepare_image import Image_load
from PIL import Image
import argparse
import os
import serial
import numpy as np

class MTA(nn.Module):
	def __init__(self):
		super(MTA, self).__init__()
		self.backbone = torchvision.models.resnet50(pretrained=False)
		fc_feature = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(fc_feature, 6, bias=True)

	def forward(self, x):
		result = self.backbone(x)
		return result


class Demo(object):
    def __init__(self, config, load_weights=True, checkpoint_dir='/home/xilinx/jupyter_notebooks/spaq/weights/MT-A_release.pt'):
        self.config = config
        self.load_weights = load_weights
        self.checkpoint_dir = checkpoint_dir
        self.prepare_image = Image_load(size=512, stride=224)

        self.model =MTA()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model_name = type(self.model).__name__

        if self.load_weights:
            self.initialize()

    def predit_quality(self):
        imagenames = []
        scores = []
        path = '/home/xilinx/jupyter_notebooks/spaq/images'
        for filename in os.listdir(path):
            #print(filename)  # 仅仅是为了测试
            # img = cv2.imread(directory_name + "/" + filename)
            image = self.prepare_image(Image.open(path + '/' + filename).convert("RGB"))
            imagenames.append(filename)
            image = image.to(self.device)
            print('评估中。。。')
            score = self.model(image)[:,0].mean()
            #images.append(image)
            scores.append(score)
        return scores,imagenames

    # image_2 = self.prepare_image(Image.open(self.config.image_2).convert("RGB"))

    # image_1 = image_1.to(self.device)
    # score_1 = self.model(image_1).mean()
    # print(score_1.item())
    # image_2 = image_2.to(self.device)
    # score_2 = self.model(image_2).mean()
    # print(score_2.item())

    def initialize(self):
        ckpt_path = self.checkpoint_dir
        could_load = self._load_checkpoint(ckpt_path)
        if could_load:
            print('成功读取预训练BIQA模型：MT-A_release.pt')
        else:
            raise IOError('Fail to load the pretrained model')

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt, map_location='cpu')
            self.model.load_state_dict(checkpoint['state_dict'])
            #self.model=checkpoint
            #torch.save(self.model, './weights/mymodel.pth')
            return True
        else:
            return False


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_1', type=str, default='./images/99998.png')
    parser.add_argument('--image_2', type=str, default='./images/99999.png')

    return parser.parse_args()


# def main():
# 	cfg = parse_config()
# 	t = Demo(config=cfg)
# 	t.predit_quality()

def test():
    with torch.no_grad():
        #s2=[]
        cfg = parse_config()
        t = Demo(config=cfg)
        scores,imagenames = t.predit_quality()
        for i in range(len(scores)):
            print('图像的质量评分为:'+str(scores[i].cpu().numpy()))
    return  str(np.round(scores[i].cpu().numpy(),3))
            #s2.append(scores[i].cpu().numpy())
            
