# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os, math
import numpy as np
import random
from PIL import Image
import paddle
import paddle.vision.transforms as T
from paddle.io import Dataset
import cv2
import warnings
warnings.filterwarnings("ignore")

from .builder import DATASETS

logger = logging.getLogger(__name__)


def data_transform(crop_size):
    transform_list = [T.RandomCrop(crop_size)]
    return T.Compose(transform_list)


@DATASETS.register()
class LapStyleDataset(Dataset):
    """
    coco2017 dataset for LapStyle model
    """
    def __init__(self, content_root, style_root, load_size, crop_size):
        super(LapStyleDataset, self).__init__()
        self.content_root = content_root
        self.paths = os.listdir(self.content_root)
        random.shuffle(self.paths)
        self.style_root = style_root
        self.style_paths = [os.path.join(self.style_root,i) for i in os.listdir(self.style_root)] if self.style_root[-1]=='/' else [self.style_root]
        self.load_size = load_size
        self.crop_size = crop_size
        self.transform = data_transform(self.crop_size)

    def __getitem__(self, index):
        """Get training sample

        return:
            ci: content image with shape [C,W,H],
            si: style image with shape [C,W,H],
            ci_path: str
        """
        path = self.paths[index]
        content_img = cv2.imread(os.path.join(self.content_root, path))
        try:
            if content_img.ndim == 2:
                content_img = cv2.cvtColor(content_img, cv2.COLOR_GRAY2RGB)
            else:
                content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        except:
            print(path)
        content_img = Image.fromarray(content_img)
        small_edge = min(content_img.width,content_img.height)
        if small_edge==content_img.width:
            intermediate_width = self.load_size
            ratio = content_img.height/content_img.width
            intermediate_height = math.floor(self.load_size*ratio)
        else:
            intermediate_height = self.load_size
            ratio = content_img.width/content_img.height
            intermediate_width = math.floor(self.load_size*ratio)
        content_img = content_img.resize((intermediate_width, intermediate_height),
                                         Image.BILINEAR)
        content_img = np.array(content_img)
        style_path = random.choice(self.style_paths) if len(self.style_paths)>1 else self.style_paths[0]
        style_img = cv2.imread(style_path)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        style_img = Image.fromarray(style_img)
        small_edge = min(style_img.width,style_img.height)
        if small_edge==style_img.width:
            intermediate_width = self.load_size
            ratio = style_img.height/style_img.width
            intermediate_height = math.floor(self.load_size*ratio)
        else:
            intermediate_height = self.load_size
            ratio = style_img.width/style_img.height
            intermediate_width = math.floor(self.load_size*ratio)
        style_img = style_img.resize((intermediate_width, intermediate_height),
                                     Image.BILINEAR)
        style_img = np.array(style_img)
        content_img = self.transform(content_img)
        style_img = self.transform(style_img)
        content_img = self.img(content_img)
        style_img = self.img(style_img)
        return {'ci': content_img, 'si': style_img, 'ci_path': path}

    def img(self, img):
        """make image with [0,255] and HWC to [0,1] and CHW

        return:
            img: image with shape [3,W,H] and value [0, 1].
        """
        # [0,255] to [0,1]
        img = img.astype(np.float32) / 255.
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1)).astype('float32')
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'LapStyleDataset'

@DATASETS.register()
class LapStyleThumbset(Dataset):
    """
    coco2017 dataset for LapStyle model
    """
    def __init__(self, content_root, style_root, load_size, crop_size, thumb_size):
        super(LapStyleThumbset, self).__init__()
        self.content_root = content_root
        self.paths = os.listdir(self.content_root)
        random.shuffle(self.paths)
        self.style_root = style_root
        self.style_paths = [os.path.join(self.style_root,i) for i in os.listdir(self.style_root)] if self.style_root[-1]=='/' else [self.style_root]
        self.load_size = load_size
        self.crop_size = crop_size
        self.thumb_size = thumb_size
        self.transform = data_transform(self.crop_size)

    def __getitem__(self, index):
        """Get training sample

        return:
            ci: content image with shape [C,W,H],
            si: style image with shape [C,W,H],
            ci_path: str
        """
        path = self.paths[index]
        content_img = cv2.imread(os.path.join(self.content_root, path))
        try:
            if content_img.ndim == 2:
                content_img = cv2.cvtColor(content_img, cv2.COLOR_GRAY2RGB)
            else:
                content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        except:
            print(path)
        content_img = Image.fromarray(content_img)
        small_edge = min(content_img.width,content_img.height)
        if small_edge==content_img.width:
            intermediate_width = self.load_size
            final_width = self.crop_size
            ratio = content_img.height/content_img.width
            intermediate_height = math.floor(self.load_size*ratio)
            final_height = math.floor(self.crop_size*ratio)
        else:
            intermediate_height = self.load_size
            final_height = self.crop_size
            ratio = content_img.width/content_img.height
            intermediate_width = math.floor(self.load_size*ratio)
            final_width = math.floor(self.crop_size*ratio)
        randx = np.random.randint(0,intermediate_width - self.thumb_size)
        randy = np.random.randint(0, intermediate_height - self.thumb_size)
        position = [randx, randx+self.thumb_size, randy, randy+self.thumb_size]
        content_patches = content_img.resize((intermediate_width, intermediate_height),
                                         Image.BILINEAR)
        content_img = content_patches.resize((final_width, final_height),
                                         Image.BILINEAR)
        content_patches = np.array(content_patches)
        content_patches = content_patches[randx:randx + self.thumb_size,
                          randy:randy+self.thumb_size]
        content_img = np.array(content_img)
        style_path = random.choice(self.style_paths) if len(self.style_paths)>1 else self.style_paths[0]
        style_img = cv2.imread(style_path)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        style_img = Image.fromarray(style_img)
        small_edge = min(style_img.width,style_img.height)
        if small_edge==style_img.width:
            intermediate_width = self.thumb_size
            ratio = style_img.height/style_img.width
            intermediate_height = math.floor(self.thumb_size*ratio)
        else:
            intermediate_height = self.thumb_size
            ratio = style_img.width/style_img.height
            intermediate_width = math.floor(self.thumb_size*ratio)
        style_img = style_img.resize((intermediate_width, intermediate_height),
                                     Image.BILINEAR)
        style_img = np.array(style_img)
        style_img = self.transform(style_img)
        content_img = self.img(content_img)
        style_img = self.img(style_img)
        content_patches = self.transform(content_patches)
        content_patches = self.img(content_patches)
        return {'ci': content_img, 'si': style_img, 'ci_path': path,'cp':content_patches,'position':position}

    def img(self, img):
        """make image with [0,255] and HWC to [0,1] and CHW

        return:
            img: image with shape [3,W,H] and value [0, 1].
        """
        # [0,255] to [0,1]
        img = img.astype(np.float32) / 255.
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1)).astype('float32')
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'LapStyleThumbset'