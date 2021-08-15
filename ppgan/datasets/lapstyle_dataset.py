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
    def __init__(self, content_root, style_root, load_size, crop_size, style_upsize=1):
        super(LapStyleDataset, self).__init__()
        self.content_root = content_root
        self.paths = os.listdir(self.content_root)
        random.shuffle(self.paths)
        self.style_root = style_root
        self.style_upsize = style_upsize
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
            intermediate_width = math.floor(self.load_size* self.style_upsize)
            ratio = style_img.height/style_img.width
            intermediate_height = math.floor(self.load_size*ratio* self.style_upsize)
        else:
            intermediate_height = math.floor(self.load_size* self.style_upsize)
            ratio = style_img.width/style_img.height
            intermediate_width = math.floor(self.load_size* ratio* self.style_upsize)
        style_img = style_img.resize((intermediate_width, intermediate_height),
                                     Image.BILINEAR)
        style_img = style_img.resize((intermediate_width,intermediate_height),Image.BILINEAR)
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
    def __init__(self, content_root, style_root, load_size, crop_size, thumb_size, style_upsize=1):
        super(LapStyleThumbset, self).__init__()
        self.content_root = content_root
        self.paths = os.listdir(self.content_root)
        random.shuffle(self.paths)
        self.style_root = style_root
        self.style_paths = [os.path.join(self.style_root,i) for i in os.listdir(self.style_root)] if self.style_root[-1]=='/' else [self.style_root]
        self.load_size = load_size
        self.crop_size = crop_size
        self.thumb_size = thumb_size
        self.style_upsize = style_upsize
        self.transform = data_transform(self.crop_size)
        self.transform_patch = data_transform(self.load_size)

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
            small_edge='width'
            intermediate_width = self.load_size
            final_width = self.thumb_size
            ratio = content_img.height/content_img.width
            intermediate_height = math.ceil(self.load_size*ratio)
            final_height = math.ceil(self.thumb_size*ratio)
        else:
            small_edge='height'
            final_height = self.thumb_size
            intermediate_height = self.load_size
            ratio = content_img.width/content_img.height
            intermediate_width = math.ceil(self.load_size*ratio)
            final_width = math.ceil(self.thumb_size*ratio)
        content_img = content_img.resize((intermediate_width, intermediate_height),
                                         Image.BILINEAR)
        content_patches = np.array(content_img)
        content_img = content_img.resize((final_width, final_height),
                                         Image.BILINEAR)
        content_img = np.array(content_img)
        if small_edge=='height':
            topmost=self.crop_size #will be divided by content_img
            bottommost=0
            if content_img.shape[0]<self.thumb_size-1:
                leftmost= random.choice(list(range(0, content_img.shape[0] - self.thumb_size,2)))
                rightmost=leftmost+self.crop_size
            else:
                leftmost=0
                rightmost=self.crop_size
        else:
            rightmost=self.crop_size
            leftmost=0
            if content_img.shape[1]<self.thumb_size-1:
                bottommost = random.choice(list(range(0, content_img.shape[1] - self.thumb_size,2)))
                topmost=bottommost+self.crop_size
            else:
                bottommost = 0
                topmost = self.crop_size
        content_img =content_img[bottommost:topmost,leftmost:rightmost]
        content_patches = content_patches[bottommost*2:topmost*2,leftmost*2:rightmost*2]
        randx = random.choice(list(range(0, self.crop_size,2)))
        randy = random.choice(list(range(0, self.crop_size,2)))
        position = [randx, randx + self.crop_size, randy, randy+self.crop_size]
        half_position = [math.floor(randx*.5), math.floor((randx + self.crop_size)*.5), math.floor(randy*.5), math.floor((randy+self.crop_size)*.5)]
        content_patches = content_patches[randx:randx + self.crop_size,
                          randy:randy+self.crop_size]
        style_path = random.choice(self.style_paths) if len(self.style_paths)>1 else self.style_paths[0]
        style_img = cv2.imread(style_path)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        style_img = Image.fromarray(style_img)
        small_edge = min(style_img.width,style_img.height)
        if small_edge==style_img.width:
            intermediate_width = math.floor(self.load_size* self.style_upsize)
            final_width = math.ceil(self.thumb_size*self.style_upsize)
            ratio = style_img.height/style_img.width
            intermediate_height = math.floor(self.load_size*ratio* self.style_upsize)
            final_height = math.ceil(self.thumb_size*ratio* self.style_upsize)
        else:
            intermediate_height = math.floor(self.load_size* self.style_upsize)
            final_height = math.ceil(self.thumb_size * self.style_upsize)
            ratio = style_img.width/style_img.height
            intermediate_width = math.floor(self.load_size* ratio* self.style_upsize)
            final_width = math.ceil(self.thumb_size*ratio* self.style_upsize)
        style_patch = style_img.resize((intermediate_width, intermediate_height),
                                     Image.BILINEAR)
        style_img = style_patch.resize((final_width,final_height),Image.BILINEAR)
        style_img = np.array(style_img)
        style_patch = np.array(style_patch)
        style_img = self.transform(style_img)
        style_patch = self.transform_patch(style_patch)
        style_patch = self.img(style_patch)
        content_img = self.img(content_img)
        style_img = self.img(style_img)
        content_patches = self.img(content_patches)
        return {'ci': content_img, 'si': style_img, 'sp':style_patch, 'ci_path': path,'cp':content_patches,'position':position,'half_position':half_position}

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

def get_crop_bounds(thumb_size,img_width,img_height):
    if thumb_size==img_width:
        leftmost=0
    else:
        leftmost= random.choice(list(range(0, int(img_width - thumb_size),4)))
    rightmost=leftmost+thumb_size
    if thumb_size==img_height:
        bottommost=0
    else:
        bottommost = random.choice(list(range(0, int(img_height - thumb_size),4)))
    topmost=bottommost+thumb_size
    return [leftmost,bottommost,rightmost,topmost]

@DATASETS.register()
class MultiPatchSet(Dataset):
    """
    coco2017 dataset for LapStyle model
    """
    def __init__(self, content_root, style_root, load_size, crop_size, thumb_size, patch_depth,style_upsize=1):
        super(MultiPatchSet, self).__init__()
        self.content_root = content_root
        self.paths = os.listdir(self.content_root)
        random.shuffle(self.paths)
        self.style_root = style_root
        self.style_paths = [os.path.join(self.style_root,i) for i in os.listdir(self.style_root)] if self.style_root[-1]=='/' else [self.style_root]
        self.load_size = load_size
        self.crop_size = crop_size
        self.thumb_size = thumb_size
        self.style_upsize = style_upsize
        self.patch_depth = patch_depth
        self.transform = data_transform(self.crop_size)
        self.transform_patch = data_transform(self.crop_size*2)

    def __getitem__(self, index):
        """Get training sample

        return:
            ci: content image with shape [C,W,H],
            si: style image with shape [C,W,H],
            ci_path: str
        """
        content_stack=[]
        style_stack= []
        position_stack = []
        size_stack = []
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
            small_edge='width'
            intermediate_width = self.load_size
            ratio = content_img.height/content_img.width
            intermediate_height = math.ceil(self.load_size*ratio)
        else:
            small_edge='height'
            final_height = self.thumb_size
            intermediate_height = self.load_size
            ratio = content_img.width/content_img.height
            intermediate_width = math.ceil(self.load_size*ratio)
            final_width = math.ceil(self.thumb_size*ratio)
        content_img = content_img.resize((intermediate_width, intermediate_height),
                                         Image.BILINEAR)

        style_path = random.choice(self.style_paths) if len(self.style_paths)>1 else self.style_paths[0]
        style_img = cv2.imread(style_path)
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        style_img = Image.fromarray(style_img)
        small_edge = min(style_img.width,style_img.height)
        if small_edge==style_img.width:
            intermediate_width = math.floor(self.load_size* self.style_upsize)
            final_width = math.ceil(self.thumb_size*self.style_upsize)
            ratio = style_img.height/style_img.width
            intermediate_height = math.floor(self.load_size*ratio* self.style_upsize)
            final_height = math.ceil(self.thumb_size*ratio* self.style_upsize)
        else:
            intermediate_height = math.floor(self.load_size* self.style_upsize)
            final_height = math.ceil(self.thumb_size * self.style_upsize)
            ratio = style_img.width/style_img.height
            intermediate_width = math.floor(self.load_size* ratio* self.style_upsize)
            final_width = math.ceil(self.thumb_size*ratio* self.style_upsize)
        style_img = style_img.resize((intermediate_width, intermediate_height),
                                     Image.BILINEAR)
        style_img = style_img.crop(box=get_crop_bounds(self.load_size,style_img.width,style_img.height))
        style_patch = style_img.resize((self.crop_size,self.crop_size))
        style_patch = np.array(style_patch)
        style_patch = self.img(style_patch)
        style_stack.append(style_patch)
        content_img = content_img.crop(box=get_crop_bounds(self.load_size,content_img.width,content_img.height))
        content_patch = content_img.resize((self.crop_size,self.crop_size))
        content_patch = np.array(content_patch)
        content_patch = self.img(content_patch)
        content_stack.append(content_patch)
        for i in range(self.patch_depth):
            content_patch = content_img
            for c in position_stack:
                content_patch=content_patch.crop(box=(c[0],c[1],c[2],c[3]))
            size_stack.append(content_patch.width)
            position_stack.append(get_crop_bounds(content_patch.width/2,content_patch.width,content_patch.height))
            content_patch=content_patch.crop(box=(position_stack[-1][0],position_stack[-1][1],position_stack[-1][2],position_stack[-1][3]))
            content_patch = content_patch.resize((self.crop_size,self.crop_size),
                                                 Image.BILINEAR)
            content_patch = np.array(content_patch)
            content_patch = self.img(content_patch)
            content_stack.append(content_patch)

        style_stack.append(self.img(np.array(style_img)))

        output = {}
        for idx,i in enumerate(content_stack):
            output['content_stack_'+str(idx+1)]=i
        for idx,i in enumerate(style_stack):
            output['style_stack_'+str(idx+1)]=i
        output['position_stack']=position_stack
        output['content']=self.img(np.array(content_img))
        output['size_stack']=size_stack
        return output

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
        return 'MultiPatchSet'

@DATASETS.register()
class LapStyleThumbsetInference(Dataset):
    """
    coco2017 dataset for LapStyle model
    """
    def __init__(self, content_root, style_root, load_size, crop_size, thumb_size, patch_depth,style_upsize=1):
        super(LapStyleThumbsetInference, self).__init__()
        self.content_root = content_root
        self.paths = os.listdir(self.content_root)
        random.shuffle(self.paths)
        self.style_root = style_root
        self.style_paths = [os.path.join(self.style_root,i) for i in os.listdir(self.style_root)] if self.style_root[-1]=='/' else [self.style_root]
        self.load_size = load_size
        self.crop_size = crop_size
        self.thumb_size = thumb_size
        self.style_upsize = style_upsize
        self.patch_depth = patch_depth
        self.transform = data_transform(self.crop_size)
        self.transform_patch = data_transform(self.crop_size*2)
        style_img = cv2.imread(self.style_paths[0])
        style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)
        self.style_img = Image.fromarray(style_img)

    def __getitem__(self, index):
        """Get training sample
        return:
            ci: content image with shape [C,W,H],
            si: style image with shape [C,W,H],
            ci_path: str
        """
        content_stack=[]
        style_stack= []
        position_stack = []
        size_stack = []
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
            small_edge='width'
            intermediate_width = self.load_size
            ratio = content_img.height/content_img.width
            #reduce_ratio = content_img.width/content_img.height
            intermediate_height = math.ceil(self.load_size*ratio)
            final_width = self.thumb_size
            final_height = math.ceil(self.thumb_size*ratio)
        else:
            small_edge='height'

            intermediate_height = self.load_size
            ratio = content_img.width/content_img.height
            #reduce_ratio = content_img.width/content_img.height
            intermediate_width = math.ceil(self.load_size*ratio)
            final_height = self.thumb_size
            final_width = math.ceil(self.thumb_size*ratio)
        content_img = content_img.resize((intermediate_width, intermediate_height),
                                         Image.BILINEAR)
        content_thumb = content_img.resize((final_width, final_height),
                                         Image.BILINEAR)
        style_path = random.choice(self.style_paths) if len(self.style_paths)>1 else self.style_paths[0]
        small_edge = min(self.style_img.width,self.style_img.height)
        max_size=max(final_height,final_width)
        if small_edge==self.style_img.width:
            intermediate_width = math.floor(self.load_size* self.style_upsize)
            final_width = math.ceil(max_size* self.style_upsize)
            ratio = style_img.height/self.style_img.width
            intermediate_height = math.floor(self.load_size*ratio* self.style_upsize)
            final_height = math.ceil((max_size*self.style_upsize)*ratio)
        else:
            intermediate_height = math.floor(self.load_size* self.style_upsize)
            final_height = math.ceil(max_size* self.style_upsize)
            ratio = self.style_img.width/self.style_img.height
            intermediate_width = math.floor(self.load_size* ratio* self.style_upsize)
            final_width = math.ceil(max_size* self.style_upsize*ratio)
        style_thumb = self.style_img.resize((final_width,final_height))
        transform = data_transform((content_thumb.height,content_thumb.width))
        style_thumb = transform(style_thumb)
        style_img = self.style_img.resize((intermediate_width, intermediate_height),
                                     Image.BILINEAR)
        style_img = style_img.crop(box=get_crop_bounds(self.load_size,style_img.width,style_img.height))

        style_img = np.array(style_img)
        content_img = np.array(content_img)
        content_thumb = np.array(content_thumb)
        content_thumb = self.img(content_thumb)
        style_thumb = np.array(style_thumb)
        style_thumb = self.img(style_thumb)
        style_img = self.img(style_img)
        sizes=style_thumb.shape
        ratio = math.floor(self.load_size/self.crop_size)
        content_img = self.img(content_img)
        #content_img = np.expand_dims(content_img, axis=0)
        if sizes[-1]%16!=0:
            closest=math.floor(sizes[-1]/16)
            style_thumb=style_thumb[:,:,:closest*16]
            content_thumb=content_thumb[:,:,:closest*16]
            content_img = content_img[:,:,:closest*16*ratio]
        if sizes[-2]%16!=0:
            closest=math.floor(sizes[-2]/16)
            style_thumb=style_thumb[:,:closest*16,:]
            content_thumb=content_thumb[:,:closest*16,:]
            content_img = content_img[:,:closest*16*ratio,:]
        assert content_thumb.shape == style_thumb.shape
        for i in range(self.patch_depth):
            bottommost = random.choice(list(range(0, content_img.shape[1] - content_thumb.shape[1],2)))
            topmost=bottommost+content_thumb.shape[1]
            leftmost = random.choice(list(range(0, content_img.shape[2] - content_thumb.shape[2],2)))
            rightmost = leftmost+content_thumb.shape[2]
            position_stack.append((math.floor(bottommost/2),math.floor(topmost/2),math.floor(leftmost/2),math.floor(rightmost/2)))
        #output = {'content':content_img,'style':style_img,'content_thumb':zero_thumb,'style_thumb':style_thumb,'content_shape':thumb_shape}
        output={'content':content_img,'ci':content_thumb,'position':position_stack,'cp':content_thumb,'si':style_thumb,'sp':style_thumb,'style':style_img,'ci_path':path}
        return output

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
        return 'LapStyleThumbsetInference'