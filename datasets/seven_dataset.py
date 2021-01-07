from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset

class SevenDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SevenDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[585/640, 0, 320/640, 0],
                           [0, 585/480, 240/480, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (640,480)

    def check_depth(self):
        return True

    def get_color(self, folder, frame_index, side, do_flip):
        f_str = "frame-{:06d}.color{}".format(frame_index, ".png")
        image_path = os.path.join(self.data_path, folder, f_str)
        color = self.loader(image_path)
        color = color.resize(self.full_res_shape, pil.NEAREST)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "frame-{:06d}.depth{}".format(frame_index, ".png")
        depth_path = os.path.join(self.data_path, folder, f_str)
        print("f_str_depth={}".format(depth_path))
        depth = pil.open(depth_path)
        # depth = depth.resize(self.full_res_shape, pil.NEAREST)
        depth = np.array(depth)
        depth[depth==65535]=100000000
        depth_gt = depth.astype(np.float)*0.001
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    