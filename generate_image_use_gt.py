# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from layers import *
from kitti_utils import *
from utils import *
from pose_error import *
from options import MonodepthOptions
from datasets import SevenDataset
import networks

class Evaluation:
    def __init__(self, option):
        self.options = option

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        opt = self.options
        device = torch.device("cpu" if opt.no_cuda else "cuda")
        backproject_depth = {}
        project_3d = {}
        for scale in opt.scales:
            h = opt.height // (2 ** scale)
            w = opt.width // (2 ** scale)

            backproject_depth[scale] = BackprojectDepth(opt.batch_size, h, w)
            backproject_depth[scale].to(device)

            project_3d[scale] = Project3D(opt.batch_size, h, w)
            project_3d[scale].to(device)

        for scale in opt.scales:
            disp = outputs[("disp", scale)]
            if opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [opt.height, opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, opt.min_depth, opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")   
                
                if not opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def evaluate(self):
        opt = self.options
        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        opt.frame_ids = [0, 50]  # pose network only takes two frames as input
        outputs = {}
        transLossAmount = 0.0
        rotLossAmount = 0.0

        filenames = readlines(
            os.path.join(os.path.dirname(__file__), "splits", opt.split, "test_files.txt"))

        dataset = SevenDataset(opt.data_path, filenames, opt.height, opt.width,
                                opt.frame_ids, 4, is_train=False)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=False)

        pred_disps = []
        pred_poses = []

        print("-> Computing pose predictions")


        with torch.no_grad():
            for inputs in dataloader:
                for key, ipt in inputs.items():
                    inputs[key] = ipt.cuda()


                outputs[("disp", 0)] = inputs["depth_gt"] / 65535
                
                pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)

                fStr1 = "frame-{:06d}.pose.txt".format(inputs["index"].item())
                gtPosePath1 = os.path.join("/content/drive/My Drive/monodepth2/splits/7scenes/chess/seq-01", "poses", fStr1)
                gtPose1 = np.loadtxt(gtPosePath1).reshape(4, 4)
                fStr2 = "frame-{:06d}.pose.txt".format(inputs["index"].item()+opt.frame_ids[1])
                print("fStr1 = {}".format(fStr1))
                print("fStr2 = {}".format(fStr2))
                gtPosePath2 = os.path.join("/content/drive/My Drive/monodepth2/splits/7scenes/chess/seq-01", "poses", fStr2)
                gtPose2 = np.loadtxt(gtPosePath2).reshape(4, 4)
                gtRelativePose = calRelativePose(gtPose1, gtPose2)

                outputs[("cam_T_cam", 0, opt.frame_ids[1])] = torch.from_numpy(gtRelativePose.reshape(1, 4, 4).astype(np.float32)).cuda()
                pred_pose = gtRelativePose
                pred_poses.append(pred_pose)

                self.generate_images_pred(inputs, outputs)
                index = inputs["index"].cpu().item()
                # transLoss, rotLoss = predictionErrorCal(np.squeeze(pred_pose, axis=0), index)
                # transLossAmount += transLoss
                # rotLossAmount += rotLoss
                # if (index + 1) % 50 == 0:
                #     print("now have predict picture index {}".format(index))
                #     print("average error of rot = {}".format(rotLossAmount/(index+1)))
                #     print("average error of trans = {}".format(transLossAmount/(index+1)))
                # picture = outputs[("color", opt.frame_ids[1], 0)].squeeze().cpu().view(480,640,3).numpy()
                img_2 = transforms.ToPILImage()(outputs[("color", opt.frame_ids[1], 0)].squeeze().cpu()).convert('RGB')
                img_2.save("/content/drive/My Drive/monodepth2/generate_gt.jpg") 

        pred_disps = np.concatenate(pred_disps)
        pred_poses = np.concatenate(pred_poses)

        print("-> Predictions saved to")
        print("average error of rot = {}".format(rotLossAmount/999))
        print("average error of trans = {}".format(transLossAmount/999))
        #print(np.squeeze(pred_poses, axis=0).shape)
        


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluation = Evaluation(options.parse())
    evaluation.evaluate()
