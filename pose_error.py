from __future__ import absolute_import, division, print_function

import math
import numpy as np
import cv2
import os

def calRelativePose(poseMatrix1, poseMatrix2):
    """Calculate the relative pose FROM matrix1 To matrix2 using in 7scenes
    # Return Value: relativeMatrix
    """
    assert (poseMatrix1.shape == (4, 4)), "shape of matrix 1 should be (4,4)"
    assert (poseMatrix2.shape == (4, 4)), "shape of matrix 1 should be (4,4)"

    rotMatrix1 = poseMatrix1[0:3, 0:3]
    rotMatrix2 = poseMatrix2[0:3, 0:3]
    transVec1 = poseMatrix1[0:3, 3]
    transVec2 = poseMatrix2[0:3, 3]

    poseMatrix2_inv = np.zeros((4, 4), dtype=np.float64)
    # [R|t]' = [R^T|-R^Tt]
    poseMatrix2_inv[0:3, 0:3] = rotMatrix2.T
    poseMatrix2_inv[0:3, 3]  = - np.matmul(rotMatrix2.T, transVec2)
    poseMatrix2_inv[3, 3]   = 1

    relativeMatrix = np.matmul(poseMatrix2_inv, poseMatrix1)
    return relativeMatrix

def calPoseError(gtRelativePose, predictionPose):
    """Calculate the loss between ground-truth pose and prediction pose
    # Return Value: transLoss, rotLoss 
    """
    assert (gtRelativePose.shape == (4, 4)), "shape of ground truth pose matrix should be (4,4)"
    assert (predictionPose.shape == (4, 4)), "shape of prediction pose matrix should be (4,4)"
    
    gtPoseRot = gtRelativePose[0:3, 0:3]
    predictionPoseRot = predictionPose[0:3, 0:3]

    R = gtPoseRot.T @ predictionPoseRot
    transLoss = np.linalg.norm(gtRelativePose[0:3, 3] - predictionPose[0:3, 3])
    rotLoss = np.linalg.norm(cv2.Rodrigues(R)[0])

    return transLoss, rotLoss


def predictionErrorCal(prediction, index):
    """Calculate the loss of a prediction, this function will read the ground truth data
    from the disk, then calculate the final loss.
    # Return Value: transLoss, rotLoss 
    """
    assert (prediction.shape == (4, 4)), "shape of prediction pose matrix should be (4,4)"

    fStr1 = "frame-{:06d}.pose.txt".format(index)
    gtPosePath1 = os.path.join("/data1/ctsai/7scenes/chess/seq-01", fStr1)
    gtPose1 = np.loadtxt(gtPosePath1).reshape(4, 4)
    fStr2 = "frame-{:06d}.pose.txt".format(index+1)
    gtPosePath2 = os.path.join("/data1/ctsai/7scenes/chess/seq-01", fStr2)
    gtPose2 = np.loadtxt(gtPosePath2).reshape(4, 4)
    gtRelativePose = calRelativePose(gtPose1, gtPose2)

    transLoss, rotLoss = calPoseError(gtRelativePose, prediction)

    return transLoss, rotLoss






