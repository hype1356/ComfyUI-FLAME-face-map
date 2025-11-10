"""
FLAME Layer: Implementation of the 3D Statistical Face model in PyTorch

It is designed in a way to directly plug in as a decoder layer in a
Deep learning framework for training and testing

It can also be used for 2D or 3D optimisation applications

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""
# Modified from smplx code [https://github.com/vchoutas/smplx] for FLAME

import pickle

import numpy as np
import torch
import torch.nn as nn
from smplx.lbs import lbs
from smplx.utils import Struct, to_np, to_tensor


class FLAME(nn.Module):
    """Given FLAME parameters this class outputs the corresponding mesh vertices."""

    def __init__(self, config):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(config.flame_model_path, "rb") as f:
            self.flame_model = Struct(**pickle.load(f, encoding="latin1"))
        self.batch_size = config.batch_size
        self.dtype = torch.float32
        self.faces = self.flame_model.f

        # Fixing remaining Shape betas
        # There are total 300 shape parameters to control FLAME; But one can use the first few parameters to express
        # the shape. For example 100 shape parameters are used for RingNet project
        default_shape = torch.zeros(
            [self.batch_size, 300 - config.shape_params],
            dtype=self.dtype,
            requires_grad=False,
        )
        self.register_parameter(
            "shape_betas", nn.Parameter(default_shape, requires_grad=False)
        )

        # Fixing remaining expression betas
        # There are total 100 shape expression parameters to control FLAME; But one can use the first few parameters to express
        # the expression. For example 50 expression parameters are used for RingNet project
        default_exp = torch.zeros(
            [self.batch_size, 100 - config.expression_params],
            dtype=self.dtype,
            requires_grad=False,
        )
        self.register_parameter(
            "expression_betas", nn.Parameter(default_exp, requires_grad=False)
        )

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros(
            [self.batch_size, 6], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter(
            "eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False)
        )

        default_neck_pose = torch.zeros(
            [self.batch_size, 3], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter(
            "neck_pose", nn.Parameter(default_neck_pose, requires_grad=False)
        )

        # Fixing 3D translation since we use translation in the image plane

        self.use_3D_translation = config.use_3D_translation

        default_transl = torch.zeros(
            [self.batch_size, 3], dtype=self.dtype, requires_grad=False
        )
        self.register_parameter(
            "transl", nn.Parameter(default_transl, requires_grad=False)
        )

        # The vertices of the template model
        self.register_buffer(
            "v_template",
            to_tensor(to_np(self.flame_model.v_template), dtype=self.dtype),
        )

        # The shape components
        shapedirs = self.flame_model.shapedirs
        # The shape components
        self.register_buffer("shapedirs", to_tensor(to_np(shapedirs), dtype=self.dtype))

        j_regressor = to_tensor(to_np(self.flame_model.J_regressor), dtype=self.dtype)
        self.register_buffer("J_regressor", j_regressor)

        # Pose blend shape basis
        num_pose_basis = self.flame_model.posedirs.shape[-1]
        posedirs = np.reshape(self.flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))

        # indices of parents for each joints
        parents = to_tensor(to_np(self.flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)

        self.register_buffer(
            "lbs_weights", to_tensor(to_np(self.flame_model.weights), dtype=self.dtype)
        )

    def forward(
        self,
        shape_params=None,
        expression_params=None,
        pose_params=None,
        neck_pose=None,
        eye_pose=None,
        transl=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters
        return:
            vertices: N X V X 3
        """
        betas = torch.cat(
            [shape_params, self.shape_betas, expression_params, self.expression_betas],
            dim=1,
        )
        neck_pose = neck_pose if neck_pose is not None else self.neck_pose
        eye_pose = eye_pose if eye_pose is not None else self.eye_pose
        transl = transl if transl is not None else self.transl
        full_pose = torch.cat(
            [pose_params[:, :3], neck_pose, pose_params[:, 3:], eye_pose], dim=1
        )
        template_vertices = self.v_template.unsqueeze(0).repeat(self.batch_size, 1, 1)

        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )

        # Always apply translation to vertices when requested
        if self.use_3D_translation:
            vertices = vertices + transl.unsqueeze(dim=1)

        return vertices
