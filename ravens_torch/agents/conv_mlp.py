# coding=utf-8
# Adapted from Ravens - Transporter Networks, Zeng et al., 2021
# https://github.com/google-research/ravens

"""Convoluational MLP Agent."""

import os
import time

import numpy as np
from ravens_torch.models import mdn_utils
from ravens_torch.models.regression import Regression
from ravens_torch.tasks import cameras
from ravens_torch.utils import utils
import torch
from transforms3d import quaternions


class ConvMlpAgent:
    """Convoluational MLP Agent."""

    def __init__(self, name, task, root_dir, verbose=False):
        self.name = name
        self.task = task
        self.total_iter = 0
        self.pixel_size = 0.003125
        self.input_shape = (320, 160, 6)
        self.camera_config = cameras.RealSenseD415.CONFIG
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

        self.total_iter = 0

        # A place to save pre-trained models.
        self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.batch_size = 4
        self.use_mdn = True
        self.theta_scale = 10.0

    def show_images(self, colormap, heightmap):
        import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
        plt.imshow(colormap)
        plt.show()
        plt.imshow(heightmap)
        plt.show()

    def extract_x_y_theta(self,
                          object_info,
                          t_worldaug_world=None,
                          preserve_theta=False):
        """Extract in-plane theta."""
        object_position = object_info[0]
        object_quat_xyzw = object_info[1]

        if t_worldaug_world is not None:
            object_quat_wxyz = (object_quat_xyzw[3], object_quat_xyzw[0],
                                object_quat_xyzw[1], object_quat_xyzw[2])
            t_world_object = quaternions.quat2mat(object_quat_wxyz)
            t_world_object[0:3, 3] = np.array(object_position)
            t_worldaug_object = t_worldaug_world @ t_world_object

            object_quat_wxyz = quaternions.mat2quat(t_worldaug_object)
            if not preserve_theta:
                object_quat_xyzw = (object_quat_wxyz[1], object_quat_wxyz[2],
                                    object_quat_wxyz[3], object_quat_wxyz[0])
            object_position = t_worldaug_object[0:3, 3]

        object_xy = object_position[0:2]
        object_theta = -np.float32(
            utils.quatXYZW_to_eulerXYZ(object_quat_xyzw)
            [2]) / self.theta_scale
        return np.hstack(
            (object_xy,
             object_theta)).astype(np.float32), object_position, object_quat_xyzw

    def act_to_gt_act(self, act, t_worldaug_world=None):
        # dont update theta due to suction invariance to theta
        pick_se2, _, _ = self.extract_x_y_theta(
            act['params']['pose0'], t_worldaug_world, preserve_theta=True)
        place_se2, _, _ = self.extract_x_y_theta(
            act['params']['pose1'], t_worldaug_world, preserve_theta=True)
        return np.hstack((pick_se2, place_se2)).astype(np.float32)

    def get_data_batch(self, dataset, augment=True):
        """Sample batch."""

        batch_obs = []
        batch_act = []

        for _ in range(self.batch_size):
            obs, act, _ = dataset.random_sample()

            # Get heightmap from RGB-D images.
            configs = act['camera_config']
            colormap, heightmap = self.get_heightmap(obs, configs)
            # self.show_images(colormap, heightmap)

            # Concatenate color with depth images.
            input_image = np.concatenate((colormap, heightmap[Ellipsis, None],
                                          heightmap[Ellipsis, None], heightmap[Ellipsis, None]),
                                         axis=2)

            # or just use rgb
            # input_image = colormap

            # Apply augmentation
            if augment:
                # note: these pixels are made up,
                # just to keep the perturb function happy.
                p0 = (160, 80)
                p1 = (160, 80)
                input_image, _, _, transform_params = utils.perturb(
                    input_image, [p0, p1], set_theta_zero=False)
                t_world_center, t_world_centeraug = utils.get_se3_from_image_transform(
                    *transform_params, heightmap, self.bounds, self.pixel_size)
                t_worldaug_world = t_world_centeraug @ np.linalg.inv(
                    t_world_center)
            else:
                t_worldaug_world = np.eye(4)

            batch_obs.append(input_image)
            batch_act.append(self.act_to_gt_act(
                act, t_worldaug_world))  # this samples pick points from surface

        batch_obs = np.array(batch_obs)
        batch_act = np.array(batch_act)
        return batch_obs, batch_act

    def train(self, dataset, num_iter, writer, validation_dataset):
        """Train on dataset for a specific number of iterations."""

        validation_rate = 100

        def pick_train_step(model, optim, in_tensor, yxtheta, loss_criterion):
            optim.zero_grad()

            output = model(in_tensor)
            loss = loss_criterion(yxtheta, output)

            loss.backward()
            optim.step()
            return loss

        def pick_valid_step(model, optim, in_tensor, yxtheta, loss_criterion):
            del optim

            with torch.no_grad():
                output = model(in_tensor)
                loss = loss_criterion(yxtheta, output)
            return loss

        for i in range(num_iter):
            start = time.time()

            batch_obs, batch_act = self.get_data_batch(dataset)

            # Compute train loss
            loss0 = self.regression_model.train_pick(batch_obs, batch_act,
                                                     pick_train_step)
            writer.add_scalar('train/pick_loss',
                              self.regression_model.metric.result(),
                              self.total_iter + i)

            print(f'Train Iter: {self.total_iter + i} Loss: {loss0:.4f} Iter time:',
                  time.time() - start)

            if (self.total_iter + i) % validation_rate == 0:
                print('Validating!')
                self.regression_model.model_query.eval()
                self.regression_model.model_key.eval()
                batch_obs, batch_act = self.get_data_batch(
                    validation_dataset, augment=False)

                # Compute valid loss
                loss0 = self.regression_model.train_pick(
                    batch_obs, batch_act, pick_valid_step, validate=True)
                writer.add_scalar('validation/pick_loss',
                                  self.regression_model.val_metric.result(),
                                  self.total_iter + i)

                self.regression_model.model_query.train()
                self.regression_model.model_key.train()

        self.total_iter += num_iter
        self.save()

    def act(self, obs, gt_act, info):
        """Run inference and return best action given visual observations."""

        del gt_act
        del info

        self.regression_model.set_batch_size(1)

        act = {'camera_config': self.camera_config, 'primitive': None}
        if not obs:
            return act

        # Get heightmap from RGB-D images.
        colormap, heightmap = self.get_heightmap(obs, self.camera_config)

        # Concatenate color with depth images.
        input_image = np.concatenate(
            (colormap, heightmap[Ellipsis, None], heightmap[Ellipsis, None], heightmap[Ellipsis,
                                                                                       None]),
            axis=2)[None, Ellipsis]

        # or just use rgb
        # input_image = colormap[None, ...]

        # Regression
        prediction = self.regression_model.forward(input_image)

        if self.use_mdn:
            mdn_prediction = prediction
            pi, mu, var = mdn_prediction
            # prediction = mdn_utils.pick_max_mean(pi, mu, var)
            prediction = mdn_utils.sample_from_pdf(pi, mu, var)
            prediction = prediction[:, 0, :]

        prediction = prediction[0]

        p0_position = np.hstack((prediction[0:2], 0.02))
        p1_position = np.hstack((prediction[3:5], 0.02))

        p0_rotation = utils.eulerXYZ_to_quatXYZW(
            (0, 0, -prediction[2] * self.theta_scale))
        p1_rotation = utils.eulerXYZ_to_quatXYZW(
            (0, 0, -prediction[5] * self.theta_scale))

        act['primitive'] = 'pick_place'
        if self.task == 'sweeping':
            act['primitive'] = 'sweep'
        elif self.task == 'pushing':
            act['primitive'] = 'push'
        params = {
            'pose0': (np.asarray(p0_position), np.asarray(p0_rotation)),
            'pose1': (np.asarray(p1_position), np.asarray(p1_rotation))
        }
        act['params'] = params
        self.regression_model.set_batch_size(self.batch_size)

        return act

    # -------------------------------------------------------------------------
    # Helper Functions
    # -------------------------------------------------------------------------

    def preprocess(self, image):
        """Pre-process images (subtract mean, divide by std).

        Args:
          image: shape: [B, H, W, C]

        Returns:
          preprocessed image.
        """
        color_mean = 0.18877631
        depth_mean = 0.00509261
        color_std = 0.07276466
        depth_std = 0.00903967

        del depth_mean
        del depth_std

        image[:, :, :, :3] = (image[:, :, :, :3] /
                              255 - color_mean) / color_std
        # image[:, :, :, 3:] = (image[:, :, :, 3:] - depth_mean) / depth_std
        return image

    def get_heightmap(self, obs, configs):
        """Reconstruct orthographic heightmaps with segmentation masks."""
        heightmaps, colormaps = utils.reconstruct_heightmaps(
            obs['color'], obs['depth'], configs, self.bounds, self.pixel_size)
        colormaps = np.float32(colormaps)
        heightmaps = np.float32(heightmaps)

        # Fuse maps from different views.
        valid = np.sum(colormaps, axis=3) > 0
        repeat = np.sum(valid, axis=0)
        repeat[repeat == 0] = 1
        colormap = np.sum(colormaps, axis=0) / repeat[Ellipsis, None]
        colormap = np.uint8(np.round(colormap))
        heightmap = np.max(heightmaps, axis=0)
        return colormap, heightmap

    def load(self, num_iter):
        pass

    def save(self):
        pass


class PickPlaceConvMlpAgent(ConvMlpAgent):

    def __init__(self, name, task, root_dir, verbose=False):
        super().__init__(name, task, root_dir)

        self.regression_model = Regression(
            in_channels=self.input_shape[2],
            preprocess=self.preprocess,
            use_mdn=self.use_mdn,
            verbose=verbose)
        self.regression_model.set_batch_size(self.batch_size)
