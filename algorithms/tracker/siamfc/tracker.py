import torch
from torch import nn
import numpy as np
import cv2


class SiamFCTracker:
    def __init__(self, backbone_z: nn.Module, backbone_x: nn.Module, head: nn.Module, device, params: dict):
        self.backbone_z = backbone_z
        self.backbone_x = backbone_x
        self.head = head
        self.device = device
        self.params = params

        # create hanning window
        self.upscale_sz = params['response_up'] * params['response_sz']

        # search scale factors
        self.scale_factors = params['scale_step'] ** np.linspace(
            -(params['scale_num'] // 2),
            params['scale_num'] // 2, params['scale_num'])

        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()
        self.hann_window = torch.tensor(self.hann_window).to(device)

        from native_extension import RGBImageTranslateAndScale, InterpolationMethod
        self.curation_function = RGBImageTranslateAndScale
        self.interpolation_method = InterpolationMethod.INTER_LINEAR

    def initialize(self, image: np.ndarray, bbox: np.ndarray) -> None:
        '''
        初始化跟踪器，输入待跟踪视频序列的初始图像及物体位置
        Arg
        ---
            image: color image in RGB format
                shape: (H, W, 3)
                type: np.uint8
            bbox: object bounding box, 0-indexed and left-top based
                shape: (4), (X, Y, W, H), index (X, Y) start from 0，should **not** be normalized to [0.0, 1.0]
                type: np.float32
        '''
        bbox = np.array([
            bbox[0] + bbox[2] / 2,
            bbox[1] + bbox[3] / 2,
            bbox[2],
            bbox[3]], dtype=np.float32)
        self.center, self.target_sz = bbox[:2], bbox[2:]

        # exemplar and search sizes
        context = self.params['context'] * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.params['instance_sz'] / self.params['exemplar_sz']

        self.min_z_sz = self.params['min_size_factor'] * self.z_sz
        self.max_z_sz = self.params['max_size_factor'] * self.z_sz
        self.min_x_sz = self.params['min_size_factor'] * self.x_sz
        self.max_x_sz = self.params['max_size_factor'] * self.x_sz

        # exemplar image
        self.channel_avg = np.round(cv2.mean(image)[0:3]).astype(np.uint8)
        z = self._get_z(image, self.channel_avg)

        with torch.no_grad():
            self.kernel = self.backbone_z(z)

    def track(self, image: np.ndarray) -> np.ndarray:
        '''
        输入待跟踪视频序列图像。一般而言，需要按视频帧顺序调用这个函数。
        Arg
        ---
            image: color image in RGB format
                shape: (H, W, 3)
                type: np.uint8
        Return
        ---
            object bounding box, 0-indexed and left-top based
                shape: (4), (X, Y, W, H), index (X, Y) start from 0
                type: np.float32
        '''
        # search images
        x = []
        center = self.center
        out_center = [self.params['instance_sz'] / 2, self.params['instance_sz'] / 2]

        for f in self.scale_factors:
            scale = self._get_scale_from_x(f)
            x11 = self.curation_function(image, [self.params['instance_sz'], self.params['instance_sz']], center, out_center, [scale, scale], self.channel_avg, self.interpolation_method)
            x.append(x11)

        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().float().to(
            self.device)

        scale_id, response_index, response_shape = self._device_inference(self.kernel, x)
        scale_id = scale_id.cpu().item()
        response_index = response_index.cpu().numpy()
        loc = np.unravel_index(response_index, response_shape)
        loc = [loc[1], loc[0]]

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           self.params['total_stride'] / self.params['response_up']
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / self.params['instance_sz']
        self.center += disp_in_image

        # update target size
        scale = (1 - self.params['scale_lr']) * 1.0 + \
                self.params['scale_lr'] * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale
        self.z_sz = np.clip(self.z_sz, self.min_z_sz, self.max_z_sz)
        self.x_sz = np.clip(self.x_sz, self.min_x_sz, self.max_x_sz)

        if self.params['exemplar_lr'] > 0:
            z = self._get_z(image, self.channel_avg)
            with torch.no_grad():
                z = self.backbone_z(z)
            self.kernel = (1 - self.params['exemplar_lr']) * self.kernel + self.params['exemplar_lr'] * z

        # return 1-indexed and left-top based bounding box
        bbox = np.array([
            self.center[0] - self.target_sz[0] / 2,
            self.center[1] - self.target_sz[1] / 2,
            self.target_sz[0], self.target_sz[1]])

        return bbox

    def _get_z(self, image, channel_avg):
        out_center = [self.params['exemplar_sz'] / 2, self.params['exemplar_sz'] / 2]
        scale = self._get_scale_from_z()
        z = self.curation_function(image, [self.params['exemplar_sz'], self.params['exemplar_sz']], self.center, out_center, [scale, scale], channel_avg, self.interpolation_method)
        z = torch.from_numpy(z).permute(2, 0, 1).unsqueeze(0).contiguous().float().to(self.device)
        return z

    def _get_scale_from_x(self, scale=1):
        return self.params['instance_sz'] / (self.x_sz * scale)

    def _get_scale_from_z(self, scale=1):
        return self.params['exemplar_sz'] / (self.z_sz * scale)

    def _device_inference(self, kernel, x):
        with torch.no_grad():
            x = self.backbone_x(x)

        responses = self.head(kernel, x)

        responses = torch.nn.functional.interpolate(responses, (self.upscale_sz, self.upscale_sz), mode='bicubic',
                                                    align_corners=True)
        responses = responses.squeeze(1)

        responses[:self.params['scale_num'] // 2] *= self.params['scale_penalty']
        responses[self.params['scale_num'] // 2 + 1:] *= self.params['scale_penalty']

        # peak scale
        scale_id = torch.argmax(responses) // (self.upscale_sz * self.upscale_sz)

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.params['window_influence']) * response + \
                   self.params['window_influence'] * self.hann_window
        return scale_id, response.argmax(), response.shape
