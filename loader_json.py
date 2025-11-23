# -*- coding: utf-8 -*- #

import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data

import imgproc
# from parsers.json_parser import ParserTRACEJSON
from parsers.npy_parser import ParserTRACENPY as Parser


class TRACE_Dataset_npy(data.Dataset):
    """OCR Dataset Object for TextAffinityField using JSON labels and pre-generated masks

    input is image, target is pre-generated mask annotation

    Arguments:
        rootpath (string): filepath to OCR folder
        datasets (string): datasets name (paths to dataset are already defined in db_params.py)
        phase (string): set 'test' or 'train' phase (default is 'train')
        transform (callable, optional): transformation to perform on the input image
        scale_down (int): scale down factor for masks (default is 2)
    """

    def __init__(
        self,
        datasets,
        rootpath,
        scale_down=2,
        out_type="HEATMAP",
        phase="train",
        transform=None,
        mixratio=[1],
    ):
        self.rootpath = rootpath
        self.datasets = datasets
        self.mixratio = mixratio
        self.scale_down = scale_down
        self.out_type = out_type
        self.phase = phase
        self.transform = transform
        self.parsers = []
        self.dataset_size = 0

        for dataset in datasets.split(","):
            parser = Parser(rootpath, dataset, phase)
            self.dataset_size += parser.lenFiles()
            self.parsers.append(
                {"name": dataset, "num": parser.lenFiles(), "parser": parser}
            )
        print("Dataset size of Training Sets : {:d}".format(self.dataset_size))

        cv2.setNumThreads(0)  # prevent deadlock caused by conflict with pytorch

    @staticmethod
    def _resize_tensor(
        tensor, target_width, target_height, interpolation=cv2.INTER_LINEAR
    ):
        if tensor is None:
            return None
        if tensor.ndim == 2:
            return cv2.resize(
                tensor, (target_width, target_height), interpolation=interpolation
            )

        channels = tensor.shape[2]
        resized = np.zeros((target_height, target_width, channels), dtype=tensor.dtype)
        for ch in range(channels):
            resized[:, :, ch] = cv2.resize(
                tensor[:, :, ch],
                (target_width, target_height),
                interpolation=interpolation,
            )
        return resized

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        img, gt, weight = self.pull_item(index)
        return img, gt, weight

    def pull_item(self, index):
        mix_rand = random.randrange(0, sum(self.mixratio))
        parser_ind = 0
        parser_ind_sum = self.mixratio[0]
        while 1:
            if mix_rand < parser_ind_sum:
                break
            parser_ind += 1
            parser_ind_sum += self.mixratio[parser_ind]
        parser = self.parsers[parser_ind]["parser"]

        while 1:
            try:
                img_file, gt = parser.parseGT()
                if isinstance(img_file, str):
                    img = imgproc.loadImage(img_file)
                else:
                    img = (
                        img_file  # Some parser return image rather than image file name
                    )
                height, width, channels = img.shape

                # Load pre-generated mask
                basename, ext = os.path.splitext(os.path.basename(img_file))
                base_folder = os.path.dirname(img_file)
                mask_file = os.path.join(base_folder, f"{basename}.npy")

                if not os.path.exists(mask_file):
                    raise FileNotFoundError(f"Mask file not found: {mask_file}")

                mask_data = np.load(mask_file, allow_pickle=True).item()
                gt_image = mask_data["mask"]
                gt_weight = mask_data["weight"]

            except Exception as e:
                print(e)
                continue

            break

        # Transformation
        if self.transform is not None:
            mask_full_res = self._resize_tensor(
                gt_image, width, height, interpolation=cv2.INTER_LINEAR
            )
            weight_full_res = gt_weight
            if weight_full_res.ndim == 2:
                weight_full_res = np.expand_dims(weight_full_res, axis=2)
            weight_full_res = self._resize_tensor(
                weight_full_res, width, height, interpolation=cv2.INTER_LINEAR
            )

            img, mask_full_res, weight_full_res = self.transform(
                img, mask_full_res, weight_full_res
            )
            width = height = self.transform.size

            target_mask_height = int(height / self.scale_down)
            target_mask_width = int(width / self.scale_down)
            gt_image = self._resize_tensor(
                mask_full_res,
                target_mask_width,
                target_mask_height,
                interpolation=cv2.INTER_LINEAR,
            )
            gt_weight = self._resize_tensor(
                weight_full_res,
                target_mask_width,
                target_mask_height,
                interpolation=cv2.INTER_LINEAR,
            )
            if gt_weight.ndim == 3 and gt_weight.shape[2] == 1:
                gt_weight = gt_weight[:, :, 0]
        else:
            target_mask_height = int(height / self.scale_down)
            target_mask_width = int(width / self.scale_down)

            if gt_image.shape[:2] != (target_mask_height, target_mask_width):
                gt_image = self._resize_tensor(
                    gt_image,
                    target_mask_width,
                    target_mask_height,
                    interpolation=cv2.INTER_LINEAR,
                )

            if gt_weight.shape[:2] != (target_mask_height, target_mask_width):
                gt_weight = cv2.resize(
                    gt_weight,
                    (target_mask_width, target_mask_height),
                    interpolation=cv2.INTER_LINEAR,
                )

        # Expand weight to match mask channels
        _, _, gt_ch = gt_image.shape
        if gt_weight.ndim == 2:
            gt_weight = np.array([gt_weight] * gt_ch).transpose(1, 2, 0)
        elif gt_weight.ndim == 3 and gt_weight.shape[2] == 1:
            gt_weight = np.repeat(gt_weight, gt_ch, axis=2)

        # Preprocessing for pre-trained model
        img = imgproc.normalizeMeanVariance(img)

        return (
            torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1),
            torch.from_numpy(gt_image.astype(np.float32)),
            torch.from_numpy(gt_weight.astype(np.float32)),
        )
