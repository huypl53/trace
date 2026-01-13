# -*- coding: utf-8 -*- #

import math
import os
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.utils.data as data

import file_utils
import imgproc
from parsers.line_parser import ParserLine
from parsers.xml_parser import ParserTRACE  # ours

gaussian_map = np.zeros((201, 201), dtype=np.float32)
gaussian_line = np.zeros((201, 201), dtype=np.float32)
sigma = math.pow(40.0, 2)
for i in range(201):
    for j in range(201):
        gaussian_map[i, j] = math.exp(-(math.pow(i - 100, 2) / 2 / sigma + math.pow(j - 100, 2) / 2 / sigma))
        gaussian_line[i, j] = math.exp(-(math.pow(i - 100, 2) / 2 / sigma))
gaussian_map /= np.max(gaussian_map)
gaussian_line /= np.max(gaussian_line)
gaussian_poly = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])


def get_heatmap_patch(quad, im_size, kernel, poly=gaussian_poly):
    bbox = np.int32([np.floor(quad.min(axis=0)), np.ceil(quad.max(axis=0))])
    if any(bbox[0] > im_size) or any(bbox[1] < 0):
        return False, None, None
    bbox = np.int32([np.maximum(bbox[0], 0), np.minimum(bbox[1], im_size)])
    if any(bbox[0] >= bbox[1]):
        return False, None, None
    patch_size = tuple(bbox[1] - bbox[0])
    tl = np.minimum(np.maximum(np.floor(quad.min(axis=0)), 0), im_size).astype(np.float32)
    M = cv2.getPerspectiveTransform(poly, quad - tl)
    img_text = cv2.warpPerspective(kernel, M, patch_size)
    return True, img_text, bbox


def GTTransform(target, width, height):
    effective_conf = 0.05
    height = int(height)
    width = int(width)
    heatmap_gt = np.zeros((height, width), dtype=np.float32)
    heatmap_gt_hor = np.zeros((height, width), dtype=np.float32)
    heatmap_gt_ver = np.zeros((height, width), dtype=np.float32)
    heatmap_gt_ihor = np.zeros((height, width), dtype=np.float32)
    heatmap_gt_iver = np.zeros((height, width), dtype=np.float32)
    weight_mask = np.ones((height, width), dtype=np.float32)

    for k, obj in enumerate(target):
        attributes = obj.copy()
        obj, lines = obj["quad"], obj["line"]

        obj = np.array(obj)
        obj *= [width, height] * 4

        obj_poly = obj[:8].astype(np.float32).reshape((4, 2))
        # constant
        bs = 5

        for i in range(4):
            # calc center of box
            center = obj_poly[i]

            # warp gaussian image to magnified character box
            corner_poly = np.float32(
                [
                    [center[0] - bs, center[1] - bs],
                    [center[0] + bs, center[1] - bs],
                    [center[0] + bs, center[1] + bs],
                    [center[0] - bs, center[1] + bs],
                ]
            )
            M = cv2.getPerspectiveTransform(gaussian_poly, corner_poly)
            img_text = cv2.warpPerspective(gaussian_map, M, (width, height))
            heatmap_gt = np.maximum(heatmap_gt, img_text)

        # link representation
        thickness = 2
        # horizontal line
        if lines[0]:  # TOP
            cv2.line(
                heatmap_gt_hor,
                tuple(obj_poly[0].astype(np.int32)),
                tuple(obj_poly[1].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        else:
            cv2.line(
                heatmap_gt_ihor,
                tuple(obj_poly[0].astype(np.int32)),
                tuple(obj_poly[1].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        if lines[1]:  # BOTTOM
            cv2.line(
                heatmap_gt_hor,
                tuple(obj_poly[2].astype(np.int32)),
                tuple(obj_poly[3].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        else:
            cv2.line(
                heatmap_gt_ihor,
                tuple(obj_poly[2].astype(np.int32)),
                tuple(obj_poly[3].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        # vertical line
        if lines[3]:  # RIGHT
            cv2.line(
                heatmap_gt_ver,
                tuple(obj_poly[1].astype(np.int32)),
                tuple(obj_poly[2].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        else:
            cv2.line(
                heatmap_gt_iver,
                tuple(obj_poly[1].astype(np.int32)),
                tuple(obj_poly[2].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        if lines[2]:  # LEFT
            cv2.line(
                heatmap_gt_ver,
                tuple(obj_poly[3].astype(np.int32)),
                tuple(obj_poly[0].astype(np.int32)),
                color=1,
                thickness=thickness,
            )
        else:
            cv2.line(
                heatmap_gt_iver,
                tuple(obj_poly[3].astype(np.int32)),
                tuple(obj_poly[0].astype(np.int32)),
                color=1,
                thickness=thickness,
            )

    # clipping in heatmap
    heatmap_gt[heatmap_gt < effective_conf] = 0

    # finalize gt
    heatmap_gt = np.concatenate(
        [
            heatmap_gt[..., np.newaxis],
            heatmap_gt_hor[..., np.newaxis],
            heatmap_gt_ver[..., np.newaxis],
        ],
        axis=-1,
    )
    heatmap_gt = np.concatenate(
        [
            heatmap_gt,
            heatmap_gt_ihor[..., np.newaxis],
            heatmap_gt_iver[..., np.newaxis],
        ],
        axis=-1,
    )

    return heatmap_gt, weight_mask


def LineGTTransform(target, width, height, line_thickness=3, use_gaussian=True):
    height = int(height)
    width = int(width)
    heatmap_h = np.zeros((height, width), dtype=np.float32)
    heatmap_v = np.zeros((height, width), dtype=np.float32)
    weight_mask = np.ones((height, width), dtype=np.float32)

    for start, end in target.get("h_lines", []):
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        cv2.line(heatmap_h, start, end, color=1.0, thickness=line_thickness)

    for start, end in target.get("v_lines", []):
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        cv2.line(heatmap_v, start, end, color=1.0, thickness=line_thickness)

    if use_gaussian:
        heatmap_h = cv2.GaussianBlur(heatmap_h, (5, 5), 0)
        heatmap_v = cv2.GaussianBlur(heatmap_v, (5, 5), 0)
        if heatmap_h.max() > 0:
            heatmap_h /= heatmap_h.max()
        if heatmap_v.max() > 0:
            heatmap_v /= heatmap_v.max()

    heatmap = np.stack([heatmap_h, heatmap_v], axis=-1)
    return heatmap, weight_mask


def normalize_mask(mask):
    mask = mask.astype(np.float32)
    max_val = mask.max() if mask.size > 0 else 0.0
    if max_val > 0:
        mask /= max_val
    return mask


class TRACE_Dataset(data.Dataset):
    """OCR Dataset Object for TextAffinityField

    input is image, target is annotation

    Arguments:
        rootpath (string): filepath to OCR folder
        datasets (string): datasets name (paths to dataset are already defined in db_params.py)
        phase (string): set 'test' or 'train' phase (default is 'train')
        transform (callable, optional): transformation to perform on the input image
        gt_transform (callable, optional): transformation to perform on the GT annotation
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
            parser = ParserTRACE(rootpath, dataset, phase)
            self.dataset_size += parser.lenFiles()
            self.parsers.append({"name": dataset, "num": parser.lenFiles(), "parser": parser})
        print("Dataset size of Training Sets : {:d}".format(self.dataset_size))

        cv2.setNumThreads(0)  # prevent deadlock caused by conflict with pytorch

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
                    img = img_file  # Some parser return image rather than image file name
                height, width, channels = img.shape
            except Exception as e:
                print(e)
                continue

            if gt is None:
                gt = []

            gt, gt_lines = gt["quads"], gt["lines"]
            gt = np.array(gt, dtype=np.float32).reshape(-1, 8)
            break

        # normalize GT coordinate
        num_pt = int(gt.shape[1] / 2)
        gt[:, : 2 * num_pt] /= [width, height] * num_pt

        # Transformation
        if self.transform is not None:
            img, gt, gt_lines, _ = self.transform(img, gt, gt_lines)
            width = height = self.transform.size

        # Create TextAffinityField GT Map
        gt_gathered = []
        for attr, quad in zip(gt_lines, gt):
            gt_gathered.append({"quad": quad, "line": attr})

        # GT transform
        width /= self.scale_down
        height /= self.scale_down
        gt_image, gt_weight = GTTransform(gt_gathered, width, height)
        _, _, gt_ch = gt_image.shape
        gt_weight = np.array([gt_weight] * gt_ch).transpose(1, 2, 0)

        # Preprocessing for pre-trained model
        img = imgproc.normalizeMeanVariance(img)

        return (
            torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1),
            torch.from_numpy(gt_image.astype(np.float32)),
            torch.from_numpy(gt_weight.astype(np.float32)),
        )


class Line_Dataset(data.Dataset):
    """Dataset for line segmentation training."""

    def __init__(
        self,
        datasets,
        rootpath,
        scale_down=2,
        phase="train",
        transform=None,
        mixratio=[1],
        line_thickness=3,
        use_gaussian=True,
    ):
        self.rootpath = rootpath
        self.datasets = datasets
        self.mixratio = mixratio
        self.scale_down = scale_down
        self.phase = phase
        self.transform = transform
        self.line_thickness = line_thickness
        self.use_gaussian = use_gaussian
        self.parsers = []
        self.dataset_size = 0

        for dataset in datasets.split(","):
            parser = ParserLine(rootpath, dataset, phase)
            self.dataset_size += parser.lenFiles()
            self.parsers.append({"name": dataset, "num": parser.lenFiles(), "parser": parser})
        print("Dataset size of Training Sets : {:d}".format(self.dataset_size))

        cv2.setNumThreads(0)  # prevent deadlock caused by conflict with pytorch

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
                    img = img_file
                height, width, _ = img.shape
            except Exception as e:
                print(e)
                continue

            if gt is None:
                gt = {"h_lines": [], "v_lines": []}
            break

        h_lines_norm = []
        for start, end in gt["h_lines"]:
            h_lines_norm.append(((start[0] / width, start[1] / height), (end[0] / width, end[1] / height)))
        v_lines_norm = []
        for start, end in gt["v_lines"]:
            v_lines_norm.append(((start[0] / width, start[1] / height), (end[0] / width, end[1] / height)))

        gt_norm = {"h_lines": h_lines_norm, "v_lines": v_lines_norm}

        if self.transform is not None:
            img, gt_norm = self.transform(img, gt_norm)
            width = height = self.transform.size

        out_w = int(width / self.scale_down)
        out_h = int(height / self.scale_down)

        h_lines_out = [
            ((int(s[0] * out_w), int(s[1] * out_h)), (int(e[0] * out_w), int(e[1] * out_h)))
            for s, e in gt_norm["h_lines"]
        ]
        v_lines_out = [
            ((int(s[0] * out_w), int(s[1] * out_h)), (int(e[0] * out_w), int(e[1] * out_h)))
            for s, e in gt_norm["v_lines"]
        ]

        gt_out = {"h_lines": h_lines_out, "v_lines": v_lines_out}
        gt_image, gt_weight = LineGTTransform(
            gt_out,
            out_w,
            out_h,
            line_thickness=self.line_thickness,
            use_gaussian=self.use_gaussian,
        )
        _, _, gt_ch = gt_image.shape
        gt_weight = np.array([gt_weight] * gt_ch).transpose(1, 2, 0)

        img = imgproc.normalizeMeanVariance(img)

        return (
            torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1),
            torch.from_numpy(gt_image.astype(np.float32)),
            torch.from_numpy(gt_weight.astype(np.float32)),
        )


class LineMask_Dataset(data.Dataset):
    """Dataset for line segmentation training using mask images."""

    def __init__(
        self,
        datasets,
        rootpath,
        scale_down=2,
        phase="train",
        transform=None,
        mixratio=[1],
        mask_suffix_h="_mask_h.png",
        mask_suffix_v="_mask_v.png",
    ):
        self.rootpath = rootpath
        self.datasets = datasets
        self.mixratio = mixratio
        self.scale_down = scale_down
        self.phase = phase
        self.transform = transform
        self.mask_suffix_h = mask_suffix_h
        self.mask_suffix_v = mask_suffix_v
        self.parsers = []
        self.dataset_size = 0

        for dataset in datasets.split(","):
            dataset = dataset.strip()
            if not dataset:
                continue
            samples = self._collect_samples(dataset, phase)
            self.dataset_size += len(samples)
            self.parsers.append({"name": dataset, "num": len(samples), "samples": samples})
        if self.dataset_size == 0:
            raise ValueError("No line mask samples found for {}".format(datasets))
        print("Dataset size of Training Sets : {:d}".format(self.dataset_size))

        cv2.setNumThreads(0)  # prevent deadlock caused by conflict with pytorch

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        img, gt, weight = self.pull_item(index)
        return img, gt, weight

    def _collect_samples(self, dataset, phase):
        base_candidates = [
            os.path.join(self.rootpath, dataset, phase),
            os.path.join(self.rootpath, dataset),
        ]
        images_dir = None
        masks_dir = None
        for base_dir in base_candidates:
            candidate_images = os.path.join(base_dir, "images")
            candidate_masks = os.path.join(base_dir, "masks")
            if os.path.isdir(candidate_images) and os.path.isdir(candidate_masks):
                images_dir = candidate_images
                masks_dir = candidate_masks
                break
        if images_dir is None or masks_dir is None:
            raise ValueError(
                "Expected images/ and masks/ under {}".format(" or ".join(base_candidates))
            )

        image_files = sorted(file_utils.get_image_list(images_dir))
        samples = []
        for img_path in image_files:
            rel_path = os.path.relpath(img_path, images_dir)
            stem = os.path.splitext(rel_path)[0]
            mask_h_path = os.path.join(masks_dir, stem + self.mask_suffix_h)
            mask_v_path = os.path.join(masks_dir, stem + self.mask_suffix_v)
            if not os.path.exists(mask_h_path) or not os.path.exists(mask_v_path):
                print("No mask files found for {}".format(img_path))
                continue
            samples.append((img_path, mask_h_path, mask_v_path))
        if not samples:
            raise ValueError("No mask pairs found in {}".format(images_dir))
        return samples

    def _resize_mask(self, mask, width, height):
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask

    def _downscale_mask(self, mask, width, height):
        if mask.ndim == 2:
            return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        channels = []
        for ch in range(mask.shape[2]):
            channels.append(cv2.resize(mask[:, :, ch], (width, height), interpolation=cv2.INTER_NEAREST))
        return np.stack(channels, axis=-1)

    def pull_item(self, index):
        mix_rand = random.randrange(0, sum(self.mixratio))
        parser_ind = 0
        parser_ind_sum = self.mixratio[0]
        while 1:
            if mix_rand < parser_ind_sum:
                break
            parser_ind += 1
            parser_ind_sum += self.mixratio[parser_ind]
        parser = self.parsers[parser_ind]

        while 1:
            try:
                img_path, mask_h_path, mask_v_path = parser["samples"][
                    random.randrange(0, parser["num"])
                ]
                img = imgproc.loadImage(img_path)
                mask_h = cv2.imread(mask_h_path, cv2.IMREAD_GRAYSCALE)
                mask_v = cv2.imread(mask_v_path, cv2.IMREAD_GRAYSCALE)
                if mask_h is None or mask_v is None:
                    raise ValueError("Failed to read mask for {}".format(img_path))
                height, width, _ = img.shape
                mask_h = self._resize_mask(mask_h, width, height)
                mask_v = self._resize_mask(mask_v, width, height)
                mask = np.stack([mask_h, mask_v], axis=-1)
            except Exception as e:
                print(e)
                continue
            break

        if self.transform is not None:
            img, mask = self.transform(img, mask)
            width = height = self.transform.size

        out_w = int(width / self.scale_down)
        out_h = int(height / self.scale_down)
        mask = self._downscale_mask(mask, out_w, out_h)

        mask = mask.astype(np.float32)
        for ch in range(mask.shape[2]):
            mask[:, :, ch] = normalize_mask(mask[:, :, ch])
        gt_weight = np.ones_like(mask, dtype=np.float32)

        img = imgproc.normalizeMeanVariance(img)

        return (
            torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1),
            torch.from_numpy(mask.astype(np.float32)),
            torch.from_numpy(gt_weight.astype(np.float32)),
        )
