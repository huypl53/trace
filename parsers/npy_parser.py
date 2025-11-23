# -*- coding: utf-8 -*-
import os
import random

import file_utils


class ParserTRACENPY:
    """Parser that pairs images with pre-generated .npy mask files."""

    def __init__(self, root_path, dataset, phase):
        self.gt = []

        base_folder = os.path.join(root_path, dataset)
        phase_folder = os.path.join(base_folder, phase) if phase else None

        if phase_folder and os.path.exists(phase_folder):
            base_folder = phase_folder

        image_files, _, _ = file_utils.list_files(base_folder)

        for img_file in image_files:
            dirname = os.path.dirname(img_file)
            basename, _ = os.path.splitext(os.path.basename(img_file))

            candidate_masks = [
                os.path.join(dirname, f"{basename}.npy"),
                os.path.join(dirname, f"{basename}_mask.npy"),
            ]

            mask_file = next((path for path in candidate_masks if os.path.exists(path)), None)
            if not mask_file:
                continue

            self.gt.append({"file_name": img_file, "mask_file": mask_file})

        if not self.gt:
            print(f"Warning: no image/mask pairs found in {base_folder}")

    def lenFiles(self):
        return len(self.gt)

    def parseGT(self, index=-1):
        if not self.gt:
            raise RuntimeError("ParserTRACENPY contains no entries.")

        if index == -1:
            gt = self.gt[random.randrange(0, len(self.gt))]
        else:
            gt = self.gt[index]

        return gt["file_name"], gt

