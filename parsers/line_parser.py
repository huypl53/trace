# -*- coding: utf-8 -*-
import json
import os
import random

import file_utils


class ParserLine:
    """Parser for JSON-based line annotations."""

    def __init__(self, root_path, dataset, phase):
        self.gt = []
        base_folder = os.path.join(root_path, dataset, phase)
        img_files, gt_files = file_utils.list_files_with_json(base_folder)

        for img_file, gt_file in zip(img_files, gt_files):
            with open(gt_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            h_lines = []
            v_lines = []
            for line in data.get("lines", []):
                line_type = line.get("type")
                points = line.get("points", [])
                if len(points) != 2:
                    continue
                start = tuple(points[0])
                end = tuple(points[1])
                if line_type == "horizontal":
                    h_lines.append((start, end))
                elif line_type == "vertical":
                    v_lines.append((start, end))

            self.gt.append({"file_name": img_file, "h_lines": h_lines, "v_lines": v_lines})

    def getDatasetSize(self):
        return len(self.gt)

    def lenFiles(self):
        return len(self.gt)

    def parseGT(self, index=-1):
        if index == -1:
            gt = self.gt[random.randrange(0, len(self.gt))]
        else:
            gt = self.gt[index]

        return gt["file_name"], gt


if __name__ == "__main__":
    parser = ParserLine("/data/db/line", "LINESET", "train")
    print(parser.lenFiles())
    print(parser.parseGT(-1))
