"""
TRACE
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import io
import json
import os
import tarfile

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

import file_utils
import imgproc
import postprocessor
from model import TraceModel
from parse_config import parse_config

app = FastAPI(title="TRACE API")

# Global variables for model and args
net = None
args = None
result_path = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "TRACE API is running"}


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """
    Upload an image and get back a tar file containing:
    - result_image.png: Processed image with annotations
    - result.json: Detection results in JSON format
    - result.xlsx: Excel file with table data
    """
    global net, args
    
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process image
    res_image, res_dict, res_xlsx_file = test_net(net, image, args)
    
    # Create tar file in memory
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        # Add result image
        img_buffer = io.BytesIO()
        Image.fromarray(res_image).save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_info = tarfile.TarInfo(name='result_image.png')
        img_info.size = len(img_buffer.getvalue())
        tar.addfile(img_info, img_buffer)
        
        # Add JSON result
        json_str = json.dumps(res_dict, indent=2)
        json_buffer = io.BytesIO(json_str.encode('utf-8'))
        json_info = tarfile.TarInfo(name='result.json')
        json_info.size = len(json_str.encode('utf-8'))
        tar.addfile(json_info, json_buffer)
        
        # Add Excel file if it exists
        if res_xlsx_file and os.path.exists(res_xlsx_file):
            tar.add(res_xlsx_file, arcname='result.xlsx')
    
    tar_buffer.seek(0)
    
    return StreamingResponse(
        tar_buffer,
        media_type='application/x-tar',
        headers={'Content-Disposition': 'attachment; filename=trace_results.tar.gz'}
    )


def test_net(net, image, args):
    global result_path

    # resize
    s = args.canvas_size
    mag_ratio = args.mag_ratio
    image = cv2.resize(image, (s, s))
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, s, mag_ratio=mag_ratio)

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = torch.autograd.Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if torch.cuda.is_available():
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y = y[0]  # ignore feature map
    res_heatmap = y[0].cpu().data.numpy()

    # Post-processing
    res_dict = postprocessor.run(res_heatmap, args, target_ratio)

    # render result image (saving xlsx file)
    res_image, res_xlsx_file = file_utils.saveTraceResult(
        "input.jpg",
        image[:, :, ::-1],
        None,
        res_dict,
        dirname=result_path,
        saveXlsx=True,
    )

    return res_image, res_dict, res_xlsx_file


if __name__ == "__main__":

    def str2bool(v):
        return v.lower() in ("yes", "y", "true", "t", "1")

    parser = argparse.ArgumentParser()
    # Base
    parser.add_argument("-c", "--config_file", default="configs/trace.json", type=str, required=True)
    parser.add_argument("-m", "--trained_model", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    # parameters
    parser.add_argument("--canvas_size", default=1024, type=int, help="Max size of image canvas")
    parser.add_argument("--mag_ratio", default=10, type=float, help="Image magnification ratio")
    parser.add_argument("--threshold1", default=0.2, type=float, help="Threshold for explicit edges")
    parser.add_argument("--threshold2", default=0.1, type=float, help="Threshold for implicit edges")

    # parse config file
    parsed_args = parser.parse_args()
    args = parse_config(parsed_args)

    result_path = "./result_app/"
    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    # prepare model
    print("Loading weights from checkpoint (" + args.trained_model + ")")
    net = TraceModel()
    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(args.trained_model))
    net.eval()

    print(f"Starting TRACE API server on {parsed_args.host}:{parsed_args.port}")
    print(f"API endpoint: POST http://{parsed_args.host}:{parsed_args.port}/process")
    print("Upload an image to get a tar.gz file with results (image, json, xlsx)")
    
    # Start FastAPI server
    uvicorn.run(app, host=parsed_args.host, port=parsed_args.port)
