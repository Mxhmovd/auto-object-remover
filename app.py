import os
import cv2
import paddlehub as hub
import gradio as gr
import torch
from PIL import Image, ImageOps
import numpy as np
os.mkdir("data")
os.mkdir("dataout")
model = hub.Module(name='U2Net')
def infer(img,mask,option):
  img = ImageOps.contain(img, (700,700))
  width, height = img.size
  img.save("./data/data.png")
  result = model.Segmentation(
      images=[cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)],
      paths=None,
      batch_size=1,
      input_size=320,
      output_dir='output',
      visualization=True)
  im = Image.fromarray(result[0]['mask'])
  im.save("./data/data_mask.png")
  os.system('python predict.py model.path=/home/user/app/ indir=/home/user/app/data/ outdir=/home/user/app/dataout/ device=cpu')
  return "./dataout/data_mask.png",im
  
inputs = [gr.inputs.Image(type='pil', label="Original Image"),gr.inputs.Image(type='pil',source="canvas", label="Mask",invert_colors=True),gr.inputs.Radio(model=["automatic (U2net)"], type="value", default="manual", label="Masking option")]
outputs = [gr.outputs.Image(type="file",label="output"),gr.outputs.Image(type="pil",label="Mask")]

gr.Interface(infer, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()