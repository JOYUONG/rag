# 示例：查看当前安装的 Gradio 版本
""" import gradio as gr
#print(f"当前 Gradio 版本: {gr.__version__}")


def greet(name):
    return f"Hello, {name}!"
 
demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch() """

import gradio as gr
from PIL import Image, ImageOps
import requests
 
def invert_image(image):
    """将图像反转"""
    if image is None:
        return None
    img = Image.open(image)                             # 打开图像文件
    inverted_img = ImageOps.invert(img.convert('RGB'))  # 转换为 RGB 模式并反转颜色
    return inverted_img
 
demo = gr.Interface(
    fn=invert_image,
    inputs=gr.Image(label="上传图像"),
    outputs=gr.Image(label="反转后的图像"),
    title="图像反转工具",
    description="上传一张图像，将其颜色反转！"
)
 
demo.launch()