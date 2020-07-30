from PIL import Image

image = Image.open('square.bmp')
src2 = image.resize((512,512))
src2.save('square.bmp')