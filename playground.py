import PIL
import PIL.Image


img = PIL.Image.open('12x8.png')

print(img.getbbox())