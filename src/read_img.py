from PIL import Image

def get_image_pixels(img_path):
    pixels = []
    img = Image.open(img_path)
    rgb_im = img.convert('RGB')
    for i in range(0, img.width):
        row = []
        for j in range(0, img.height):
            r, g, b = rgb_im.getpixel((i, j))
            row.append([r, g, b])
        pixels.append(row)
    return pixels