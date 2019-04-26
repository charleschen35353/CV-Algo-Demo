import argparse
from skimage.io import imread, imsave

from inpainter import Inpainter


def main():

    image = imread("1.jpg")
    mask = imread("mask1.jpg", as_grey=True)

    output_image = Inpainter(image, mask, 9).inpaint()
    imsave("output.jpg", output_image, quality=100)




if __name__ == '__main__':
    main()
