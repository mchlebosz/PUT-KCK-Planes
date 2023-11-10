import copy
import skimage as ski
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

PLANES_TO_SHOW = 6


def set_path():
    #change python path to script location
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)


def load_planes(amount):
    planes = []
    for i in range(amount+1):
        string_i = str(i)
        if i < 10:
             string_i = "0" + string_i
        planes.append(ski.io.imread("planes/samolot" + string_i + ".jpg"))
    return planes

def histogram_matching(planes):
    #match histograms of images
    reference = copy.deepcopy(planes[6])
    matched_images = []
    for plane in planes:
        matched_image = ski.exposure.match_histograms(plane, reference, channel_axis=-1)
        matched_images.append(matched_image)
    return matched_images

def adjust_contrast(planes, gamma=0.5, gain=0.9):
    #adjust contrast of images
    contrast_images = []
    for plane in planes:
        contrast_image = ski.exposure.adjust_gamma(plane, gamma, gain)
        contrast_images.append(contrast_image)
    return contrast_images

def modify_threshold(planes):
    #change images to grayscale and modify threshold, minimize color amount
    binary_images = []
    for plane in planes:
        threshold = ski.filters.threshold_otsu(plane)
        binary_image = plane > threshold
        binary_image = binary_image.astype(np.uint8) * 255
        binary_images.append(binary_image)
    return binary_images

def to_grey(planes):
    #change images to grayscale
    grey_images = []
    for plane in planes:
        grey_image = ski.color.rgb2gray(plane)
        grey_images.append(grey_image)
    return grey_images


def show_images(images):
    num_images = len(images)
    rows = 3  # Number of rows
    cols = (num_images + 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap='gray')
            ax.set_title(f'Plane {i}')
            ax.axis('off')  # Disable empty subplot axes if there are more subplots than images
    plt.tight_layout()

def soften_edges(images):
    #soften edges of images
    softened_images = []
    for image in images:
        softened_image = ski.filters.gaussian(image, sigma=1)
        softened_images.append(softened_image)
    return softened_images

def show_histogram(images):
    num_images = len(images)
    rows = 3  # Number of rows
    cols = (num_images + 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    bins = 256

    for i, ax_hist in enumerate(axes.flat):
            # Calculate histograms for red, green, and blue channels
            r_hist, r_bins = ski.exposure.histogram(images[i][:, :, 0], nbins=bins)
            g_hist, g_bins = ski.exposure.histogram(images[i][:, :, 1], nbins=bins)
            b_hist, b_bins = ski.exposure.histogram(images[i][:, :, 2], nbins=bins)
            hist, bins = ski.exposure.histogram(images[i], nbins=bins)
            #divide by 3 to make avg because of 3 channels
            hist = hist / 3



            ax_hist.plot(r_bins, r_hist, color='red', label='Red')
            ax_hist.plot(g_bins, g_hist, color='green', label='Green')
            ax_hist.plot(b_bins, b_hist, color='blue', label='Blue')
            ax_hist.plot(bins, hist, color='black', label='Avg')
            ax_hist.legend()
            ax_hist.set_title(f'Plane {i}')
            ax_hist.set_xlabel("Pixel intensity")
            ax_hist.set_ylabel("Number of pixels")

    fig.suptitle('Histograms of planes')
    plt.tight_layout()

def image_sobel(images):
    sobel_images = []
    for image in images:
        sobel_image = ski.filters.sobel(image)
        sobel_images.append(sobel_image)
    return sobel_images

def images_contour(images, level=0.8):
    images_contour = []
    for image in images:
        contours = ski.measure.find_contours(image, level)
        images_contour.append(contours)
    return images_contour


def show_img_and_hist(images):
    show_images(images)
    show_histogram(images)
    plt.show()


def show_contours(contours):
    num_images = len(contours)
    rows = 3  # Number of rows
    cols = (num_images + 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        for contour in contours[i]:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            ax.set_title(f'Plane {i}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()


if __name__ == '__main__':
    set_path()
    planes = load_planes(PLANES_TO_SHOW)
    planes2 = copy.deepcopy(planes)
    planes2 = adjust_contrast(planes2, 0.85, 1.2)
    planes2 = to_grey(planes2)
    planes2 = modify_threshold(planes2)
    edges = image_sobel(planes2)
    contours = images_contour(edges, 0.2)
    show_contours(contours)
    #show_images(edges)
    plt.show()
    exit()
    show_img_and_hist(planes2)
    planes2 = histogram_matching(planes2)
    planes2 = histogram_matching(planes2)
    planes2 = hsv_mask(planes2)

    planes2 = soften_edges(planes2)
    show_images(planes2)
