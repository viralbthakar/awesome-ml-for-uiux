import cv2
import numpy as np
import os
from os.path import join, isfile, splitext, isdir
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True,
                    help="Directorio con las images originales")
parser.add_argument("--output-dir", required=True,
                    help="Directorio de guardado")
parser.add_argument("--ext-dir", required=True, help="ha")
a = parser.parse_args()


def auto_canny(image, sigma=0.33):
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(image, lower, upper)


def preprocess(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # return cv2.GaussianBlur(gray_image, (3, 3), 0)
    return gray_image


def draw_edges():
    count = 0
    # Se crean los directorios por estilo para las imágenes de bordes.
    if not isdir(a.output_dir):
        os.makedirs(a.output_dir)
    if not isdir(a.ext_dir):
        os.makedirs(a.ext_dir)
    for artwork_name in os.listdir(a.input_dir):
        count += 1
        print("Processing %d images \n" % (count))
        artwork_path = os.path.join(a.input_dir, artwork_name)
        image = preprocess(artwork_path)
        edge = 255 - auto_canny(image)
        target_path = os.path.join(a.output_dir, artwork_name)
        print(target_path)
        cv2.imwrite(target_path, edge)
        vis = np.concatenate((image, edge), axis=1)
        target_path2 = os.path.join(a.ext_dir, artwork_name)
        cv2.imwrite(target_path2, vis)


draw_edges()
