from argparse import ArgumentParser
import yaml
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import sys
import os
from pathlib import Path



class PrepareDataset:
    def __init__(self, path_inp, path_out):
        self.path_inp = path_inp
        self.path_out = path_out
        self.counter = 0
        self.exc_counter = 0
        self.dimens = [120, 100]

    def generate_dir(self, path_out):
        Path(path_out).mkdir(parents=True, exist_ok=True)

    def iterate(self):
        for file in os.listdir(self.path_inp):
            img_name = os.path.join(self.path_inp, file)
            if os.path.isfile(img_name):
                self.filter(file)

    def filter(self, img_name):
        image = cv2.imread(os.path.join(self.path_inp,img_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30))
        if len(faces) == 0:
            self.exc_counter += 1
            Image.fromarray(image.astype(np.uint8)).save(os.path.join('/Users/aleksandrsimonyan/Desktop/trash', img_name))

        else:

            for (x, y, w, h) in faces:
                faces = image[y:y + h, x:x + w]
                warped = cv2.resize(faces, self.dimens)
            Image.fromarray(warped.astype(np.uint8)).save(os.path.join(self.path_out,img_name))
            self.counter += 1
            print(f"\rImages processed -> {self.counter} and number of excluded images -> {self.exc_counter}", end='')
            sys.stdout.flush()


if __name__ =='__main__':
    prep = PrepareDataset('/Users/aleksandrsimonyan/Desktop/CACD2000/', '/Users/aleksandrsimonyan/Desktop/cleaned_coco/')
    prep.generate_dir('/Users/aleksandrsimonyan/Desktop/cleaned_coco/')
    prep.iterate()










