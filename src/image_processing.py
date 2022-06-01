import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from shapely.geometry import Point, Polygon


TEST_JSON_FILE = 'data/json/test.json'

class ValidationObject:
    def __init__(self, filename, size, all_points_x, all_points_y, is_tumor):
        self.filename = filename
        self.size = size
        self.all_points_x = all_points_x
        self.all_points_y = all_points_y
        self.is_tumor = is_tumor
    def __str__(self):
        return (
            f'Image {self.filename}:\n'
            f'Size: {self.size}\n'
            f'X points: {self.all_points_x}\n'
            f'Y points: {self.all_points_y}\n'
            f'Tumor?: {self.is_tumor}'
        )

def readJsonData():
    test_file = open(TEST_JSON_FILE)
    test_data = json.load(test_file)
    Y = []
    for doc_name in test_data:
        filename = test_data[doc_name]['filename']
        size = test_data[doc_name]['size']
        regions = test_data[doc_name]['regions'][0]
        regions_shape_attributes = regions['shape_attributes']
        all_points_x = regions_shape_attributes['all_points_x']
        all_points_y = regions_shape_attributes['all_points_y']
        is_tumor = True if regions['region_attributes']['shape'] == 'tumor' else False
        t = ValidationObject(filename, size, all_points_x, all_points_y, is_tumor)
        Y.append(t)
    return Y


def ratioOfDetectedPointsInsideOriginalTumor(filename, Y, detected_tumor_pixels_coords):
    coords = []
    number_of_inside_points = 0
    total_number_of_points = len(detected_tumor_pixels_coords)
    #1. recuperar el objeto correspondiente a la imagen clusterizada
    for y in Y:
        if(y.filename == filename):
            find_y = y
            break
    #2. generar tupla conjunto de coordenadas
    x_coords = find_y.all_points_x
    y_coords = find_y.all_points_y
    for i in range(len(x_coords)):
        coords.append((x_coords[i], y_coords[i]))
    poly = Polygon(coords)
    #3. calculo del ratio
    for pixel in detected_tumor_pixels_coords:
        p = Point(pixel.x, pixel.y)
        if(poly.covers(p)):
            number_of_inside_points += 1
    return (number_of_inside_points / total_number_of_points)*100



def analyze(filename, tumor_pixels):
    Y = readJsonData()
    ratio = ratioOfDetectedPointsInsideOriginalTumor(filename, Y, tumor_pixels)
    print(f"El ratio de puntos para la imagen {filename} es: {ratio}%")
    return ratio
