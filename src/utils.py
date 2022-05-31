import numpy as np
import cv2
import os
from PIL import Image
import pywt

def resize_and_save_img(src, destination_path):
    """
    resize_and_save_img aplica un resize a la imagen en src y la guarda en destination_path.

    :param src: Dirección a la imagen a la que se le desea hacer resize.
    :param destination_path: Dirección en la cual se debe guardar la imagen procesada.
    :return: None.
    """
    original_img = cv2.imread(src)
    old_image_height, old_image_width, channels = original_img.shape
    new_image_width = 256        
    new_image_height = 256
    color = (255,255,255)

    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # Centrar imagen
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = original_img

    Image.fromarray(result).save(destination_path)



def generate_new_data():
    """
    generate_new_data se corre solo una vez. Junta la data de Train y Test en un solo directorio unificado.

    :return: None.
    """
    train_dir = "data/dataset"
    destination_dir = "data/resize_dataset"

    for train_img in os.listdir(train_dir):
        resize_and_save_img(f"{train_dir}/{train_img}", f"{destination_dir}/{train_img}")



def get_vector_from_image(image, iterations):
    """
    get_vector_from_image obtiene el vector característico de la imagen image

    :param image: Imagen en formato vector.
    :param iterations: Entero que indica la cantidad de veces que se aplica el wavelet a la imagen.
    :return LL: Vector característico sin la compresión a 1D.
    :return LL.flatten(): Vector característico en 1D.
    """
    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
    for _ in range(iterations - 1):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL, LL.flatten()



def get_data(src_dir, iterations):
    """
    get_data

    :param src_dir: Directorio origen para leer las imágenes.
    :param iterations: Entero que indica la cantidad de veces que se aplica el wavelet a la imagen.
    :return np.asarray(x): Vector con los vectores característicos de las imágenes en 1D.
    :return np.asarray(raw_x): Vector con los vectores característicos de las imágenes sin la compresión a 1D.
    """
    x = []
    raw_x = []

    for train_img in os.listdir(src_dir):
        image_path = f"{src_dir}/{train_img}"
        img = Image.open(image_path)

        fv = get_vector_from_image(img, iterations)
        raw_x.append(fv[0])
        x.append(fv[1])
    return np.asarray(x), np.asarray(raw_x)


def iterate_data(X_raw):
    """
    iterate_data aplica una compresión adicional a los datos de X_raw

    :param X_raw: Data X sin comprimir.
    :return np.asarray(X): Nuevos vectores característicos en 1D.
    :return np.asarray(X_new_raw): Nuevos vectores característicos sin la compresión a 1D.
    """
    X = []
    X_new_raw = []
    for i in range(X_raw.shape[0]):
        LL , (LH, HL, HH) = pywt.dwt2(X_raw[i], 'haar')
        X_new_raw.append(LL)
        X.append(LL.flatten())
    return np.asarray(X), np.asarray(X_new_raw)



def normalization(data):
    """
    normalization aplica la normalización a un conjunto de datos.

    :param data: Datos a comprimir
    :return np.asarray(normalized_data).transpose(): Conjunto de datos normalizados.
    """
    columns = data.transpose()
    normalized_data = []
    for column in columns:
        minimum = min(column)
        maximum = max(column)
        normalized_column = np.asarray([(n - minimum) / (maximum - minimum) for n in column])
        normalized_data.append(normalized_column)
    return np.asarray(normalized_data).transpose()
    