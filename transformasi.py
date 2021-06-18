import sys
import numpy as np

from PIL import Image

HASIL_PATH = 'hasil.jpg'

reflection_x_matrix = np.array([[-1, 0],
                               [0, 1]])
reflection_y_matrix = np.array([[1, 0],
                               [0, -1]])
dilation_matrix = np.array([[2, 0],
                            [0, 2]])
shear_matrix = np.array([[1, 2],
                         [0, 1]])


def apply_transformation_matrix(
        pixels_array: np.ndarray,
        transformation_matrix: np.ndarray):
    rows, columns = pixels_array.shape[:2]

    new_resolution = np.dot(
        np.array(pixels_array.shape[:2]),
        transformation_matrix)
    transformed_image_array = np.zeros((abs(new_resolution[0]),
                                        abs(new_resolution[1]), 3), np.uint8)

    for row_num in range(rows):
        for column_num in range(columns):
            current_pixel_position = np.array([row_num, column_num])
            new_pixel_position = np.dot(current_pixel_position,
                                        transformation_matrix)

            transformed_image_array[new_pixel_position[0],
                                    new_pixel_position[1]] = (
                                        image_array[current_pixel_position[0],
                                                    current_pixel_position[1]]
                                    )

    return transformed_image_array


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_input_path')
    parser.add_argument('transformation_type')
    args = parser.parse_args()

    transformation_matrix = None

    if args.transformation_type == 'flip_vertical':
        transformation_matrix = reflection_x_matrix
    elif args.transformation_type == 'flip_horizontal':
        transformation_matrix = reflection_y_matrix
    elif args.transformation_type == 'shear':
        transformation_matrix = shear_matrix
    elif args.transformation_type == 'dilation_2':
        transformation_matrix = dilation_matrix
    else:
        print('''
transformation_type yang tersedia:
- flip_vertical
- flip_horizontal
- shear
- dilation_2
''')
        sys.exit(1)

    image = Image.open(args.image_input_path)
    image_array = np.asarray(image)

    transformed_image_array = apply_transformation_matrix(
        image_array, transformation_matrix)
    transformed_image = Image.fromarray(transformed_image_array)
    transformed_image.save(HASIL_PATH)
    print(f'Hasil dari transformasi disimpan ke: {HASIL_PATH}')
