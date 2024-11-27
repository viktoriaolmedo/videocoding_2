class RGBto_YUV:
    def YUV_to_RGB(self, Y,U,V):
        B = round(1.163 * (Y -16) + 2.018 * (U - 128))
        G = round(1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128))
        R = round(1.164 * (Y - 16) + 1.596 * (V - 128)) 
        return(R, G, B)
    
    def RGB_to_YUV(self, R,G,B):
        Y = round(0.257 * R + 0.504 * G + 0.098 * B +16)
        U = round(-0.148 * R - 0.291 * G + 0.439 * B +128)
        V = round(0.439 * R - 0.368 * G - 0.071 * B +128)
        return(Y,U,V)
    
import subprocess
import os


def resize_image(input_image, output_image, width, height, quality = 28):

    command = ["ffmpeg","-y","-loglevel", "info","-i", input_image,"-vf", f"scale={width}:{height}"]

    if quality is not None:
        command.extend(["-q:v", str(quality)])
    
    command.append(output_image)
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        print(f"Resized image saved to {output_image}")
    except subprocess.CalledProcessError as e:
        print("Error ocurred while running FFmpeg", e.stderr)

import numpy as np

matrix = np.arange(1, 65).reshape(8, 8)

def serpentine_matrix(matrix):
    result = [None] * 64  
    n = matrix.shape[0]  
    
    a = 0  
    i = 0 
    j = 0 
    result[a] = matrix[i][j]  # First value
    
    while a < 63:  
        #step 1: (one to the right)
        if j == n - 1:  
            i += 1
        else:
            j += 1

        a += 1
        if a < 64:
            result[a] = matrix[i][j]
        
        #step 2: (Diagonal down to the left until reaching j = 0)
        while i + 1 < n and j - 1 >= 0 and a < 63:
            i += 1
            j -= 1
            a += 1
            result[a] = matrix[i][j]
        
        #step 3: (one down) 
        if i == n - 1:  
            j += 1
        else:
            i += 1

        a += 1
        if a < 64:
            result[a] = matrix[i][j]
        
        #step 4: (Diagonal up to the right until reaching i = 0)
        while j + 1 < n and i - 1 >= 0 and a < 63:
            i -= 1
            j += 1
            a += 1
            result[a] = matrix[i][j]
    
    return result

import numpy as np
import cv2
import subprocess

def load_image_bw(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  

def load_image_rgb(image_path):
    return cv2.imread(image_path)  

def serpentine(matrix):
    n = matrix.shape[0]  
    result = [None] * (n * n)  
    
    a = 0  
    i = 0 
    j = 0 
    result[a] = matrix[i, j]  # First value
    
    while a < n * n - 1:  
        #step 1: (one to the right)
        if j == n - 1:  
            i += 1
        else:
            j += 1

        a += 1
        if a < n * n:
            result[a] = matrix[i, j]
        
        #step 2: (Diagonal down to the left until reaching j = 0)
        while i + 1 < n and j - 1 >= 0 and a < n * n - 1:
            i += 1
            j -= 1
            a += 1
            result[a] = matrix[i, j]
        
        #step 3: (one down)
        if i == n - 1:  
            j += 1
        else:
            i += 1

        a += 1
        if a < n * n:
            result[a] = matrix[i, j]
        
        #step 4: (Diagonal up to the right until reaching i = 0)
        while j + 1 < n and i - 1 >= 0 and a < n * n - 1:
            i -= 1
            j += 1
            a += 1
            result[a] = matrix[i, j]
    
    return result

def serpentine_color(image):
    channels = cv2.split(image)  # Separate into B, G, and R channels
    serpentine_channels = []

    for idx, channel in enumerate(channels):
        serpentine_result = serpentine(channel)
        
        # Print serpentine result for each channel
        print(f"Serpentine traversal output for channel {idx}:")
        print(serpentine_result)
        
        n = channel.shape[0]
        serpentine_matrix = np.array(serpentine_result).reshape((n, n))
        serpentine_channels.append(serpentine_matrix)

    # Merge channels back into a color image
    serpentine_image_color = cv2.merge(serpentine_channels)
    return serpentine_image_color

def compress_to_bw(input_image_path, output_image_path):
    try:
        subprocess.run([
            'ffmpeg', '-i', input_image_path, 
            '-vf', 'format=gray',  # Convert to black and white
            '-compression_level', '10',  # Max compression for PNG (range 0-10)
            '-qscale:v', '31',  # Highest compression for JPEG (range 2-31)
            '-y', output_image_path
        ], check=True)
        
        print(f"Compressed and converted image saved at: {output_image_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"FFMPEG failed with error: {e}")

def encoding(byte_sequence):
    if not isinstance(byte_sequence, (bytes, bytearray)):
        raise ValueError("Input must be a bytes or bytearray object.")
    
    encoded_bytes = bytearray()
    i = 0
    
    while i < len(byte_sequence):
        count = 1
        current_byte = byte_sequence[i]
        j = i

        while j < len(byte_sequence) - 1:
            if byte_sequence[j] == byte_sequence[j + 1]:
                count += 1
                j += 1
            else:
                break

        encoded_bytes.append(count)
        encoded_bytes.append(current_byte)
        
        i = j + 1
    
    return bytes(encoded_bytes)

import numpy as np
from scipy.fftpack import dct, idct


class DCT:
    
    def encode_dct(self, input):    
        return dct(input, type=2, norm='ortho')
    
    def decode_dct(self, input_inverse):
        return idct(input_inverse, type=2, norm='ortho')

import pywt
import numpy as np

class DWT:
    def __init__(self, wavelet='db2', mode='smooth'):
        self.wavelet = wavelet  # db1: simplest wavelet. db2:used for more complex transformations
        self.mode = mode        # boundary handling. smooth: smooth continuity at the boundaries. periodic: periodic data

    def encode_dwt(self, input_signal):
        
        cA, cD = pywt.dwt(input_signal, self.wavelet, self.mode)
        return cA, cD
    
    def decode_dwt(self, cA, cD):

        return pywt.idwt(cA, cD, self.wavelet, self.mode)

import unittest
from unittest.mock import patch, MagicMock

class TestRGBtoYUV(unittest.TestCase):
    def test_YUV_to_RGB(self):
        yuv_converter = RGBto_YUV()
        R, G, B = yuv_converter.YUV_to_RGB(16, 128, 128)
        self.assertEqual((R, G, B), (0, 0, 0), "YUV to RGB conversion for (16,128,128) should be (0,0,0)")

    def test_RGB_to_YUV(self):
        yuv_converter = RGBto_YUV()
        Y, U, V = yuv_converter.RGB_to_YUV(0, 0, 0)
        self.assertEqual((Y, U, V), (16, 128, 128), "RGB to YUV conversion for (0,0,0) should be (16,128,128)")

class TestResizeImage(unittest.TestCase):
    def test_resize_image_success(self):
        input_image = "/Users/isall/OneDrive/UNI/4_uni/1_trim_4/Audio/video_coding/seminar1/logo_fcb_1.jpg"
        output_image = "/Users/isall/OneDrive/UNI/4_uni/1_trim_4/Audio/video_coding/seminar1/resized_image.jpg"
        width = 320
        height = 240
        quality = 28

        # Llamar a la función sin "mock"
        resize_image(input_image, output_image, width, height, quality)

        # Verificar si el archivo se creó
        import os
        self.assertTrue(os.path.exists(output_image), "El archivo de salida no se creó correctamente")
        
        
class TestSerpentine(unittest.TestCase):

    def setUp(self):
        # Input image       
        self.image = "/Users/isall/OneDrive/UNI/4_uni/1_trim_4/Audio/video_coding/seminar1/logo_fcb.jpeg"
        self.input_image = "/Users/isall/OneDrive/UNI/4_uni/1_trim_4/Audio/video_coding/seminar1/logo_fcb_resize_serpentine.jpeg"
        resize_image(image, input_image, 8, 8, quality=28) #We resize it, to obtain a nxn matrix. 

    def test_serpentine_bw(self):
    
        image_bw = cv2.imread(self.input_image, cv2.IMREAD_GRAYSCALE)
        
        # Check if we have load correctly the image
        self.assertIsNotNone(image_bw, "Failed to load the image in BW")

        # Check if the image is (nxn)
        self.assertEqual(image_bw.shape[0], image_bw.shape[1], "The BW image is not nxn")
        
        # Print matrix image
        print("Original Matrix in BW:")
        print(image_bw)

        # We apply the function
        serpentine_result = serpentine(image_bw)
        
        # print of the result
        print("Matrix after serpentine in BW:")
        print(serpentine_result)

        # Check if the result is correct. 
        self.assertEqual(len(serpentine_result), image_bw.size, "Serpentine result has incorrect size")

    def test_serpentine_color(self):
        
        image_rgb = cv2.imread(self.input_image)

        # Check if we have load correctly the image
        self.assertIsNotNone(image_rgb, "Failed to load the RGB image")

        # Check if the image is (nxn)
        self.assertEqual(image_rgb.shape[0], image_rgb.shape[1], "The RGB image is not nxn")
        
        # Print matrix image
        print("Original Matrix in RGB:")
        print(image_rgb)

        # We apply the function
        serpentine_result = serpentine_color(image_rgb)
        
        # print of the result
        print("Matrix after serpentine in RGB:")
        print(serpentine_result)

        # Check if the result is correct. 
        self.assertEqual(serpentine_result.shape, image_rgb.shape, "The serpentine color result has incorrect size")

        # Check if the 3 channels are correctly transformed
        red_channel, green_channel, blue_channel = cv2.split(serpentine_result)

        # Check if each channel has the same size as the original image
        self.assertEqual(red_channel.shape, image_rgb.shape[:2], "Red channel size mismatch")
        self.assertEqual(green_channel.shape, image_rgb.shape[:2], "Green channel size mismatch")
        self.assertEqual(blue_channel.shape, image_rgb.shape[:2], "Blue channel size mismatch")

class TestCompressToBW(unittest.TestCase):

    def test_compress_to_bw_success(self):
        # Define the paths (you can use actual image paths for testing)
        input_image_path = "/Users/isall/OneDrive/UNI/4_uni/1_trim_4/Audio/video_coding/seminar1/logo_fcb.jpeg"
        output_image_path =  "/Users/isall/OneDrive/UNI/4_uni/1_trim_4/Audio/video_coding/seminar1/output_image_compressed.jpeg"

        # Ensure the output path doesn't exist before running the test
        if os.path.exists(output_image_path):
            os.remove(output_image_path)

        # Call the function
        compress_to_bw(input_image_path, output_image_path)
        
        # Check if the output image was created
        self.assertTrue(os.path.exists(output_image_path), "Output image was not created")
class TestEncoding(unittest.TestCase):
    def test_encoding(self):
        input_bytes = bytes([0, 65, 65, 65, 66, 66, 66, 67, 67, 68, 65, 65])
        encoded = encoding(input_bytes)
        self.assertEqual(list(encoded), [1, 0, 3, 65, 3, 66, 2, 67, 1, 68, 2, 65])

    def test_encoding_non_bytes_input(self):
        with self.assertRaises(ValueError):
            encoding([1, 2, 3])  # Input that is not a byte sequence

class TestDCT(unittest.TestCase):
    def setUp(self):
        self.dct_processor = DCT()
        self.input_block = np.array([
            [52, 55, 61, 66, 70, 61, 64, 73],
            [63, 59, 66, 90, 109, 85, 69, 72],
            [62, 59, 68, 113, 144, 104, 66, 73],
            [63, 58, 71, 122, 154, 106, 70, 69],
            [67, 61, 68, 104, 126, 88, 68, 70],
            [79, 65, 60, 70, 77, 68, 58, 75],
            [85, 71, 64, 59, 55, 61, 65, 83],
            [87, 79, 69, 68, 65, 76, 78, 94]
        ])

    def test_encode_dct(self):
        dct_result = self.dct_processor.encode_dct(self.input_block)
        self.assertIsNotNone(dct_result)  # Ensure result is not None

    def test_decode_dct(self):
        dct_result = self.dct_processor.encode_dct(self.input_block)
        reconstructed = self.dct_processor.decode_dct(dct_result)
        np.testing.assert_array_almost_equal(self.input_block, reconstructed, decimal=0)

class TestDWT(unittest.TestCase):
    def setUp(self):
        self.wavelet_processor = DWT(wavelet='db2', mode='smooth')
        self.input_signal = [1, 2, 3, 4, 5, 6]

    def test_encode_dwt(self):
        cA, cD = self.wavelet_processor.encode_dwt(self.input_signal)
        self.assertIsNotNone(cA)
        self.assertIsNotNone(cD)

    def test_decode_dwt(self):
        cA, cD = self.wavelet_processor.encode_dwt(self.input_signal)
        reconstructed_signal = self.wavelet_processor.decode_dwt(cA, cD)
        np.testing.assert_array_almost_equal(self.input_signal, reconstructed_signal, decimal=1)

# Ejecuta las pruebas
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

