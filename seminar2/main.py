from fastapi import FastAPI, Query, HTTPException
from typing import List, Optional
import numpy as np
from semi1 import RGBto_YUV, YUVto_RGB, resize_image, compress_to_bw, encoding, DCT, DWT
from typing import Union
import pywt
import subprocess
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI application!"}
            
@app.get("/api/convert_rgb_to_yuv")
def convert_rgb_to_yuv(R: int, G: int, B: int):
    converter = RGBto_YUV()
    Y, U, V = converter.RGB_to_YUV(R, G, B)
    return {"Y": Y, "U": U, "V": V}

        
@app.get("/api/convert_yuv_to_rgb")
def convert_rgb_to_yuv(Y: int, U: int, V: int):
    converter = YUVto_RGB()
    R, G, B = converter.YUV_to_RGB(Y, U, V)
    return {"R": R, "G": G, "B": B}

        
@app.post("/api/encoding")
def encoding_endpoint(byte_sequence: List[int]):
    byte_sequence = bytes(byte_sequence)
    encoded_result = encoding(byte_sequence)
    return {"encoded_result": list(encoded_result)}
    

@app.get("/api/encode_dwt")
def encode_dwt_endpoint(input_signal: str):
    
    input_signal = [int(x) for x in input_signal.split(",")]
    wavelet_processor = DWT(wavelet='db2', mode='smooth')
    
    cA, cD = wavelet_processor.encode_dwt(input_signal)
    
    return {"approximation_coefficients": cA.tolist(), "detail_coefficients": cD.tolist()}



@app.get("/api/decode_dwt")
def decode_dwt_endpoint(cA: str, cD: str):

    cA = [float(x) for x in cA.split(",")]
    cD = [float(x) for x in cD.split(",")]

    wavelet_processor = DWT(wavelet='db2', mode='smooth')
        
    decoded_signal = wavelet_processor.decode_dwt(cA, cD)
        
    return {"decoded_signal": decoded_signal.tolist()}

########################## seminar 2 #############################


@app.get("/api/modify_resolution")
def modify_resolution_endpoint(width: int, height: int):
    input_file = "/Users/isall/OneDrive/UNI/4_uni/1_trim_4/Audio/videocoding_2/seminar2/bbb_sunflower_1080p_30fps_normal.mp4"  
    
    output_file = os.path.join(os.path.dirname(input_file),f"output_{width}x{height}.mp4")
    print("Ruta actual:", os.getcwd())
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Archivo de entrada no encontrado.")


    # Construimos el comando para ffmpeg
    command = [
        "ffmpeg",
        "-i", input_file,
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264",
        "-crf", "23",
        "-c:a", "copy",
        output_file
    ]

    # Ejecutamos el comando usando subprocess
    result = subprocess.run(command, capture_output=True, text=True)
     
    return {"message": "Resolución cambiada exitosamente", "output_file": output_file}

@app.get("/api/modify_chroma_subsampling")
def modify_chroma_subsampling():

    input_file = "/Users/isall/OneDrive/UNI/4_uni/1_trim_4/Audio/videocoding_2/seminar2/bbb_sunflower_1080p_30fps_normal.mp4"  
    
    output_file = os.path.join(os.path.dirname(input_file),f"output_chroma_subsampling_420.mp4")
    print("Ruta actual:", os.getcwd())
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Archivo de entrada no encontrado.")


    # Construimos el comando para ffmpeg
    command = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", "libx264",
        "-vf", "format=yuv420p",
        output_file
    ]

    # Ejecutamos el comando usando subprocess
    result = subprocess.run(command, capture_output=True, text=True)
     
    return {"message": "Chroma subsampling cambiado a YUV 4:2:0 exitosamente", "output_file": output_file}
    
 





