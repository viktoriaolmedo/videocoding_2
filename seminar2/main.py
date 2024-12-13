from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from typing import List, Optional
import numpy as np
from semi1 import RGBto_YUV, YUVto_RGB, resize_image, compress_to_bw, encoding, DCT, DWT
from typing import Union
import pywt
import subprocess
import os
import cv2
from tempfile import NamedTemporaryFile
import shutil
import tempfile


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
    input_file = "/Users/viktoriaolmedo/Downloads/bbb_sunflower_1080p_30fps_normal.mp4"
    
    output_file = os.path.join(os.path.dirname(input_file),f"output_{width}x{height}.mp4")
    print("Ruta actual:", os.getcwd())
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Archivo de entrada no encontrado.")


    # ffmpeg command
    command = ["ffmpeg","-i", input_file,"-vf", f"scale={width}:{height}","-c:v", "libx264","-crf",
    "23","-c:a", "copy",output_file]

    # execute command using subprocess
    result = subprocess.run(command, capture_output=True, text=True)
     
    return {"message": "Resolución cambiada", "output_file": output_file}

@app.get("/api/modify_chroma_subsampling")
def modify_chroma_subsampling():

    input_file = "/Users/viktoriaolmedo/Downloads/bbb_sunflower_1080p_30fps_normal.mp4"
    
    output_file = os.path.join(os.path.dirname(input_file),f"output_chroma_subsampling_411.mp4")
    print("Ruta actual:", os.getcwd())
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Archivo de entrada no encontrado.")

    command = ["ffmpeg","-i", input_file,"-c:v", "libx264","-vf", "format=yuv411p",output_file]

    result = subprocess.run(command, capture_output=True, text=True)
     
    return {"message": "Chroma subsampling cambiado a YUV 4:1:1 ", "output_file": output_file}
    

@app.get("/api/print_data")
def get_video_info_cv2():
    input_file = "/Users/viktoriaolmedo/Downloads/bbb_sunflower_1080p_30fps_normal.mp4"
    
    # open video using OpenCV
    video_capture = cv2.VideoCapture(input_file)

    # retrieve data using OpenCV methods
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    duration = frame_count / fps if fps > 0 else 0

    # convert integer to string
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

    video_capture.release()

    return {
        "duration": f"{duration:.2f} seconds",
        "frame_rate": f"{fps:.2f} FPS",
        "resolution": f"{width}x{height}",
        "codec": codec_str,
        "frame_count": frame_count
    }
    
    
@app.post("/api/bbb_container")
def package_bbb_container():
    input_file = "/Users/viktoriaolmedo/Downloads/bbb_sunflower_1080p_30fps_normal.mp4"
    output_dir = "/Users/viktoriaolmedo/Downloads/bbb_output"
    
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Original BBB file not found.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # File paths for intermediate and final output
    video_output = os.path.join(output_dir, "bbb_20s.mp4")
    audio_aac_mono = os.path.join(output_dir, "bbb_20s_aac_mono.aac")
    audio_mp3_stereo = os.path.join(output_dir, "bbb_20s_mp3_stereo.mp3")
    audio_ac3 = os.path.join(output_dir, "bbb_20s_ac3.ac3")
    packaged_output = os.path.join(output_dir, "bbb_packaged.mp4")

    try:
        # cut BBB into 20 seconds
        subprocess.run(
            ["ffmpeg",
                "-i", input_file,
                "-t", "20",
                "-c:v", "libx264",
                "-c:a", "aac",
                video_output
            ],
            check=True
        )

        # AAC mono track
        subprocess.run(
            [
                "ffmpeg",
                "-i", video_output,
                "-vn",
                "-ac", "1",
                "-c:a", "aac",
                audio_aac_mono
            ],
            check=True
        )

        # MP3 stereo with lower bitrate
        subprocess.run(
            [
                "ffmpeg",
                "-i", video_output,
                "-vn",
                "-ac", "2",
                "-b:a", "128k",
                "-c:a", "libmp3lame",
                audio_mp3_stereo
            ],
            check=True
        )

        # AC3 codec
        subprocess.run(
            [
                "ffmpeg",
                "-i", video_output,
                "-vn",
                "-c:a", "ac3",
                audio_ac3
            ],
            check=True
        )

        # package everything into one MP4 container
        subprocess.run(
            [
                "ffmpeg",
                "-i", video_output,
                "-i", audio_aac_mono,
                "-i", audio_mp3_stereo,
                "-i", audio_ac3,
                "-map", "0:v:0",   #mapping
                "-map", "1:a:0",
                "-map", "2:a:0",
                "-map", "3:a:0",
                "-c:v", "copy",
                "-c:a", "copy", 
                packaged_output
            ],
            check=True
        )

        return {
            "message": "Packaged BBB container successfully created.",
            "packaged_output": packaged_output
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {e}")


@app.post("/api/count_tracks")
async def count_tracks(file: UploadFile = File(...)):
    """
    This endpoint accepts an MP4 file and returns the number of tracks (streams) in the file.
    """
    try:
        # temporary file to store uploaded file
        with NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name
        
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=index", "-of", "csv=p=0", tmp_file_path],
            stdout=subprocess.PIPE,  # capture the standard output
            stderr=subprocess.PIPE,  # capture error output in case of issues
            text=True,  # capture output as text
            check=False
        )

        #if error
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"FFprobe error: {result.stderr}")
        
        # count number of tracks based on the output
        track_count = len(result.stdout.strip().splitlines())

        # clean up temporary file
        os.remove(tmp_file_path)

        return {"track_count": track_count}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error while processing the file: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error handling the uploaded file: {e}")


@app.post("/generate-video-with-macroblocks")
async def generate_video_with_macroblocks(file: UploadFile = File(...)):
    try:
        # temporary directory to input the video
        temp_dir = tempfile.mkdtemp()
        input_path = f"{temp_dir}/input.mp4"
        output_path = "/Users/viktoriaolmedo/Downloads/macroblocks_vectors.mp4"

        # save file to the temporary directory
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        ffmpeg_command = [
            "ffmpeg",
            "-flags2", "+export_mvs",
            "-i", input_path,
            "-vf", "codecview=mv=pf+bf+bb",
            "-y", output_path
        ]

        subprocess.run(ffmpeg_command, check=True)

        return {"message": "Video processed successfully", "file": output_path}

    except Exception as e:
        return {"error": f"Error processing the file: {str(e)}"}


@app.post("/generate-video-with-yuv-histogram")
async def generate_video_with_yuv_histogram(file: UploadFile = File(...)):
    try:
        # temporary directory
        temp_dir = tempfile.mkdtemp()
        input_path = f"{temp_dir}/input.mp4"
        output_path = "/Users/viktoriaolmedo/Downloads/yuv_histogram.mp4"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        ffmpeg_command = [
            "ffmpeg",
            "-i", input_path,
            "-vf", "histogram",
            "-y", output_path
        ]

        subprocess.run(ffmpeg_command, check=True)

        return {"message": "Video processed successfully", "file": output_path}

    except Exception as e:
        return {"error": f"Error processing the file: {str(e)}"}

@app.get("/api/convert_video_codecs")
def convert_video_codecs():
    input_file = "/Users/isall/Downloads/bbb_sunflower_1080p_30fps_normal_2.mp4"
    output_dir = os.path.dirname(input_file)
    
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail="Original BBB file not found.")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define codec configurations
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    codecs = {
        "vp8": {"ext": "webm", "codec": "libvpx"},
        "vp9": {"ext": "webm", "codec": "libvpx-vp9"},
        "h265": {"ext": "mp4", "codec": "libx265"},
        "av1": {"ext": "mkv", "codec": "libaom-av1"}
    }

    output_files = {}

    try:
        for codec_name, settings in codecs.items():
            output_file = os.path.join(output_dir, f"{base_name}_{codec_name}.{settings['ext']}")
            command = [
                "ffmpeg",
                "-i", input_file,
                "-c:v", settings["codec"],
                "-crf", "30",  # Compression level
                "-b:v", "0",   # Use constant quality for libvpx/libaom
                output_file
            ]

            # Run the FFmpeg command
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to convert video to {codec_name}: {result.stderr}"
                )

            output_files[codec_name] = output_file

        return {
            "message": "Video successfully converted to all codecs.",
            "output_files": output_files
        }

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing FFmpeg: {e}"
        )
        

# @app.post("/api/encoding_ladder")
# def create_encoding_ladder(resolutions: list = [(1920, 1080), (1280, 720), (640, 360)], bitrates: list = [5000, 2500, 1000]):
#     """
#     Create an encoding ladder by generating multiple video versions with different resolutions and bitrates.

#     Args:
#     - resolutions (list): List of tuples specifying width and height (e.g., [(1920, 1080), (1280, 720)]).
#     - bitrates (list): List of bitrates in kbps corresponding to the resolutions (e.g., [5000, 2500]).

#     Returns:
#     - JSON response with paths to the encoded files.
#     """
#     input_file = "/Users/isall/Downloads/bbb_sunflower_1080p_30fps_normal_2.mp4"
#     output_dir =  os.path.dirname(input_file)

#     if not os.path.exists(input_file):
#         raise HTTPException(status_code=404, detail="Input video file not found.")

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     base_name = os.path.splitext(os.path.basename(input_file))[0]

#     if len(resolutions) != len(bitrates):
#         raise HTTPException(status_code=400, detail="Resolutions and bitrates must have the same length.")

#     output_files = {}

#     for (width, height), bitrate in zip(resolutions, bitrates):
#         output_file = os.path.join(output_dir, f"{base_name}_{width}x{height}_{bitrate}kbps.mp4")
#         command = [
#             "ffmpeg",
#             "-i", input_file,
#             "-vf", f"scale={width}:{height}",
#             "-b:v", f"{bitrate}k",
#             "-c:v", "libx264",
#             output_file
#         ]

#         result = subprocess.run(command, capture_output=True, text=True)

#         if result.returncode != 0:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to create encoding ladder for resolution {width}x{height}: {result.stderr}"
#             )

#         output_files[f"{width}x{height}_{bitrate}kbps"] = output_file

#     return {"message": "Encoding ladder successfully created.", "output_files": output_files}
