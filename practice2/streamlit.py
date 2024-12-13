import streamlit as st
import subprocess
import os
import tempfile
import shutil
import uuid

# Streamlit app title
st.title("Video Coding")

# Sidebar for method selection
st.sidebar.title("Choose a Processing Method")
processing_method = st.sidebar.selectbox(
    "Select a video processing method",
    [
        "Modify Resolution",
        "Modify Chroma Subsampling",
        "Package BBB Container",
        "Generate Video with YUV Histogram",
        "Convert Video Codecs",
        "Encoding Ladder"
    ]
)

# Modify resolution
if processing_method == "Modify Resolution":
    st.header("Step 1: Upload Video for Resolution Modification")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, uploaded_file.name)

        with open(input_file, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded video saved to: {input_file}")

        st.header("Step 2: Modify Video Resolution")
        width = st.number_input("Width (in pixels):", min_value=1, value=1280, step=1)
        height = st.number_input("Height (in pixels):", min_value=1, value=720, step=1)

        if st.button("Apply Resolution Modification"):
            resolution_output_file = os.path.join(temp_dir, f"output_{width}x{height}.mp4")
            command = [
                "ffmpeg", "-i", input_file,
                "-vf", f"scale={width}:{height}",
                "-c:v", "libx264",
                "-crf", "23",
                "-c:a", "copy",
                resolution_output_file
            ]

            try:
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success(f"Resolution modified successfully! Output saved to: {resolution_output_file}")
                    with open(resolution_output_file, "rb") as f:
                        st.download_button(
                            label="Download Video with Modified Resolution",
                            data=f,
                            file_name="modified_resolution_video.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error(f"Error in processing: {result.stderr}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Modify chroma subsampling
if processing_method == "Modify Chroma Subsampling":
    st.header("Step 1: Upload Video for Chroma Subsampling")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, uploaded_file.name)

        with open(input_file, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded video saved to: {input_file}")

        if st.button("Apply Chroma Subsampling"):
            chroma_output_file = os.path.join(temp_dir, "output_chroma_subsampling_411.mp4")
            command = [
                "ffmpeg", "-i", input_file,
                "-c:v", "libx264",
                "-vf", "format=yuv411p",
                chroma_output_file
            ]

            try:
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success(f"Chroma subsampling modified successfully! Output saved to: {chroma_output_file}")
                    with open(chroma_output_file, "rb") as f:
                        st.download_button(
                            label="Download Video with Modified Chroma Subsampling",
                            data=f,
                            file_name="modified_chroma_video.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error(f"Error in processing: {result.stderr}")
            except Exception as e:
                st.error(f"An error occurred: {e}")



# Package BBB Container
if processing_method == "Package BBB Container":
    st.header("Step 1: Upload Video for BBB Packaging")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, uploaded_file.name)

        with open(input_file, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded video saved to: {input_file}")

        if st.button("Package Video into BBB Container"):
            output_dir = os.path.join(temp_dir, "bbb_output")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # File paths for intermediate and final output
            video_output = os.path.join(output_dir, "bbb_20s.mp4")
            audio_aac_mono = os.path.join(output_dir, "bbb_20s_aac_mono.aac")
            audio_mp3_stereo = os.path.join(output_dir, "bbb_20s_mp3_stereo.mp3")
            audio_ac3 = os.path.join(output_dir, "bbb_20s_ac3.ac3")
            packaged_output = os.path.join(output_dir, "bbb_packaged.mp4")

            try:
                # Cut BBB into 20 seconds
                subprocess.run(
                    ["ffmpeg", "-i", input_file, "-t", "20", "-c:v", "libx264", "-c:a", "aac", video_output],
                    check=True
                )

                # AAC mono track
                subprocess.run(
                    ["ffmpeg", "-i", video_output, "-vn", "-ac", "1", "-c:a", "aac", audio_aac_mono],
                    check=True
                )

                # MP3 stereo with lower bitrate
                subprocess.run(
                    ["ffmpeg", "-i", video_output, "-vn", "-ac", "2", "-b:a", "128k", "-c:a", "libmp3lame", audio_mp3_stereo],
                    check=True
                )

                # AC3 codec
                subprocess.run(
                    ["ffmpeg", "-i", video_output, "-vn", "-c:a", "ac3", audio_ac3],
                    check=True
                )

                # Package everything into one MP4 container
                subprocess.run(
                    ["ffmpeg", "-i", video_output, "-i", audio_aac_mono, "-i", audio_mp3_stereo, "-i", audio_ac3,
                     "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-map", "3:a:0", "-c:v", "copy", "-c:a", "copy", packaged_output],
                    check=True
                )

                st.success(f"BBB container packaged successfully! Output saved to: {packaged_output}")
                with open(packaged_output, "rb") as f:
                    st.download_button(
                        label="Download Packaged BBB Video",
                        data=f,
                        file_name="bbb_packaged_video.mp4",
                        mime="video/mp4"
                    )

            except subprocess.CalledProcessError as e:
                st.error(f"Error during packaging: {e}")
                
                
# Count video tracks
if processing_method == "Count Tracks in Video":
    st.header("Step 1: Upload Video to Count Tracks")
    uploaded_file = st.file_uploader("Choose an MP4 video file", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, uploaded_file.name)

        with open(input_file, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded video saved to: {input_file}")

        if st.button("Count Tracks"):
            try:
                # Run FFprobe to count the number of tracks (streams) in the video
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=index", "-of", "csv=p=0", input_file],
                    stdout=subprocess.PIPE,  # capture the standard output
                    stderr=subprocess.PIPE,  # capture error output in case of issues
                    text=True,  # capture output as text
                    check=False
                )

                # Check for errors in the result
                if result.returncode != 0:
                    st.error(f"FFprobe error: {result.stderr}")
                else:
                    # Count the number of video tracks based on the output
                    track_count = len(result.stdout.strip().splitlines())
                    st.subheader(f"Track Count: {track_count}")

            except subprocess.CalledProcessError as e:
                st.error(f"Error while processing the file: {e}")
            except Exception as e:
                st.error(f"Error handling the uploaded file: {e}")
                
                
# Generate video with macroblocks visualization
if processing_method == "Generate Video with Macroblocks":
    st.header("Step 1: Upload Video for Macroblocks Visualization")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, "input.mp4")

        with open(input_file, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded video saved to: {input_file}")

        if st.button("Generate Video with Macroblocks"):
            try:
                # Define the output path for the processed video
                output_file = os.path.join(temp_dir, "macroblocks_vectors.mp4")

                # FFmpeg command to process the video and visualize motion vectors
                ffmpeg_command = [
                    "ffmpeg",
                    "-flags2", "+export_mvs",  # Export motion vectors
                    "-i", input_file,
                    "-vf", "codecview=mv=pf+bf+bb",  # Visualize forward, backward, and bidirectional motion vectors
                    "-y", output_file  # Overwrite the output file if it exists
                ]

                # Run the FFmpeg command
                subprocess.run(ffmpeg_command, check=True)

                st.success(f"Video processed successfully! The output is saved to: {output_file}")

                # Provide download button for the processed video
                with open(output_file, "rb") as f:
                    st.download_button(
                        label="Download Video with Macroblocks",
                        data=f,
                        file_name="macroblocks_vectors.mp4",
                        mime="video/mp4"
                    )

            except subprocess.CalledProcessError as e:
                st.error(f"Error during video processing: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                
                
# Generate video with YUV Histogram
if processing_method == "Generate Video with YUV Histogram":
    st.header("Step 1: Upload Video to Generate YUV Histogram")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, uploaded_file.name)

        with open(input_file, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded video saved to: {input_file}")

        if st.button("Generate Video with YUV Histogram"):
            output_file = os.path.join(temp_dir, "yuv_histogram_output.mp4")

            # FFmpeg command to generate the YUV histogram
            command = [
                "ffmpeg",
                "-i", input_file,
                "-vf", "histogram",  # Apply histogram filter for YUV channels
                "-y", output_file  # Overwrite output file if it already exists
            ]

            try:
                # Run the FFmpeg command
                result = subprocess.run(command, capture_output=True, text=True)

                if result.returncode == 0:
                    st.success(f"Video processed successfully! Output saved to: {output_file}")
                    with open(output_file, "rb") as f:
                        st.download_button(
                            label="Download Video with YUV Histogram",
                            data=f,
                            file_name="yuv_histogram_video.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error(f"Error during processing: {result.stderr}")

            except subprocess.CalledProcessError as e:
                st.error(f"An error occurred during processing: {e}")
                

# Convert video codecs
if processing_method == "Convert Video Codecs":
    st.header("Step 1: Upload Video to Convert Codecs")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Create a temporary directory and save the uploaded file
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, uploaded_file.name)

        with open(input_file, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded video saved to: {input_file}")

        # Define codec configurations
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        codecs = {
            "vp8": {"ext": "webm", "codec": "libvpx"},
            "vp9": {"ext": "webm", "codec": "libvpx-vp9"},
            "h265": {"ext": "mp4", "codec": "libx265"},
            "av1": {"ext": "mkv", "codec": "libaom-av1"}
        }

        output_files = {}

        if st.button("Convert Video to Multiple Codecs"):
            try:
                for codec_name, settings in codecs.items():
                    output_file = os.path.join(temp_dir, f"{base_name}_{codec_name}.{settings['ext']}")
                    command = [
                        "ffmpeg",
                        "-i", input_file,
                        "-c:v", settings["codec"],
                        "-crf", "30",  # Compression level
                        "-b:v", "0",   # Use constant quality for libvpx/libaom
                        "-ac", "2",    # Stereo audio
                        output_file
                    ]

                    # Run the FFmpeg command
                    result = subprocess.run(command, capture_output=True, text=True)

                    if result.returncode != 0:
                        st.error(f"Failed to convert video to {codec_name}: {result.stderr}")

                    output_files[codec_name] = output_file

                # Show success message
                st.success("Video successfully converted to all codecs.")

                # Provide download links for the converted files
                for codec_name, output_file in output_files.items():
                    with open(output_file, "rb") as f:
                        st.download_button(
                            label=f"Download {codec_name} video",
                            data=f,
                            file_name=os.path.basename(output_file),
                            mime=f"video/{settings['ext']}"
                        )

            except subprocess.CalledProcessError as e:
                st.error(f"An error occurred during FFmpeg processing: {e}")

# Encoding ladder processing
if processing_method == "Encoding Ladder":
    st.header("Step 1: Upload Video for Encoding Ladder")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        # Create a temporary directory and save the uploaded file
        temp_dir = tempfile.mkdtemp()
        input_file = os.path.join(temp_dir, uploaded_file.name)

        with open(input_file, "wb") as f:
            f.write(uploaded_file.read())

        st.success(f"Uploaded video saved to: {input_file}")

        # Define encoding ladder configurations (resolutions and bitrates)
        ladder = [
            {"width": 426, "height": 240, "bitrate": "500k"},
            {"width": 640, "height": 360, "bitrate": "800k"},
            {"width": 854, "height": 480, "bitrate": "1200k"},
            {"width": 1280, "height": 720, "bitrate": "2500k"},
            {"width": 1920, "height": 1080, "bitrate": "5000k"},
        ]

        base_name = os.path.splitext(os.path.basename(input_file))[0]
        results = []

        if st.button("Generate Encoding Ladder"):
            try:
                for level in ladder:
                    res_width = level["width"]
                    res_height = level["height"]
                    bitrate = level["bitrate"]

                    # Define output paths for each codec
                    vp9_output = os.path.join(temp_dir, f"{base_name}_vp9_{res_width}x{res_height}.webm")
                    h265_output = os.path.join(temp_dir, f"{base_name}_h265_{res_width}x{res_height}.mp4")

                    # Generate VP9 version
                    vp9_command = [
                        "ffmpeg",
                        "-i", input_file,
                        "-vf", f"scale={res_width}:{res_height}",
                        "-c:v", "libvpx-vp9",
                        "-b:v", bitrate,
                        vp9_output
                    ]
                    result = subprocess.run(vp9_command, capture_output=True, text=True)

                    if result.returncode != 0:
                        st.error(f"Failed to generate VP9 video for {res_width}x{res_height}: {result.stderr}")
                        continue

                    # Generate H.265 version
                    h265_command = [
                        "ffmpeg",
                        "-i", input_file,
                        "-vf", f"scale={res_width}:{res_height}",
                        "-c:v", "libx265",
                        "-b:v", bitrate,
                        h265_output
                    ]
                    result = subprocess.run(h265_command, capture_output=True, text=True)

                    if result.returncode != 0:
                        st.error(f"Failed to generate H.265 video for {res_width}x{res_height}: {result.stderr}")
                        continue

                    # Add results for download links
                    results.append({
                        "resolution": f"{res_width}x{res_height}",
                        "vp9_output": vp9_output,
                        "h265_output": h265_output
                    })

                # Show success message and provide download links
                st.success("Encoding ladder generated successfully!")
                
                for result in results:
                    resolution = result["resolution"]
                    vp9_file = result["vp9_output"]
                    h265_file = result["h265_output"]

                    st.subheader(f"Resolution: {resolution}")
                    with open(vp9_file, "rb") as vp9_f, open(h265_file, "rb") as h265_f:
                        st.download_button(
                            label=f"Download VP9 {resolution} (WebM)",
                            data=vp9_f,
                            file_name=os.path.basename(vp9_file),
                            mime="video/webm"
                        )
                        st.download_button(
                            label=f"Download H.265 {resolution} (MP4)",
                            data=h265_f,
                            file_name=os.path.basename(h265_file),
                            mime="video/mp4"
                        )

            except subprocess.CalledProcessError as e:
                st.error(f"An error occurred while processing FFmpeg: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")


# Cleanup temporary files when finished
if st.button("Clear Temporary Files"):
    shutil.rmtree(temp_dir)
    st.success("Temporary files cleared.")
