import os
import zipfile

def unpack_dir(zipdir, unpacked_dir):
    directory_path = zipdir

    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a ZIP file
        if filename.endswith(".zip"):
            # Construct the full path of the ZIP file
            zip_file_path = os.path.join(directory_path, filename)

            # Specify the directory where you want to extract the contents
            activity = filename.split(".")[-2]
            extract_dir = f"{unpacked_dir}/{activity}"

            # Create the extraction directory if it doesn't exist
            os.makedirs(extract_dir, exist_ok=True)

            # Open the ZIP file for extraction
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                # Extract all contents to the extraction directory
                zip_ref.extractall(extract_dir)

            # print(f"Extracted '{zip_file_path}' to '{extract_dir}'")