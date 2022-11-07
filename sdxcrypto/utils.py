import io
import os
import zipfile
import logging
from PIL import Image
from fastapi import Response

# def get_bytes_value(image:Image):
#     img_byte_arr = io.BytesIO()
#     image.save(img_byte_arr, format='JPEG')
#     return img_byte_arr.getvalue()

def createLogHandler(job_name, log_file):
    logger = logging.getLogger(job_name)
    logger.setLevel(logging.INFO)
    
    ## create a file handler ##
    handler = logging.FileHandler(log_file)
    ## create a logging format ##
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

def zipfiles(images):
    zip_filename = "archive.zip"

    s = io.BytesIO()
    zf = zipfile.ZipFile(s, "w")

    for fn, img in images:
        # Add file, at correct path
        zf.writestr(fn, img)

    # Must close zip for all contents to be written
    zf.close()

    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(s.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

    return resp