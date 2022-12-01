import os
import zipfile
import requests
#import gdown

# url = 'https://drive.google.com/uc?id=16leuM9PuFXAkmw34XeQy-84h8WGAYxJw&export=download'
# output = 'lematizacion-es.zip'
# gdown.download(url, output, quiet=False)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_lemma():
    file_id = '16leuM9PuFXAkmw34XeQy-84h8WGAYxJw'
    output = "lematizacion-es.zip"
    download_file_from_google_drive(file_id, output)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall()

if __name__ == "__main__":
    download_lemma