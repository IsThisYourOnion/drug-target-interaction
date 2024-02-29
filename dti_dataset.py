import requests
import os

def download_davis_dataset(destination_path="davis_data"):
    url = "http://staff.cs.utu.fi/~aatapa/data/DrugTarget/davis.zip"
    file_name = "davis.zip"

    # Create directory if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)

    response = requests.get(url)

    with open(os.path.join(destination_path, file_name), "wb") as f:
        f.write(response.content)

    print("Dataset downloaded!")
