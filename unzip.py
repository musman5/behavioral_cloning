import zipfile
with zipfile.ZipFile('../data/data.zip', 'r') as zip_ref:
    zip_ref.extractall('../data/')