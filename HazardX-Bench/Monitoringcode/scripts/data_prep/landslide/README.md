# Data Setup
## Landslide

Run the commands below from the repository root. Current landslide configs expect
all prepared datasets under `./data/landslides/`.
For the Python data-prep helpers, install dependencies first: `pip install -e .`.

Config-expected runtime paths:
- Landslide4Sense: `data/landslides/Landslide4Sense/{train,val,test}.txt`
- GVLM: `data/landslides/GVLM_CD/{train,val,test}.txt`
- HRGLDD: `data/landslides/HR_GLDD/{trainX,trainY,valX,valY,testX,testY}.npy`
- CASLandslide: the `dataset_path` and `data_list_path` values in
  `configs/landslide/cas/*.yaml` (the supplied configs use
  `/root/workcsc/datasets/2CASLandslide` and the split files under
  `scripts/data_prep/landslide/CASLandslide/`)

### Landslide4Sense
```bash
mkdir -p ./data/landslides/Landslide4Sense/raw
# download:
#   https://zenodo.org/records/10463239/files/TrainData.zip?download=1
#   https://zenodo.org/records/10463239/files/ValidData.zip?download=1
#   https://zenodo.org/records/10463239/files/TestData.zip?download=1
# save them as:
#   ./data/landslides/Landslide4Sense/raw/TrainData.zip
#   ./data/landslides/Landslide4Sense/raw/ValidData.zip
#   ./data/landslides/Landslide4Sense/raw/TestData.zip
wget -O ./data/landslides/Landslide4Sense/raw/TrainData.zip https://zenodo.org/records/10463239/files/TrainData.zip
wget -O ./data/landslides/Landslide4Sense/raw/ValidData.zip https://zenodo.org/records/10463239/files/ValidData.zip
wget -O ./data/landslides/Landslide4Sense/raw/TestData.zip https://zenodo.org/records/10463239/files/TestData.zip

unzip -o ./data/landslides/Landslide4Sense/raw/TrainData.zip -d ./data/landslides/Landslide4Sense
unzip -o ./data/landslides/Landslide4Sense/raw/ValidData.zip -d ./data/landslides/Landslide4Sense
unzip -o ./data/landslides/Landslide4Sense/raw/TestData.zip -d ./data/landslides/Landslide4Sense
cp scripts/data_prep/landslide/Landslide4Sense/train.txt ./data/landslides/Landslide4Sense/
cp scripts/data_prep/landslide/Landslide4Sense/val.txt ./data/landslides/Landslide4Sense/
cp scripts/data_prep/landslide/Landslide4Sense/test.txt ./data/landslides/Landslide4Sense/

# optional: reclaim space after verifying setup
# rm -rf ./data/landslides/Landslide4Sense/raw
```

Expected directory structure after setup:
```text
data/
|___landslides
|   |___Landslide4Sense
|   |   |___TrainData
|   |   |   |___img
|   |   |   |   |___image_*.h5
|   |   |   |___mask
|   |   |       |___mask_*.h5
|   |   |___ValidData
|   |   |   |___img
|   |   |   |___mask
|   |   |___TestData
|   |   |   |___img
|   |   |   |___mask
|   |   |___train.txt
|   |   |___val.txt
|   |   |___test.txt
|   |   |___raw
```

### GVLM
```bash
mkdir -p ./data/landslides/GVLM_CD/raw
# manually download GVLM from https://drive.google.com/file/d/1R6U5GmBHVDi9g3XM09jYCnaqWSwEpBj-
# place the raw per-site folders under ./data/landslides/GVLM_CD/raw/GVLM_CD/
# each site folder should contain im1.png, im2.png, and ref.png

python scripts/data_prep/landslide/GVLM/1GVLM_Img_clip.py
cp scripts/data_prep/landslide/GVLM/train.txt ./data/landslides/GVLM_CD/
cp scripts/data_prep/landslide/GVLM/val.txt ./data/landslides/GVLM_CD/
cp scripts/data_prep/landslide/GVLM/test.txt ./data/landslides/GVLM_CD/

# optional: reclaim space after verifying setup
# rm -rf ./data/landslides/GVLM_CD/raw
```

Expected directory structure before clipping:
```text
data/
|___landslides
|   |___GVLM_CD
|   |   |___raw
|   |       |___GVLM_CD
|   |           |___<site_a>
|   |           |   |___im1.png
|   |           |   |___im2.png
|   |           |   |___ref.png
|   |           |___<site_b>
|   |               |___...
```

Expected directory structure after setup:
```text
data/
|___landslides
|   |___GVLM_CD
|   |   |___t1
|   |   |   |___*.jpg
|   |   |___t2
|   |   |   |___*.jpg
|   |   |___label
|   |   |   |___*.png
|   |   |___train.txt
|   |   |___val.txt
|   |   |___test.txt
|   |   |___raw
|   |       |___GVLM_CD
|   |           |___...
```

### HRGLDD
```bash
mkdir -p ./data/landslides/HR_GLDD
# manually download HRGLDD from https://zenodo.org/records/7189381#.Y0a2UHZBxD9
# place the prepared numpy arrays under ./data/landslides/HR_GLDD
wget -O ./data/landslides/HR_GLDD/testX.npy https://zenodo.org/records/7189381/files/testX.npy
wget -O ./data/landslides/HR_GLDD/testY.npy https://zenodo.org/records/7189381/files/testY.npy
wget -O ./data/landslides/HR_GLDD/trainX.npy https://zenodo.org/records/7189381/files/trainX.npy
wget -O ./data/landslides/HR_GLDD/trainY.npy https://zenodo.org/records/7189381/files/trainY.npy
wget -O ./data/landslides/HR_GLDD/valX.npy https://zenodo.org/records/7189381/files/valX.npy
wget -O ./data/landslides/HR_GLDD/valY.npy https://zenodo.org/records/7189381/files/valY.npy

# current configs use trainX.npy, trainY.npy, valX.npy, valY.npy, testX.npy, and testY.npy
```

Expected directory structure after setup:
```text
data/
|___landslides
|   |___HR_GLDD
|   |   |___trainX.npy
|   |   |___trainY.npy
|   |   |___valX.npy
|   |   |___valY.npy
|   |   |___testX.npy
|   |   |___testY.npy
```

### CASLandslide

Download and extract the official CAS Landslide archive. The dataset root must
contain one directory per region/source, and every region must contain paired
TIFF files in `img/` and `mask/`. The optional `label/` directory distributed
with the dataset is not read by the binary segmentation adapter.

Expected directory structure:

```text
<CAS_ROOT>/
|___<region_a>
|   |___img
|   |   |___<sample_1>.tif
|   |___mask
|   |   |___<sample_1>.tif
|   |___label                 # optional; not used
|___<region_b>
|   |___img
|   |___mask
|   |___label
|___...
```

Generate deterministic train/validation/test lists from the repository root.
The default ratios are 70%/15%/15%, and only image files with a matching mask
are included. Paths written to the lists are relative to `<CAS_ROOT>`, so the
lists remain valid if the whole dataset directory is moved later.

```bash
python scripts/data_prep/landslide/CASLandslide/make_splits.py \
  --dataset-root /path/to/CASLandslide \
  --output-dir scripts/data_prep/landslide/CASLandslide \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42

wc -l scripts/data_prep/landslide/CASLandslide/{train,val,test}.txt
```

The repository also includes pre-generated split files made with seed 42. They
can be used directly when the extracted archive has the same regional folder
names and layout. Regenerate them after adding/removing regions or changing the
desired split ratios.

Before training, update all `dataset_path` entries in the selected CAS config
to the extracted dataset root. If the repository is located somewhere other
than `/root/workcsc/AnyDisasterMapping`, also update its three
`data_list_path` entries and `output_dir`. For example:

```yaml
dataset:
  name: cas_landslide
  train:
    dataset_path: /path/to/CASLandslide
    data_list_path: /path/to/AnyDisasterMapping/scripts/data_prep/landslide/CASLandslide/train.txt
  val:
    dataset_path: /path/to/CASLandslide
    data_list_path: /path/to/AnyDisasterMapping/scripts/data_prep/landslide/CASLandslide/val.txt
  test:
    dataset_path: /path/to/CASLandslide
    data_list_path: /path/to/AnyDisasterMapping/scripts/data_prep/landslide/CASLandslide/test.txt
```

The adapter loads `img` as RGB, converts every non-zero `mask` pixel to the
landslide class (`1`), and treats zero as background (`0`). The baseline CAS
configs resize the original tiles to 128 x 128; adjust the resize settings and
batch size if native-resolution training is required.

Train and evaluate a model with a CAS config, for example:

```bash
python train.py --config configs/landslide/cas/seg_unet.yaml

python test.py \
  --exp_path output/CASLandslide/seg_unet \
  --checkpoint best.pth
```
