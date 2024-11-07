import streamlit as st
import os
import json
import csv
import numpy as np
from PIL import Image
import torch
from sklearn.neighbors import NearestNeighbors
from streamlit_drawable_canvas import st_canvas
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from pathlib import Path
CODE_PATH = Path('tsbir/code/')
MODEL_PATH = Path('tsbir/model/')

import sys
sys.path.append(str(CODE_PATH))

##make sure CODE_PATH is pointing to the correct path containing clip.py before running
from clipmodel.model import convert_weights, CLIP
from clipmodel.clip import _transform, load


model_config_file = CODE_PATH / 'training/model_configs/ViT-B-16.json'
model_file = MODEL_PATH / 'tsbir_model_final.pt'

st.set_page_config(layout="wide")


with open(model_config_file, 'r') as f:
    model_info = json.load(f)

sketch_model = CLIP(**model_info)
checkpoint = torch.load(model_file, map_location='cpu')

sd = checkpoint["state_dict"]
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items()}

sketch_model.load_state_dict(sd, strict=False)

sketch_model = sketch_model.eval()

convert_weights(sketch_model)
preprocess_train = _transform(sketch_model.visual.input_resolution, is_train=True)
preprocess_val = _transform(sketch_model.visual.input_resolution, is_train=False)
preprocess_fn = (preprocess_train, preprocess_val)

@st.cache_data
def load_image(image_path):
    return Image.open(image_path)

all_image_path = [] 

@st.cache_data
def loadKeyframes():
    all_image_embedding = []

    
    path = "Data/Keyframe/keyframes"
    map_path = "Data/MapKeyframe/map-keyframes"
    metadata_path = "Data/Metadata/media-info"
    img_emb_path = "Data/Sketch"

    keyframes = []

    for video in sorted(os.listdir(map_path)):
        vid = video[:-4]

        if vid == '.DS_S': continue
        with open(os.path.join(metadata_path, f"{vid}.json"), 'r', encoding='utf-8') as m:
            metadata = json.load(m)

        # with open(os.path.join(img_emb_path, f"{vid}.npy"), 'rb') as o:
        #     sketch_embeddings = np.load(o)

        with open(os.path.join(map_path, video), 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                n, pts_time, _, frame_idx = row

                if int(n) < 10: 
                    n = f"00{n}"
                elif int(n) < 100:
                    n = f"0{n}"
                frame_idx = int(frame_idx)
                # semb = sketch_embeddings[int(n)-1]
                all_image_path.append(os.path.join(path, vid, f"{n}.jpg"))

                keyframe = {
                    "video": vid,
                    "n": n,
                    "time": pts_time,
                    "frame_idx": frame_idx,
                    "path": os.path.join(path, vid, f"{n}.jpg"),
                    "metadata": metadata
                    # "embedding": semb
                }

                keyframes.append(keyframe)

    all_path = 'Data/Sketch/full_sketch.npy'

    with open(all_path,'rb') as fi:
        all_image_embedding = np.load(fi)

    return keyframes,all_image_embedding

from clip.clip import tokenize

def get_feature(query_sketch, query_text, sketch_model):

    img1 = transformer(query_sketch).unsqueeze(0)

    txt = tokenize([str(query_text)])[0].unsqueeze(0)
    with torch.no_grad():
        sketch_feature = sketch_model.encode_sketch(img1)
        text_feature = sketch_model.encode_text(txt)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        sketch_feature = sketch_feature / sketch_feature.norm(dim=-1, keepdim=True)

    return sketch_model.feature_fuse(sketch_feature,text_feature)

def showKFrames(dataset,indices,k, numcols=3):
    cols = st.columns(numcols)

    for i, data in enumerate(indices):
        keyframe = dataset[data]
        with cols[i % numcols]:
            img = load_image(keyframe['path'])
            st.image(img, use_column_width=True)

            with st.popover(label=f"{keyframe['video']} - {keyframe['frame_idx']}", use_container_width=True):
                st.write(f"Video: {keyframe['video']}")
                st.write(f"Time: {keyframe['time']}")
                st.write(f"Frame Index: {keyframe['frame_idx']}")
                st.write(f"Youtube: {keyframe['metadata']['watch_url']}" + f"&t={int(float(keyframe['time']))}")

def handleSketchQuery(dataset,text_query,sketch_query,sketch_model,k,feats):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine').fit(feats)
    query_feat = get_feature(sketch_query, text_query,sketch_model)
    distances, indices = nbrs.kneighbors(query_feat.cpu().numpy())
    showKFrames(dataset,indices[0],k,3)


st.title("HoshinoAI Retrieval System")

text_query = st.sidebar.text_area("Text:", key="text_query")
#sketch_query = st.sidebar.file_uploader("Upload image:")
with st.sidebar:
    canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Color inside shapes
    stroke_width=5,
    stroke_color="#000000",
    background_color="#ffffff",
    update_streamlit=True,
    height=200,
    width=300,
    drawing_mode="freedraw",  # Freehand drawing
    key="canvas",
)

# if canvas_result.image_data is not None:
#     # Convert the NumPy array to a PIL image
#     sketch_query = Image.fromarray((canvas_result.image_data).astype('uint8'))
    if st.sidebar.button("Get Drawing"):
        if canvas_result.image_data is not None:
            sketch_query = Image.fromarray((canvas_result.image_data).astype('uint8'))

k = st.sidebar.text_input("Top k = ", 100, key="text_k")
k = int(k)

dataset,all_image_embedding = loadKeyframes()

feats = all_image_embedding
image_paths = all_image_path
transformer = preprocess_val
sketch_model = sketch_model.eval()

if text_query and sketch_query:
    handleSketchQuery(dataset,text_query,sketch_query,sketch_model,k,feats)
