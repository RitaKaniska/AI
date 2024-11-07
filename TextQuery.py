import streamlit as st
import os
import json
import csv
import numpy as np
from PIL import Image
import torch
import clip
from sentence_transformers import SentenceTransformer
import faiss

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

st.set_page_config(layout="wide")

device = "cpu       "

@st.cache_data
def load_image(image_path):
    return Image.open(image_path)

@st.cache_data
def loadKeyframes():
    path = "Data/Keyframe/keyframes"
    map_path = "Data/MapKeyframe/map-keyframes"
    metadata_path = "Data/Metadata/media-info"
    clip_embeddings_path = "Data/Clip/clip-features-l14"

    keyframes = []

    for video in sorted(os.listdir(map_path)):
        vid = video[:-4]

        with open(os.path.join(metadata_path, f"{vid}.json"), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        if not os.path.exists(os.path.join(clip_embeddings_path, f"{vid}.npy")):
            break
        with open(os.path.join(clip_embeddings_path, f"{vid}.npy"), 'rb') as f:
            embeddings = np.load(f)

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
                emb = embeddings[int(n)-1]

                keyframe = {
                    "video": vid,
                    "n": n,
                    "time": pts_time,
                    "frame_idx": frame_idx,
                    "path": os.path.join(path, vid, f"{n}.jpg"),
                    "metadata": metadata,
                    "embedding": emb 
                }

                keyframes.append(keyframe)

    return keyframes


@st.cache_resource
def loadClipModel():
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess


@st.cache_resource
def loadClipMultiLingualModel():
    model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
    return model


def showKFrames(dataset, k, numcols=3):
    cols = st.columns(numcols)

    for i, keyframe in enumerate(dataset[:k]):
        with cols[i % numcols]:
            img = load_image(keyframe['path'])
            st.image(img, use_column_width=True)
            
            with st.popover(label=f"{keyframe['video']} - {keyframe['frame_idx']}", use_container_width=True):
                st.write(f"Video: {keyframe['video']}")
                st.write(f"Time: {keyframe['time']}")
                st.write(f"Frame Index: {keyframe['frame_idx']}")
                st.write(f"Youtube: {keyframe['metadata']['watch_url']}" + f"&t={int(float(keyframe['time']))}")


def findNIndex(n_query, dataset):
    for i, keyframe in enumerate(dataset):
        if keyframe['n'] == int(n_query):
            return i
    return None


def findFrameIndex(frame_query, dataset):
    for i, keyframe in enumerate(dataset):
        if keyframe['frame_idx'] == int(frame_query):
            return i
    return None


def handleMetadataQuery(dataset, video_query, n_query, frame_query):
    results = dataset
    if video_query:
        results = [keyframe for keyframe in results if video_query == keyframe['video']]

    if n_query:
        idx = findNIndex(n_query, results)
        if idx is None:
            return []
        else:
            results = [results[i] for i in range(idx - int(n_range), idx + int(n_range)) if i >= 0 and i < len(results)]

    if frame_query:
        idx = findFrameIndex(frame_query, results)
        if idx is None:
            return []
        else:
            results = [results[i] for i in range(idx - int(frame_range), idx + int(frame_range) + 1) if i >= 0 and i < len(results)]

    return results


def handleTextQuery(text_query, k, extext_query, ex_k, dataset, index, clip_model, clip_multiLingual_model, multiLingual=False):
    if not multiLingual:
        token = clip.tokenize([text_query]).to(device)
        token1 = clip.tokenize([extext_query]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(token).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy()
            
            extext_features = clip_model.encode_text(token1).float()
            extext_features /= extext_features.norm(dim=-1, keepdim=True)
            extext_features = extext_features.cpu().numpy()
    else:
        text_features = clip_multiLingual_model.encode([text_query], )

    D, I = index.search(text_features, k + ex_k)
    D1, I1 = index.search(extext_features, ex_k)
    results = []
    for i in I[0]:
        if i not in I1[0]:
            results.append(dataset[i])
        if len(results) == k:
            break
    if video_query or n_query or frame_query:
        results = handleMetadataQuery(results, video_query, n_query, frame_query)
    
    showKFrames(results, min(k, len(results)))


st.title("HoshinoAI Retrieval System")

multiLingual = st.sidebar.checkbox("Multi-lingual", False, key="multiLingual")
text_query = st.sidebar.text_area("Text:", key="text_query")
extext_query = st.sidebar.text_area("Ex_Text:", key="extext_query")
k = st.sidebar.text_input("Top k = ", 500, key="text_k")
k = int(k)
ex_k = st.sidebar.text_input("Top ex_k = ", 500, key="ex_k")
ex_k = int(ex_k)
metric = st.sidebar.selectbox("Metric:", ["L2", "Cosine"])

video_query = st.sidebar.text_area("Video:", key="1")
n_query = st.sidebar.text_area("Frame ID:", key="2")
n_range = st.sidebar.text_input("Frame IDs Range:", 10, key="3")
frame_query = st.sidebar.text_area("Frame index:", key="4")
frame_range = st.sidebar.text_input("Index Range:", 10, key="5")


dataset = loadKeyframes()
clip_model, clip_preprocess = loadClipModel()
clip_multiLingual_model = loadClipMultiLingualModel()


dimension = len(dataset[0]['embedding'])
# L2 distance
l2 = faiss.IndexFlatL2(dimension)
embeddings = np.array([keyframe['embedding'] for keyframe in dataset]).astype(np.float32)
l2.add(embeddings)

# Cosine similarity
cosine = faiss.IndexFlatIP(dimension)
cosine.add(embeddings)


if not text_query and not video_query and not n_query and not frame_query:
    showKFrames(dataset, k)
elif text_query:
    if metric == "L2":
        handleTextQuery(text_query, k, extext_query, ex_k, dataset, l2, clip_model, clip_multiLingual_model, multiLingual)
    elif metric == "Cosine":
        handleTextQuery(text_query, k, extext_query, ex_k, dataset, cosine, clip_model, clip_multiLingual_model, multiLingual)
else:
    results = handleMetadataQuery(dataset, video_query, n_query, frame_query)
    showKFrames(results, k)