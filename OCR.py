import streamlit as st
import os
import json
import csv
import numpy as np
from PIL import Image
import torch
import clip
from sentence_transformers import SentenceTransformer
import unidecode
from fuzzywuzzy import fuzz


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

st.set_page_config(layout="wide")

@st.cache_data
def load_image(image_path):
    return Image.open(image_path)


@st.cache_data
def loadKeyframes():
    path = "Data\\Keyframe\\keyframes"
    map_path = "Data\\MapKeyframe\\map-keyframes"
    metadata_path = "Data\\Metadata\\media-info"
    ocr_path = "Data\\OCR"

    keyframes = []

    for video in os.listdir(map_path):
        vid = video[:-4]

        with open(os.path.join(metadata_path, f"{vid}.json"), 'r', encoding='utf-8') as m:
            metadata = json.load(m)

        with open(os.path.join(ocr_path, f"{vid}.json"), 'r', encoding='utf-8') as o:
            ocr = json.load(o)

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

                ocr_data = ocr[n+ ".jpg"]
                text = []

                for box, t, conf in ocr_data:
                    top_left = box[0]
                    bottom_right = box[2]
                
                    if top_left[1] >= 640 and bottom_right[1] <= 700:
                        continue
                    text.append(t)
                  

                keyframe = {
                    "video": vid,
                    "n": n,
                    "time": pts_time,
                    "frame_idx": frame_idx,
                    "path": os.path.join(path, vid, f"{n}.jpg"),
                    "metadata": metadata,
                    "text": text
                }

                keyframes.append(keyframe)

    
    return keyframes


# device = "cuda" if torch.cuda.is_available() else "cpu"

# @st.cache_resource
# def loadClipModel():
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     return model, preprocess


# @st.cache_resource
# def loadClipMultiLingualModel():
#     model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
#     return model


def showKFrames(dataset, k, numcols=3):
    cols = st.columns(numcols)

    for i, keyframe in enumerate(dataset[:k]):
        score, keyframe = keyframe
        with cols[i % numcols]:
            img = load_image(keyframe['path'])
            st.image(img, use_column_width=True)
            
            with st.popover(label=f"{keyframe['video']} - {keyframe['frame_idx']} - {score}", use_container_width=True):
                st.write(f"Video: {keyframe['video']}")
                st.write(f"Time: {keyframe['time']}")
                st.write(f"Frame Index: {keyframe['frame_idx']}")
                st.write(f"Text: {keyframe['text']}")
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
            return
        else:
            results = [results[i] for i in range(idx - int(n_range), idx + int(n_range)) if i >= 0 and i < len(results)]

    if frame_query:
        idx = findFrameIndex(frame_query, results)
        if idx is None:
            return
        else:
            results = [results[i] for i in range(idx - int(frame_range), idx + int(frame_range) + 1) if i >= 0 and i < len(results)]

    return results


def processText(text):
    text = unidecode.unidecode(text).lower()
    return text


def handleOcrQuery(dataset, text_query):
    top_result = []
    fuzzy_result = []
    text_query = processText(text_query)
    for keyframe in dataset:
        for text in keyframe['text']:
            text = processText(text)
            if text_query in text:
                top_result.append((100, keyframe))
                break
            
    
    if fuzzy:
        for keyframe in dataset:
            score = 0
            for text in keyframe['text']:
                text = processText(text)
                mean_ratio = (fuzz.ratio(text_query, text) + 4*fuzz.partial_ratio(text_query, text)) / 5
                score = max(score, mean_ratio)
            
            if score > 70:
                fuzzy_result.append((score, keyframe))
                
        fuzzy_result = fuzzy_result[:k]
        fuzzy_result = sorted(fuzzy_result, key=lambda x: x[0], reverse=True)
    
    result = top_result + fuzzy_result

    # if video_query or n_query or frame_query:
    #     result = handleMetadataQuery(result, video_query, n_query, frame_query)
    showKFrames(result, min(k, len(result)))


st.title("HoshinoAI Retrieval System")

text_query = st.sidebar.text_area("Text:", key="text_query")
k = st.sidebar.text_input("Top k = ", 500, key="text_k")
k = int(k)
fuzzy = st.sidebar.checkbox("Fuzzy Search", key="fuzzy")

video_query = st.sidebar.text_area("Video:", key="1")
n_query = st.sidebar.text_area("Frame ID:", key="2")
n_range = st.sidebar.text_input("Frame IDs Range:", 10, key="3")
frame_query = st.sidebar.text_area("Frame index:", key="4")
frame_range = st.sidebar.text_input("Index Range:", 10, key="5")

dataset = loadKeyframes()


if text_query:
    handleOcrQuery(dataset, text_query)










