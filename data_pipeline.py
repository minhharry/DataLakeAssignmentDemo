# Importing libraries
import datetime
import glob
import os
import subprocess

import cv2
import lancedb
import open_clip
import pandas as pd
import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def handle_video_file(dir_path) :
    def get_video_fps(video_path) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    all_files = glob.glob(dir_path+"/**/*", recursive=True)
    video_files = [file_path for file_path in all_files if file_path.endswith(".mkv") or file_path.endswith(".mp4")]
    for video_path in video_files:
        print(f"Extracting keyframes for {video_path}")
        command = [
            "scenedetect", 
            "-i", video_path, 
            "detect-adaptive", 
            "save-images", 
            "--filename", f"$VIDEO_NAME/image-$IMAGE_NUMBER-scene-$SCENE_NUMBER-frame-$FRAME_NUMBER-fps-{get_video_fps(video_path):.2f}",
            "-o", 'output'
        ]
        result = subprocess.run(command)

        # Move video file to output directory
        video_name = video_path.rsplit('\\', 1)[-1]
        video_name_without_ext = video_name.rsplit('.', 1)[0]
        os.rename(video_path, f"output/{video_name_without_ext}/{video_name}")
        
# Dataset and dataloader
class KeyframesDataset(Dataset):
    def __init__(self, preprocess, path):
        self.PATH = path
        print(f"Image path for dataset: {self.PATH}")
        self.images_paths = glob.glob(f"{self.PATH}\\**\\*.jpg", recursive=True)
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.images_paths[idx]))
        video_name = self.images_paths[idx].rsplit('\\', 2)[1]
        image_name = self.images_paths[idx].rsplit('\\', 1)[1]
        frame_idx = image_name.rsplit('-', 3)[1]
        fps = image_name.rsplit('-', 1)[1].rsplit('.', 1)[0]
        return image, video_name, image_name, int(frame_idx), self.images_paths[idx], float(fps)
    
def seconds_to_hhmmss(seconds):
    time_obj = datetime.timedelta(seconds=seconds)
    return str(time_obj)

if __name__ == '__main__':
    handle_video_file('input')

    # Load CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    model.to(device)

    dataset = KeyframesDataset(preprocess, "output")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=11, persistent_workers=True)
    print("Number of images: ", len(dataset))
    print("Number of batches: ", len(dataloader))

    # Create lancedb instance
    lancedb_instance = lancedb.connect("database.lance")
    TABLE_NAME = "patch14v2_openclip"
    if TABLE_NAME in lancedb_instance.table_names():
        database = lancedb_instance[TABLE_NAME]
        print(f"Warning: Table {TABLE_NAME} already exists!")
    else:
        schema = pa.schema([
            pa.field("embedding", pa.list_(pa.float32(), 1024)),
            pa.field("video_name", pa.string()),
            pa.field("image_name", pa.string()),
            pa.field("frame_idx", pa.int32()),
            pa.field("path", pa.string()),
            pa.field("time", pa.string()),
        ])
        lancedb_instance.create_table(TABLE_NAME, schema=schema)
        database = lancedb_instance[TABLE_NAME]

    LEN_DATALOADER = len(dataloader)
    SAVE_EVERY = int(0.05 * LEN_DATALOADER)
    if SAVE_EVERY == 0:
        SAVE_EVERY = 1000000000

    df = pd.DataFrame(columns=['embedding', 'video_name', 'image_name', 'frame_idx', 'path', 'time'])
    for i, (images, video_names, image_names, frame_idxs, paths, fpss) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            embeddings = model.encode_image(images)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.squeeze().cpu().numpy()
        data = {
            'embedding': [],
            'video_name': [],
            'image_name': [],
            'frame_idx': [],
            'path': [],
            'time': []
        }
        for embedding, video_name, image_name, frame_idx, path, fps in zip(embeddings, video_names, image_names, frame_idxs, paths, fpss):
            data['embedding'].append(embedding)
            data['video_name'].append(video_name)
            data['image_name'].append(image_name)
            data['frame_idx'].append(int(frame_idx))
            data['path'].append(path)
            data['time'].append(seconds_to_hhmmss(int(frame_idx) / float(fps)))

        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
        if (i + 1) % SAVE_EVERY == 0 or i + 1 == LEN_DATALOADER:
            lancedb_instance[TABLE_NAME].add(df)
            df = pd.DataFrame(columns=['embedding', 'video_name', 'image_name', 'frame_idx', 'path', 'time'])
            print(f"Saved embeddings for batch {i+1}/{LEN_DATALOADER}")