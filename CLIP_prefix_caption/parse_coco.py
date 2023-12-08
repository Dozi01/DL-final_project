import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse


def load_checkpoint(file_path):
    if os.path.exists(file_path):
        print('checkpoint loaded')
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_checkpoint(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)



def main(clip_model_type: str, data_path:'str', out_path: 'str', start_index: 'int'):
    device = torch.device('cuda:0')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"{out_path}/oscar_split_{clip_model_name}_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    with open(f"{data_path}/annotations/train_caption.json", 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    
    checkpoint = load_checkpoint(out_path)
    checkpoint_emb_shape = checkpoint['clip_embedding'].shape
    print(f'checkpoint shape: {checkpoint_emb_shape}')

    all_embeddings = checkpoint['clip_embedding'] if checkpoint else []
    all_captions = checkpoint['captions'] if checkpoint else []
    new_embeddings = []

    for i in tqdm(range(start_index, len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"{data_path}/train2017/{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"{data_path}/val2017/{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
                
        d["clip_embedding"] = i
        new_embeddings.append(prefix)
        all_captions.append(d)
        
        if (i + 1) % 10000 == 0:
            print(f'index {i} saved')
            all_embeddings = torch.cat((all_embeddings, torch.cat(new_embeddings, dim=0)), dim=0)
            new_embeddings = []
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": all_embeddings, "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        print(f'index {i} saved')
        all_embeddings = torch.cat((all_embeddings, torch.cat(new_embeddings, dim=0)), dim=0)
        pickle.dump({"clip_embedding": all_embeddings, "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--data_path', type=str, default='./data/coco/')
    parser.add_argument('--out_path', type=str, default='./data/coco/')
    parser.add_argument('--start_index', type=int, default=0)


    args = parser.parse_args()
    exit(main(args.clip_model_type, args.data_path, args.out_path, args.start_index))
