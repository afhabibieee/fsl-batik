import torch
from torchviz import make_dot
import cv2
from gradcam import GradCam
import mlflow
import os
from configs import DEVICE, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, SUPPORT_PATH, MODEL_DIR
import random
from torch.utils.data import Dataset
from embeddings import Embedding, EmbeddingPostProcess
import numpy as np
import torchvision.transforms as T
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import b2sdk.v2 as b2
import streamlit as st

# @st.cache_resource
def class_activation_map(_embedding, _query_image_tensor):
    # Load model
    pretrained = torch.nn.Sequential(*(list(_embedding.embedding.backbone.children())[:-2]))
    pretrained.eval()

    # Set the target layer (last layer before avg/global pool or layer you're interested in)
    target_layer = [name for name, _ in pretrained.named_modules()][-1]
    
    # Initialize GradCam
    grad_cam = GradCam(pretrained, target_layer)
    # Forward pass
    query_image_tensor = _query_image_tensor.div(225)
    query_image_tensor = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(query_image_tensor)
    output = grad_cam.forward(query_image_tensor)
    # Backward pass
    output.backward(torch.ones_like(output))

    # Get gradients and activations
    gradients = grad_cam.get_activations_gradient()
    activations = grad_cam.get_activations()

    # Compute the weights for the final linear combination
    alpha_k = torch.mean(gradients, dim=(2, 3), keepdim=True)

    # Perform the linear combination to compute the CAM
    cam = torch.sum(alpha_k * activations, dim=1, keepdim=True)
    # ReLU on the CAM (only keep the positive values)
    cam = torch.nn.functional.relu(cam)
    # Normalize the CAM to be between 0 and 1
    cam -= cam.min()
    cam /= cam.max()
    # Detach and move the CAM to CPU
    cam = cam.detach().cpu().numpy()
    # Resize the cam to the size of the query image
    cam = cv2.resize(cam[0, 0, :, :], (query_image_tensor.shape[2], query_image_tensor.shape[3]))

    return cam

@st.cache_resource
def get_embedding(run_id, backend):
    artifact_path = os.path.join(MODEL_DIR, run_id)
    if not os.path.exists(artifact_path):
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path='model',
            dst_path=artifact_path
        )
    embedding = Embedding('convnext_tiny')
    embedding = torch.compile(embedding, backend=backend)
    embedding.load_state_dict(torch.load(os.path.join(artifact_path, 'model/model.pt'), map_location=DEVICE))
    embedding = embedding.eval()
    embedding_post_process = EmbeddingPostProcess(embedding)
    embedding_post_process = embedding_post_process.to(DEVICE).eval()
    return embedding_post_process

@st.cache_resource
def auth_load_bucket(id, app_key):
    info = b2.InMemoryAccountInfo()
    b2_api = b2.B2Api(info)
    b2_api.authorize_account('production', id, app_key)

    bucket_name = 'allBatik'
    bucket = b2_api.get_bucket_by_name(bucket_name)
    return bucket

@st.cache_data
def download_support_set(_bucket, selected_batik):
    folder = 'batik'
    n_files = 20
    for batik in selected_batik:
        # Create the local directory if it does not exist
        subfolder = os.path.join(folder, batik)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder, exist_ok=True)

            # Get a list of file names in the subfolder
            files = [file_info for file_info, _ in _bucket.ls(folder_to_list=subfolder, recursive=True)]

            # Randomly select and download a certain number of files from the subfolder
            selected_files = random.sample(files, min(n_files, len(files)))
            for file_info in selected_files:
                # Download the file by its ID
                downloaded_file = _bucket.download_file_by_id(file_info.id_)
                # Save the downloaded file to local file path
                downloaded_file.save_to(file_info.file_name, 'wb+')

class SupportSets(Dataset):
    def __init__(self, path, X, y, label_encoded):
        self.path = path
        self.X = X
        self.y = y
        self.label_encoded = label_encoded

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.path, str(self.y[index]), str(self.X[index])))
        image = image[:,:,::-1]
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = torch.from_numpy(image.transpose((2,0,1))).contiguous().to(dtype=torch.float32)

        label = self.label_encoded[self.y[index]]

        return image, label
    
    def __len__(self):
        return len(self.X)

@st.cache_data
def get_support_images_and_labels(selected_batik, N_SHOT):
    batik_path = [os.path.join(SUPPORT_PATH, batik) for batik in selected_batik]
    support_images, support_labels = [], []
    for path in batik_path:
        temp_images = random.sample(os.listdir(path), k=N_SHOT)
        temp_class = [path.split('/')[-1]]*N_SHOT
        support_images.extend(temp_images)
        support_labels.extend(temp_class)

    # Categorical encoding map
    label_encoded = dict((v, k) for k, v in enumerate(selected_batik))
    label_decoded = dict(enumerate(selected_batik))

    # for models with post processes
    support_set = SupportSets(SUPPORT_PATH, support_images, support_labels, label_encoded)

    support_images = torch.cat([x[0].unsqueeze(0) for x in support_set])
    support_labels = torch.tensor([x[1] for x in support_set])

    return support_images, support_labels, label_decoded

@st.cache_data
def process_queries(rgb_images):
    queries = []
    for rgb_image in rgb_images:
        query = torch.from_numpy(rgb_image.transpose((2,0,1))).contiguous().to(dtype=torch.float32)
        query = query.unsqueeze(0)
        queries.append(query)
    return torch.cat(queries)

def compute_prototype_and_logits(support_labels, support_features, query_features):
    N_WAY = torch.unique(support_labels).shape[0]
    prototypes = torch.cat([
        support_features[torch.nonzero(support_labels==label)].mean(0)
        for label in range(N_WAY)
    ])

    dist = torch.cdist(query_features, prototypes)
    scores = -dist
    scores = scores.detach()
    prob = scores.softmax(-1).data

    return prob

def calc_least_confidence(prob):
    most_conf = torch.max(prob)
    num_labels = prob.numel()

    numerator = (num_labels * (1-most_conf))
    dominator = (num_labels - 1)

    return numerator / dominator

def calc_margin_of_confidence(prob):
    prob, _ = torch.sort(prob, descending=True)
    difference = (prob.data[0] - prob.data[1])
    return 1 - difference

def calc_ratio_of_confidence(prob):
    prob, _ = torch.sort(prob, descending=True)

    return (prob.data[1] / prob.data[0])

def calc_entropy(prob):
    prbslogs = prob * torch.log2(prob)

    numerator = 0 - torch.sum(prbslogs)
    denominator = np.log2(prob.numel())

    return numerator / denominator

# @st.cache_data
def inference(
    _support_images,
    _support_labels,
    _query_tensors,
    _pretrained,
    label_decoded,
    mode='recognition'
):
    support_features = _pretrained(_support_images.to(DEVICE))
    query_features = _pretrained(_query_tensors.to(DEVICE))    

    prob = compute_prototype_and_logits(
        _support_labels.to(DEVICE), support_features, query_features
    )

    most_conf, predicted_labels = torch.max(prob, 1)
    predicted_classes = [
        label_decoded[predicted_labels[query].item()]
        for query in range(len(_query_tensors))
    ]

    # Uncertainty calculation
    least_confs = list(map(calc_least_confidence, prob))
    margin_confs = list(map(calc_margin_of_confidence, prob))
    ratio_confs = list(map(calc_ratio_of_confidence, prob))
    entorpies = list(map(calc_entropy, prob))

    # Class activatin map
    cams = class_activation_map(_pretrained, _query_tensors) if mode=='recognition' else None

    return most_conf, predicted_classes, least_confs, margin_confs, ratio_confs, entorpies, cams

@st.cache_data
def get_table_experiment():
    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
    runs = mlflow.search_runs(experiment_ids=all_experiments)
    runs.set_index('run_id', inplace=True)
    unnecessary_column = ['start_time', 'end_time', 'experiment_id', 'artifact_uri', 'tags.mlflow.source.name', 'tags.mlflow.source.type', 'tags.mlflow.source.git.commit', 'status', ]
    runs.drop(columns=unnecessary_column, inplace=True)

    return runs

@st.cache_data
def show_experiment_metrics(run_id):
    mlflow_client = MlflowClient()
    metric_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    metrics_data = pd.DataFrame()
    for metric_key in metric_keys:
        metrics = mlflow_client.get_metric_history(run_id, metric_key)
        metrics_data[metric_key] = [metric.value for metric in metrics]

    metrics_data.rename(index=lambda x: x+1, inplace=True)

    return metrics_data

def display_model_graph(pretained, run_id):
    shape = torch.Size([1, 3, 84, 84])
    x = torch.zeros(shape)
    y = pretained(x)

    # Create a directed graph of the model
    dot = make_dot(y.mean(), params=dict(pretained.named_parameters()))
    
    # Save the graph to a file
    if not os.path.exists(f'model-registry/{run_id}/model/model.png'):
        dot.format = 'png'
        dot.render(filename=f'model-registry/{run_id}/model/model')

def zipdir(path, ziph):
    #ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
            )