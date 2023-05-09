import os
import sys
import random
import json
from pathlib import Path
from fewshotdataset import ImageDataset, FewShotBatchSampler
from torch.utils.data import DataLoader

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_path not in sys.path:
    sys.path.append(current_path)
from configs import BATIK_SPECS_DIR, IMAGE_SIZE, N_WAY, N_SHOT, N_QUERY, N_TRAINING_EPISODES, N_WORKERS

def name_roots(motifs):
    """
    Generate a dictionary containing class name and their corresponding class roots.

    Parameters:
        motifs (list): A list of motif names.

    Returns:
        dict: A dictionary with two keys - 'class_names' and 'class_roots'.
              The value of 'class_names' is the list of motif names passed as input.
              The value of 'class_roots' is a list of class roots, obtained by the path each motif name.
    """
    return {
        'class_names': motifs,
        'class_roots': [os.path.join(BATIK_SPECS_DIR, motif) for motif in motifs]
    }

def create_jsonfile(train_size, seed):
    """
    Generate train, validation, and test JSON files for batik data.
    
    Parameters:
        train_size (float): The proportion of motifs to be included in the training set.
        seed (int): The seed value for randomization.

    Returns:
        None
    """
    random.seed(seed)

    motifs = sorted(os.listdir(BATIK_SPECS_DIR))
    n_train_class = int(round(train_size*len(motifs)))
    train_motifs = random.sample(motifs, k=n_train_class)

    val_test = list(set(motifs).difference(train_motifs))
    val_motifs = val_test[:round(len(val_test)/2)]
    test_motifs = val_test[round(len(val_test)/2):]

    files = ['train.json', 'val.json', 'test.json']
    data = [train_motifs, val_motifs, test_motifs]

    for file, content in zip(files, data):
        with open(os.path.join('data', file), 'w') as outfile:
            json.dump(name_roots(content), outfile, indent=4)

    random.seed(None)

def batik(split, image_size, **kwargs):
    """
    Create an ImageDataset object for the batik motifs dataset.

    Args:
        split (str): The split of the dataset (e.g., 'train', 'val', 'test').
        image_size (int): The desired size of the images.
        **kwargs: Additional keyword arguments to ve pass to the ImageDataset constructor.

    Return:
        ImageDataset: An ImageDataset object for the specified split of the batik motifs dataset.

    Raises:
        ValueError: If the specs file is not in JSON format or cannot be found.
    """
    specs_file = Path(BATIK_SPECS_DIR) / '{}.json'.format(split)
    if specs_file.suffix != '.json':
        raise ValueError('Requires specs in a JSON file')
    elif specs_file.is_file():
        return ImageDataset(specs_file=specs_file, image_size=image_size, **kwargs)
    else:
        raise ValueError(
            "Couldn't find specs file {} in {}".format(specs_file.name, Path(BATIK_SPECS_DIR))
        )

def generate_loader(
    split,
    image_size=IMAGE_SIZE,
    n_way=N_WAY,
    n_shot=N_SHOT,
    n_query=N_QUERY,
    n_task=N_TRAINING_EPISODES,
    n_workers=N_WORKERS,
    **kwargs
):
    """
    Generate a DataLoader for few-shot learning using the batik dataset.

    Args:
        split (str): The split of the dataset to generate the loader for (e.g., 'train', 'val', 'test').
        image_size (int, optional): The desired size of the image.
        n_way (int, optional): Number of classes in each task.
        n_shot (int, optional): Number of samples per class in the support set.
        n_query (int, optional): Number of samples per class in the query set.
        n_task (int, optional): Number of task (iterations).
        n_workers (int, optional): Number of worker processes to use for data loading.
        **kwargs: Additional keyword arguments to be passed to the ImageDataset constructor.
    """
    training = True if split=='train' else False
    dataset = batik(split=split, image_size=image_size, training=training, **kwargs)
    batch_sampler = FewShotBatchSampler(dataset, n_way, n_shot, n_query, n_task)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=batch_sampler.collate_fn
    )