import configs
import argparse

def train_args():
    parser = argparse.ArgumentParser(description='training few-shot model')

    parser.add_argument('--img_size',       type=int,   default=configs.IMAGE_SIZE,             help='Set image size')
    parser.add_argument('--train_size',     type=float, default=configs.TRAIN_SIZE,             help='Percentage of training classes')
    parser.add_argument('--seed_class',     type=int,   default=configs.RANDOM_SEED,            help='Random seed for shuffling the train, val, and test classes')
    parser.add_argument('--n_way',          type=int,   default=configs.N_WAY,                  help='Class num to classify')
    parser.add_argument('--n_shot',         type=int,   default=configs.N_SHOT,                 help='Number of labeled data in each class')
    parser.add_argument('--n_query',        type=int,   default=configs.N_SHOT,                 help='Number of query image each task')
    parser.add_argument('--n_train_task',   type=int,   default=configs.N_TRAINING_EPISODES,    help='Number of task episodes during meta training')
    parser.add_argument('--n_val_task',     type=int,   default=configs.N_VALIDATION_TASK,      help='Number of task for meta validation')
    parser.add_argument('--n_workers',      type=int,   default=configs.N_WORKERS,              help='Number of concurrent processing')
    parser.add_argument('--epochs',         type=int,   default=configs.EPOCHS,                 help='Number of passes of the entire training')
    parser.add_argument('--lr',             type=float, default=configs.LEARNING_RATE,          help='Step size at each iteration while moving toward a minimum of a loss function')
    parser.add_argument('--wd',             type=float, default=configs.WEIGHT_DECAY,           help='Decreasing the learning rate during training')
    parser.add_argument('--backbone_name',  type=str,   default=None,                           help='eg: resnet18, ect')
    parser.add_argument('--dropout',        type=float, default=None,                           help='Randomly zeroes some of the elements before flatten on backbone')
    parser.add_argument('--use_softmax',    type=bool,  default=False,                          help='Flag indicating whether to apply softmax to the scores')
    parser.add_argument('--compile',        type=bool,  default=True,                           help='New in PyTorch 2.0 compile the model')

    return parser.parse_args()