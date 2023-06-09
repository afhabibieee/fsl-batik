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
    parser.add_argument('--only_train',     action='store_true',                                help='Only train without tuning')
    parser.add_argument('--epochs',         type=int,   default=configs.EPOCHS,                 help='Number of passes of the entire training')
    parser.add_argument('--optimizer',      type=str,   default='Adam',                         help='eg. Adam, AdamW, SGD, ect')
    parser.add_argument('--learning_rate',  type=float, default=configs.LEARNING_RATE,          help='Hypreparameter use to govern the pace at which an algorithm updates')
    parser.add_argument('--weight_decay',   type=float, default=configs.WEIGHT_DECAY,           help='Regularization value to reduce the complexity of a model')
    parser.add_argument('--n_trials',       type=int,   default=configs.N_TRIALS,               help='Number of trials of the entire tuning')
    parser.add_argument('--backbone_name',  type=str,   default='resnet18',                     help='eg. resnet18, ect')
    parser.add_argument('--dropout',        type=float, default=configs.DROPOUT,                help='Rate value to randomly selected neurons are ignored during training')
    parser.add_argument('--use_softmax',    action='store_true',                                help='Flag indicating whether to apply softmax to the scores')
    parser.add_argument('--compile',        action='store_true',                                help='New in PyTorch 2.0 compile the model')
    parser.add_argument('--backend',        type=str,   default='eager',                        help='New in PyTorch 2.0 training & inference backends (eg. inductor, aot_ts_nvfuser, cudagraphs, ect')

    return parser.parse_args()