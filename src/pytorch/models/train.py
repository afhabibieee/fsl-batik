import sys
import os
import datetime
from dotenv import load_dotenv
import optuna
import mlflow
import torch
import torch._dynamo as dynamo
# import torch._dynamo.config
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.verbose = True
# os.environ['TORCHDYNAMO_REPRO_AFTER'] = 'dynamo'

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if current_path not in sys.path:
    sys.path.insert(0, current_path)

from configs import DEVICE, MODEL_CHECKPOINT_DIR
from data.fewshotdataloader import generate_loader
from models.io_utils import train_args
from models.protonet import PrototypicalNetwork
from models.utils import get_experiment_id
from models.utils import train_per_epoch

def objective(
    trial,
    train_loader, val_loader,
    backbone_name, compile, backend,
    mode, epochs
):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): A trial is a process of evaluating an objective function.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        val_loader (torch.utils.data.DataLoader): The data loader for validation data.
        backbone_name (str): The name of the backbone network.
        compile (bool): If True, compile the model using torch.jit.
        backend (str): The backend used when compiling with torch.jit.
        mode (str): The mode of operation, either 'training' or 'tuning'.
        epochs (int): The number of epochs to train for.
    
    Returns:
        float: The accuracy of the model on the validation set.
    """
    search_params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD']),
        'weight_decay': trial.suggest_loguniform('weight_decay', 1e-7, 1e-3),
        'dropout': trial.suggest_uniform('dropout', 0.0, 0.7)
    }

    model = PrototypicalNetwork(backbone_name, search_params['dropout']).to(DEVICE)
    model = torch.compile(model, backend=backend) if compile else model

    accuracy = fit_model(mode, search_params, train_loader, val_loader, model, epochs, trial=trial)
    return accuracy


def fit_model(mode, search_params, train_loader, val_loader, model, epochs, trial=None):
    """
    Fit a model using the given parameters and data loaders.

    Args:
        mode (str): The mode of operation, either 'training' or 'tuning'.
        search_params (dict): The hyperparameters to use for training.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        val_loader (torch.utils.data.DataLoader): The data loader for validation data.
        model (torch.nn.Module): The model to train.
        epochs (int): The number of epochs to train for.
        trial (optuna.Trial, optional): A trial is a process of evaluating an objective function.
    
    Returns:
        float: The validation accuracy for 'tuning' mode.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, search_params['optimizer'])(
        model.parameters(), lr=search_params['learning_rate'], weight_decay=search_params['weight_decay']
    )

    best_val_acc = 0.0
    for epoch in range(1, epochs+1):
        train_loss, val_loss, train_acc, val_acc, train_epoch_time, val_epoch_time = train_per_epoch(
            model, criterion, optimizer, epoch,
            train_loader, val_loader
        )

        if mode == 'training':
            print(
                '\nloss: {:.4f} - '.format(train_loss),
                'val_loss: {:.4f} - '.format(val_loss),
                'acc: {:.4f} - '.format(train_acc),
                'val_acc: {:.4f}\n'.format(val_acc),
                'train_time: {:.4f} - '.format(train_epoch_time),
                'val_time: {:.4f}\n'.format(val_epoch_time)
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if not os.path.exists(MODEL_CHECKPOINT_DIR):
                    os.makedirs(MODEL_CHECKPOINT_DIR)
                torch.save(model.state_dict(), os.path.join(MODEL_CHECKPOINT_DIR, 'model.pt'))
                mlflow.log_artifact(os.path.join(MODEL_CHECKPOINT_DIR, 'model.pt'), artifact_path='model')
                print("Yeay! we found a new best model :')\n")

            mlflow.log_metrics(
                {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_time': train_epoch_time,
                    'val_time': val_epoch_time
                },
                step=epoch
            )
        else:
            # Add prune mechanism
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if mode == 'tuning':
        return val_acc

def main():
    """
    Main function to run the training and hyperparameter tuning.
    """
    
    load_dotenv()
    if DEVICE.type.lower() == 'cuda':
        torch.multiprocessing.set_start_method('spawn')

    # Set the device globally
    torch.set_default_device(DEVICE)

    today = datetime.date.today().strftime('%Y-%m-%d')
    mlflow.set_experiment(today)

    params = train_args()

    mlflow.end_run()
    with mlflow.start_run(experiment_id=get_experiment_id(today)):
        train_loader = generate_loader(
            'train',
            image_size=params.img_size,
            n_way=params.n_way,
            n_shot=params.n_shot,
            n_query=params.n_query,
            n_task=params.n_train_task,
            n_workers=params.n_workers,
            train_size=params.train_size,
            seed=params.seed_class
        )

        val_loader = generate_loader(
            'val',
            image_size=params.img_size,
            n_way=params.n_way,
            n_shot=params.n_shot,
            n_query=params.n_query,
            n_task=params.n_val_task,
            n_workers=params.n_workers
        )

        print('\nData loader was generated successfully.\n')

        print('Hyperparameter tuning begins...\n')
        study = optuna.create_study(
            direction='maximize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(
            lambda trial: objective(
                trial, train_loader, val_loader,
                params.backbone_name, params.compile, params.backend,
                'tuning', params.epochs
            ),
            n_trials=params.epochs
        )
        best_trial = study.best_trial

        print('Model training begins...\n')

        params_dict = vars(params)
        params_dict['device'] = DEVICE.type.lower()
        params_dict.update(best_trial.params)
        mlflow.log_params(params_dict)

        model = PrototypicalNetwork(
            params.backbone_name,
            params_dict['dropout'],
            use_softmax=params.use_softmax
        ).to(DEVICE)
        model = torch.compile(model, backend=params.backend) if params.compile else model
        
        fit_model('training', best_trial, train_loader, val_loader, model, params.epochs)

    mlflow.end_run()

if __name__=='__main__':
    main()