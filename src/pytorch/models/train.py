import sys
import os
import datetime
from dotenv import load_dotenv
import mlflow
import torch
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

def main():
    load_dotenv()
    today = datetime.date.today().strftime('%Y-%m-%d')
    mlflow.set_experiment(today)

    params = train_args()

    mlflow.end_run()
    with mlflow.start_run(experiment_id=get_experiment_id(today)):
        # Set the device globally
        torch.set_default_device(DEVICE)
        
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

        params_dict = vars(params)
        params_dict['device'] = DEVICE.type.upper()
        mlflow.log_params(params_dict)

        model = PrototypicalNetwork(
            params.backbone_name,
            params.dropout,
            use_softmax=params.use_softmax
        )
        model = torch.compile(model, backend='eager') if params.compile else model

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.wd)

        print('Model training begins...\n')

        best_val_acc = 0.0
        for epoch in range(1, params.epochs+1):
            train_loss, val_loss, train_acc, val_acc, train_epoch_time, val_epoch_time = train_per_epoch(
                model, criterion, optimizer, epoch,
                train_loader, val_loader
            )

            print(
                'Epoch: {}/{}\n'.format(epoch, params.epochs),
                'loss: {:.4f} - '.format(train_loss),
                'val_loss: {:.4f} - '.format(val_loss),
                'acc: {:.4f} - '.format(train_acc),
                'val_acc: {:.4f}\n'.format(val_acc),
                'train_epoch_time: {:.4f} - '.format(train_epoch_time),
                'val_epoch_time: {:.4f}\n'.format(val_epoch_time)
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # mlflow.pytorch.log_model(model, 'model')
                torch.save(model.state_dict(), os.path.join(MODEL_CHECKPOINT_DIR, 'model.pt'))
                mlflow.log_artifact(os.path.join(MODEL_CHECKPOINT_DIR, 'model.pt'), artifact_path='model')
                print("Yeay! we found a new best model :')\n")

            mlflow.log_metrics(
                {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_epoch_time': train_epoch_time,
                    'val_epoch_time': val_epoch_time
                },
                step=epoch
            )

if __name__=='__main__':
    main()