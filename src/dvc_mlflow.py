import os
import argparse

USERNAME = 'afhabibieee'
REPONAME = 'fsl-batik'
DAGSHUB_URL = f'https://dagshub.com/{USERNAME}/{REPONAME}'
TOKEN = ''

# Write .env file for MLFlow tracking
def write_dotenv():

    with open(".env", "w") as f:
        mlflow_uri = f'{DAGSHUB_URL}.mlflow'
        f.write(f"MLFLOW_TRACKING_URI={mlflow_uri}\n")
        f.write(f"MLFLOW_TRACKING_USERNAME={USERNAME}\n")
        f.write(f"MLFLOW_TRACKING_PASSWORD={TOKEN}")
    
    os.system("echo '.env' >> .gitignore")

# Init DVC and configure MLFlow
def init_dvc():
    write_dotenv()
    if os.path.exists('.dvc'):
        os.system('rm -f .dvc/config')
        os.system('dvc init -f')
    else:
        os.system('dvc init')

    # Configure the remote with DVC
    dvc_uri = f'{DAGSHUB_URL}.dvc'
    os.system(f'dvc remote add myremote {dvc_uri}')
    os.system(f'dvc remote modify myremote --local auth basic')
    os.system(f'dvc remote modify myremote --local user {USERNAME}')
    os.system(f'dvc remote modify myremote --local password {TOKEN}')

# Pull DVC and configure MLFlow
def pull_dvc():
    write_dotenv()

    # DVC user configuration
    os.system(f'dvc remote modify myremote --local user {USERNAME}')
    os.system(f'dvc remote modify myremote --local password {TOKEN}')
    
    os.system('dvc pull -r myremote')
    # Make sure that all files were pulled
    os.system('dvc pull -r myremote')

def main():
    parser = argparse.ArgumentParser(description='Init or Pull DVC data and Configure MLFlow')
    parser.add_argument('--mode', default=None, help='init/pull')
    parser.add_argument('--token', default=None, help='DagsHub token')
    params = parser.parse_args()

    global TOKEN
    TOKEN = params.token

    if params.mode == 'init':
        init_dvc()
    elif params.mode == 'pull':
        pull_dvc()
    elif params.mode == None:
        ValueError('Enter the mode to run')
    else:
        ValueError('The arg entered is not available')

if __name__=='__main__':
    main()