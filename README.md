# ML ENDPOİNT SERVICE AND MODEL REGISTRY

ML model inference service solution and how to make model registry is included in this repo. To run this solution please follow steps below

# Prerequisites for Testing, Training, and Preprocessing
To ensure a smooth setup and execution of our testing, training, and preprocessing workflows, the following dependencies must be installed:
- Operating System: Windows (WSL required)
- Docker Desktop: Necessary for containerized environments
- WSL (Windows Subsystem for Linux): Required for compatibility and execution
- Python: Version 3.12 must be installed for preprocessing and related tasks
- Git: Required for version control
- DVC (Data Version Control): Needs Git to function properly
Before proceeding, ensure all tools are properly installed and configured.
# How to run
* Please use a windows 11 machine for docker service host
* In order to run the solution first of all make sure that  Docker version 24.0.6 i running on host machine
* In order to run the solution  make sure that python 3.13 or higher is installed in your env
## Step 1 
* Vlone this repo with and change working directory to this repo
```
git clone https://github.com/Atsyurt/mlops_tracking_server.git
cd mlops_tracking_server


```
## step 2
* Install python dependencies
```
pip install -r requirements.txt
```
* Download the data using download_data.py inside scripts folder(works for windows cmd.exe, change this if you use different terminal)
```
python scripts\download_data.py
```

## step 3
* Please note that i used mlflow for mlops system
* First of all You should build the docker image with this cmd in order to run mlflow tracking,registry service and api service
```
docker compose build
```
## step 4
* run docker containers with this cmd  in order to start services
```
docker compose up

```
* Right now your mlflow service should be accessible from
[Mlflow tracking ](http://localhost:5000)

![Alt text](img/step2_mlflow_service.png)