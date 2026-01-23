#!/bin/bash
#SBATCH --job-name=state_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gnode067
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=80:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log

mkdir -p /scratch/saigum
cd /scratch/saigum

git clone https://github.com/Saigum/stateExperiments.git
uv venv state
source state/bin/activate
uv pip install -e .
mkdir -p competition
scp ada.iiit.ac.in:/share1/saigum/competition_support_set.zip  competition/
mkdir -p se_model
scp ada.iiit.ac.in:/share1/saigum/protein_embeddings.zip ada.iiit.ac.in:/share1/saigum/config.yaml ada.iiit.ac.in:/share1/saigum/se600m_epoch4.ckpt se_model/
cd competition
unzip competition_support_set.zip
cd ..
state emb transform --model-folder /scratch/saigum/stateExperiments/se_model/ --input competition/competition_train.h5 --output competition/competition_train_embedded.h5 
