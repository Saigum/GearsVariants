git clone  https://github.com/Saigum/PerturbationPredictionUsingGeneNetworks.git
cd PerturbationPredictionUsingGeneNetworks
uv venv perturb
source perturb/bin/activate
uv pip install --no-cache-dir torch torch_geometric scanpy cell-gears ipykernel
scp ada.iiit.ac.in/share1/saigum/vcc_data.zip ./
unzip vcc_data.zip

