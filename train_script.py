import os
import time
import torch

from src.RL import RL
from NN import NN_17
from GNN import GNNDecoder

DECODER_MODE = "grid"  # "grid" or "gnn"

SYSTEM_SIZE = 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Novelty #1
P_MEAS = 0.02
STACK_T = 4

# Novelty #2
NOISE_CONDITIONING = True
PX_PY_PZ_LIST = [
    (0.033, 0.033, 0.034),
    (0.01,  0.01,  0.08),
    (0.08,  0.01,  0.01),
]

# Novelty #3 (grid path)
USE_GLOBAL_POOL = True

if DECODER_MODE == "gnn":
    NETWORK = GNNDecoder
    NUMBER_OF_ACTIONS = 12
    NETWORK_FILE_NAME = "GNNDecoder_multisize"
else:
    NETWORK = NN_17
    NUMBER_OF_ACTIONS = 3
    NETWORK_FILE_NAME = "Size_7_NN_17"

def main():
    timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
    OUT = f"data/training__{NETWORK_FILE_NAME}__{DECODER_MODE}__{timestamp}"
    os.makedirs(OUT, exist_ok=True)

    rl = RL(
        Network=NETWORK,
        Network_name=NETWORK_FILE_NAME,
        system_size=SYSTEM_SIZE,
        p_error=0.1,
        replay_memory_capacity=20000,
        learning_rate=0.00025,
        discount_factor=0.95,
        number_of_actions=NUMBER_OF_ACTIONS,
        max_nbr_actions_per_episode=75,
        device=device,
        replay_memory="proportional",
        p_meas=P_MEAS,
        stack_T=STACK_T,
        noise_conditioning=NOISE_CONDITIONING,
        px_py_pz_list=PX_PY_PZ_LIST,
        decoder_mode=DECODER_MODE,
        gnn_k=8,
        use_global_pool=USE_GLOBAL_POOL,
    )

    rl.train(
        training_steps=5000,
        target_update=200,
        batch_size=32,
        replay_start_size=500,
        optimizer="Adam",
        minimum_nbr_of_qubit_errors=0,
    )

    # save
    os.makedirs("network", exist_ok=True)
    rl.save_network(f"network/{NETWORK_FILE_NAME}.pt")


if __name__ == "__main__":
    main()
