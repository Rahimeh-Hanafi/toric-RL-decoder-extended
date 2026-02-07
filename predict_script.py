import os
import time
import torch
import numpy as np

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

if DECODER_MODE == "gnn":
    NETWORK = GNNDecoder
    NUMBER_OF_ACTIONS = 12
    NETWORK_FILE_NAME = "GNNDecoder_multisize"
else:
    NETWORK = NN_17
    NUMBER_OF_ACTIONS = 3
    NETWORK_FILE_NAME = "Size_7_NN_17"

CKPT_PATH = f"network/{NETWORK_FILE_NAME}.pt"

PREDICTION_NOISES = [(0.01, 0.01, 0.08), (0.033, 0.033, 0.034)]
NUM_PREDICTIONS = 50
NUM_STEPS = 75

def main():
    timestamp = time.strftime("%y_%m_%d__%H_%M_%S__")
    OUT = f"data/prediction__{NETWORK_FILE_NAME}__{DECODER_MODE}__{timestamp}"
    os.makedirs(OUT, exist_ok=True)

    rl = RL(
        Network=NETWORK,
        Network_name=NETWORK_FILE_NAME,
        system_size=SYSTEM_SIZE,
        device=device,
        number_of_actions=NUMBER_OF_ACTIONS,
        p_meas=P_MEAS,
        stack_T=STACK_T,
        noise_conditioning=NOISE_CONDITIONING,
        decoder_mode=DECODER_MODE,
        gnn_k=8,
        use_global_pool=True,
    )

    out = rl.prediction(
        num_of_predictions=NUM_PREDICTIONS,
        num_of_steps=NUM_STEPS,
        PATH=CKPT_PATH,
        prediction_list_p_error=PREDICTION_NOISES,
        minimum_nbr_of_qubit_errors=int(SYSTEM_SIZE / 2) + 1,
        epsilon=0.0,
    )

    (err_corr, ground, avg_steps, mean_q, _, logical_success, noises, failure_rate) = out

    print("error_corrected:", err_corr)
    print("ground_state:", ground)
    print("logical_success:", logical_success)
    print("avg_steps:", avg_steps)
    print("mean_q:", mean_q)
    print("failure_rate:", failure_rate)

    np.savetxt(os.path.join(OUT, "summary.txt"),
               np.array([err_corr[0], ground[0], logical_success[0], avg_steps[0], mean_q[0], failure_rate], dtype=float))

if __name__ == "__main__":
    main()
