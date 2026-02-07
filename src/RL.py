import os
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from .toric_model import Toric_code
from .util import incremental_mean, convert_from_np_to_tensor, Transition
from .Replay_memory import Replay_memory_uniform, Replay_memory_prioritized

from NN import NN_11, NN_17
from ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from GNN import GNNDecoder


class RL:
    """
    CHANGED vs original:
    - decoder_mode: "grid" (original perspective-style) or "gnn" (Novelty #3).
    - Novelty #1: p_meas + stack_T (handled in env obs/history).
    - Novelty #2: noise_conditioning + biased noise list (px,py,pz).
    """

    def __init__(
        self,
        Network,
        Network_name: str,
        system_size: int,
        p_error: float = 0.1,
        replay_memory_capacity: int = 100000,
        learning_rate: float = 0.00025,
        discount_factor: float = 0.95,
        number_of_actions: int = 3,
        max_nbr_actions_per_episode: int = 50,
        device: str = "cpu",
        replay_memory: str = "uniform",
        # Novelty #1
        p_meas: float = 0.0,
        stack_T: int = 1,
        # Novelty #2
        noise_conditioning: bool = False,
        px_py_pz_list=None,
        # Novelty #3
        decoder_mode: str = "grid",
        gnn_k: int = 8,
        use_global_pool: bool = False,
    ):
        self.device = device
        self.system_size = int(system_size)
        self.grid_shift = self.system_size // 2

        self.discount_factor = float(discount_factor)
        self.number_of_actions = int(number_of_actions)
        self.learning_rate = float(learning_rate)
        self.max_nbr_actions_per_episode = int(max_nbr_actions_per_episode)

        self.p_error = float(p_error)
        self.p_meas = float(p_meas)
        self.stack_T = int(stack_T)

        self.noise_conditioning = bool(noise_conditioning)
        self.px_py_pz_list = px_py_pz_list

        self.decoder_mode = decoder_mode
        self.gnn_k = int(gnn_k)
        self.use_global_pool = bool(use_global_pool)

        self.toric = Toric_code(self.system_size, p_meas=self.p_meas, stack_T=self.stack_T, noise_conditioning=self.noise_conditioning)

        self.replay_memory_capacity = int(replay_memory_capacity)
        self.replay_memory = replay_memory
        if self.replay_memory == "proportional":
            self.memory = Replay_memory_prioritized(self.replay_memory_capacity, 0.6)
        else:
            self.memory = Replay_memory_uniform(self.replay_memory_capacity)

        self.network_name = Network_name
        self.network = Network

        # grid in_channels (2*stack_T + optional 3)
        self.in_channels = 2 * self.stack_T + (3 if self.noise_conditioning else 0)

        self.policy_net = self._build_network().to(self.device)
        self.target_net = deepcopy(self.policy_net).to(self.device)

    # -------------------------
    # Network build/load/save
    # -------------------------
    def _build_network(self):
        if self.decoder_mode == "gnn":
            node_feat_dim = 7 + (3 if self.noise_conditioning else 0)
            return GNNDecoder(
                system_size=self.system_size,
                number_of_actions=self.number_of_actions,
                device=self.device,
                node_feat_dim=node_feat_dim,
                edge_feat_dim=3,
                hidden_dim=128,
                num_layers=3,
            )

        # grid mode (try new signature first)
        try:
            return self.network(self.system_size, self.number_of_actions, self.device,
                                in_channels=self.in_channels, use_global_pool=self.use_global_pool)
        except TypeError:
            return self.network(self.system_size, self.number_of_actions, self.device)

    def save_network(self, PATH: str):
        payload = {
            "state_dict": self.policy_net.state_dict(),
            "config": {
                "decoder_mode": self.decoder_mode,
                "system_size": self.system_size,
                "number_of_actions": self.number_of_actions,
                "stack_T": self.stack_T,
                "p_meas": self.p_meas,
                "noise_conditioning": self.noise_conditioning,
                "use_global_pool": self.use_global_pool,
            },
        }
        torch.save(payload, PATH)

    def load_network(self, PATH: str):
        obj = torch.load(PATH, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            self.policy_net = self._build_network()
            self.policy_net.load_state_dict(obj["state_dict"], strict=True)
            self.policy_net = self.policy_net.to(self.device)
            self.target_net = deepcopy(self.policy_net).to(self.device)
            return
        if isinstance(obj, nn.Module):
            self.policy_net = obj.to(self.device)
            self.target_net = deepcopy(self.policy_net).to(self.device)
            return
        raise ValueError("Unknown checkpoint format.")

    # -------------------------
    # Episode noise sampler
    # -------------------------
    def _sample_episode_noise(self):
        if self.px_py_pz_list is not None and len(self.px_py_pz_list) > 0:
            px, py, pz = random.choice(self.px_py_pz_list)
            return float(px), float(py), float(pz), "biased"
        p = float(self.p_error)
        return p / 3.0, p / 3.0, p / 3.0, "depolarizing"

    # -------------------------
    # Reward
    # -------------------------
    def get_reward(self):
        if np.all(self.toric.next_state == 0):
            return 100.0
        return float(np.sum(self.toric.current_state) - np.sum(self.toric.next_state))

    # ============================================================
    # GRID MODE (original perspective-based selection + replay)
    # ============================================================
    def select_action_grid(self, epsilon: float):
        self.policy_net.eval()
        obs = self.toric.get_current_observation()  # [C,d,d]
        perspectives = self.toric.generate_perspective(self.grid_shift, obs)
        if len(perspectives) == 0:
            # fallback random
            qm = random.randint(0, 1)
            r = random.randrange(self.system_size)
            c = random.randrange(self.system_size)
            op = random.randint(1, 3)
            return (qm, r, c), op

        batch_persp = np.array([p.perspective for p in perspectives], dtype=np.float32)
        batch_pos = [p.position for p in perspectives]

        x = convert_from_np_to_tensor(batch_persp).to(self.device)

        if random.random() > epsilon:
            with torch.no_grad():
                q = self.policy_net(x).detach().cpu().numpy()  # [Np,3]
            row, col = np.where(q == np.max(q))
            pi = int(row[0])
            op = int(col[0]) + 1
            pos = batch_pos[pi]
        else:
            pi = random.randrange(len(perspectives))
            pos = batch_pos[pi]
            op = random.randint(1, 3)

        return pos, op

    def get_network_output_next_state_grid(self, batch_next_state: torch.Tensor, batch_size: int, action_index=None):
        """
        Original-style: for each next_state, regenerate perspectives and take max Q.
        FIXED: avoid torch scalar comparisons by converting to numpy.
        """
        self.target_net.eval()
        out_q = np.zeros(batch_size, dtype=float)
        for i in range(batch_size):
            s = batch_next_state[i].detach().cpu().numpy()
            if np.sum(s) == 0:
                continue
            perspectives = self.toric.generate_perspective(self.grid_shift, s)
            if len(perspectives) == 0:
                continue
            batch_p = np.array([p.perspective for p in perspectives], dtype=np.float32)
            x = convert_from_np_to_tensor(batch_p).to(self.device)
            with torch.no_grad():
                q = self.target_net(x).detach().cpu().numpy()  # [Np,3]
            if action_index is None or action_index[i] is None:
                out_q[i] = float(np.max(q))
            else:
                out_q[i] = float(np.max(q[:, int(action_index[i])]))
        return convert_from_np_to_tensor(out_q)

    def experience_replay_grid(self, criterion, optimizer, batch_size: int):
        self.policy_net.train()
        self.target_net.eval()

        transitions, weights, indices = self.memory.sample(batch_size, 0.4)
        if transitions is None:
            return

        mini = Transition(*zip(*transitions))

        # states are perspectives [C,d,d]
        s = convert_from_np_to_tensor(np.stack(mini.state, axis=0)).to(self.device)
        ns = convert_from_np_to_tensor(np.stack(mini.next_state, axis=0)).to(self.device)

        # actions are centered Action namedtuples: action in 1..3
        a = np.array([t.action for t in mini.action], dtype=int) - 1
        a = torch.tensor(a, dtype=torch.long, device=self.device)

        r = torch.tensor(np.array(mini.reward, dtype=float), dtype=torch.float32, device=self.device)
        term = torch.tensor(np.array(mini.terminal, dtype=float), dtype=torch.float32, device=self.device)

        q_all = self.policy_net(s)
        q = q_all.gather(1, a.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            qn = self.get_network_output_next_state_grid(ns, batch_size).to(self.device)

        y = r + term * self.discount_factor * qn
        loss = criterion(y, q)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    # ============================================================
    # GNN MODE (active defect + move)
    # ============================================================
    def _valid_action_mask(self, node_type: torch.Tensor):
        """
        Optional stability improvement:
        - Vertex defects: disallow X
        - Plaquette defects: disallow Z
        """
        N = node_type.numel()
        A = self.number_of_actions  # 12
        mask = torch.ones((N, A), dtype=torch.bool, device=node_type.device)
        pauli_id = torch.arange(A, device=node_type.device) % 3  # 0=X,1=Y,2=Z
        is_vertex = (node_type == 0).unsqueeze(1)
        is_plaq = (node_type == 1).unsqueeze(1)
        mask &= ~(is_vertex & (pauli_id == 0).unsqueeze(0))
        mask &= ~(is_plaq & (pauli_id == 2).unsqueeze(0))
        return mask

    def select_action_gnn(self, epsilon: float):
        g = self.toric.build_defect_graph(k=self.gnn_k, use_next=False)
        nf = torch.tensor(g["node_features"], dtype=torch.float32, device=self.device)
        ei = torch.tensor(g["edge_index"], dtype=torch.long, device=self.device)
        ea = torch.tensor(g["edge_attr"], dtype=torch.float32, device=self.device)
        nt = torch.tensor(g["node_type"], dtype=torch.long, device=self.device)

        N = nf.shape[0]
        if N == 0:
            return Action((0, 0, 0), 1), {"node": 0, "move": 0}, g

        if random.random() < epsilon:
            node = random.randrange(N)
            valid = []
            for move in range(self.number_of_actions):
                pid = move % 3
                if nt[node].item() == 0 and pid == 0:
                    continue
                if nt[node].item() == 1 and pid == 2:
                    continue
                valid.append(move)
            move = random.choice(valid) if valid else random.randrange(self.number_of_actions)
        else:
            self.policy_net.eval()
            with torch.no_grad():
                q = self.policy_net(nf, ei, ea)  # [N,12]
                q = q.masked_fill(~self._valid_action_mask(nt), -1e9)
                flat = int(torch.argmax(q.view(-1)).item())
                node = flat // self.number_of_actions
                move = flat % self.number_of_actions

        dtype = int(g["node_type"][node])
        i, j = map(int, g["node_ij"][node])
        action_env = self.toric.map_defect_move_to_action(dtype, i, j, int(move))
        return action_env, {"node": int(node), "move": int(move)}, g

    # ============================================================
    # Training loop (both modes)
    # ============================================================
    def train(
        self,
        training_steps: int,
        target_update: int,
        epsilon_start: float = 1.0,
        num_of_epsilon_steps: int = 10,
        epsilon_end: float = 0.1,
        reach_final_epsilon: float = 0.5,
        optimizer: str = "Adam",
        batch_size: int = 32,
        replay_start_size: int = 500,
        minimum_nbr_of_qubit_errors: int = 0,
    ):
        criterion = nn.MSELoss(reduction="none")
        opt = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate) if optimizer != "RMSprop" \
            else optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)

        steps_per_bucket = int(np.round(training_steps / num_of_epsilon_steps))
        epsilon_decay = float(np.round((epsilon_start - epsilon_end) / num_of_epsilon_steps, 5))
        epsilon_update = int(steps_per_bucket * reach_final_epsilon)
        epsilon = float(epsilon_start)

        steps_counter = 0
        update_counter = 1
        iteration = 0

        while iteration < training_steps:
            self.toric = Toric_code(self.system_size, p_meas=self.p_meas, stack_T=self.stack_T, noise_conditioning=self.noise_conditioning)

            px, py, pz, mode = self._sample_episode_noise()
            self.toric.set_noise_params(px, py, pz)

            terminal_state = 0
            while terminal_state == 0:
                if minimum_nbr_of_qubit_errors == 0:
                    if mode == "biased":
                        self.toric.generate_random_error_biased(px, py, pz)
                    else:
                        self.toric.generate_random_error(self.p_error)
                else:
                    self.toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
                terminal_state = self.toric.terminal_state(self.toric.current_state)

            steps_ep = 0
            while terminal_state == 1 and steps_ep < self.max_nbr_actions_per_episode and iteration < training_steps:
                steps_ep += 1
                iteration += 1
                steps_counter += 1

                if self.decoder_mode == "gnn":
                    action_env, action_dict, state_g = self.select_action_gnn(epsilon=epsilon)
                    self.toric.step(action_env)
                    reward = self.get_reward()
                    next_g = self.toric.build_defect_graph(k=self.gnn_k, use_next=True)
                    terminal = self.toric.terminal_state(self.toric.next_state)
                    self.memory.save(Transition(state_g, action_dict, reward, next_g, terminal), 10000)
                else:
                    pos, op = self.select_action_grid(epsilon=epsilon)
                    self.toric.step(Action(pos, op))
                    reward = self.get_reward()
                    entry = self.toric.generate_memory_entry(Action(pos, op), reward, self.grid_shift)
                    self.memory.save(Transition(*entry), 10000)

                if steps_counter > replay_start_size:
                    update_counter += 1
                    if self.decoder_mode == "gnn":                                             
                        pass
                    else:
                        self.experience_replay_grid(criterion, opt, batch_size)

                if update_counter % target_update == 0:
                    self.target_net = deepcopy(self.policy_net)

                if epsilon_update > 0 and (update_counter % epsilon_update == 0):
                    epsilon = float(np.round(max(epsilon - epsilon_decay, epsilon_end), 3))

                self.toric.commit_next_state()
                terminal_state = self.toric.terminal_state(self.toric.current_state)

    # ============================================================
    # Prediction (works for both modes)
    # ============================================================
    def prediction(
        self,
        num_of_predictions: int = 1,
        epsilon: float = 0.0,
        num_of_steps: int = 50,
        PATH: str = None,
        prediction_list_p_error=None,
        minimum_nbr_of_qubit_errors: int = 0,
    ):
        if prediction_list_p_error is None:
            prediction_list_p_error = [0.1]
        if PATH is not None:
            self.load_network(PATH)

        self.policy_net.eval()

        error_corrected_list = np.zeros(len(prediction_list_p_error), dtype=float)
        ground_state_list = np.zeros(len(prediction_list_p_error), dtype=float)
        avg_steps_list = np.zeros(len(prediction_list_p_error), dtype=float)
        mean_q_list = np.zeros(len(prediction_list_p_error), dtype=float)
        logical_success_list = np.zeros(len(prediction_list_p_error), dtype=float)

        for i, p_item in enumerate(prediction_list_p_error):
            error_corrected = np.zeros(num_of_predictions, dtype=float)
            ground_ok = np.ones(num_of_predictions, dtype=bool)
            logical_success = np.zeros(num_of_predictions, dtype=float)

            mean_steps = 0.0
            mean_q = 0.0
            steps_counter = 0

            for j in range(num_of_predictions):
                self.toric = Toric_code(self.system_size, p_meas=self.p_meas, stack_T=self.stack_T, noise_conditioning=self.noise_conditioning)

                if isinstance(p_item, (tuple, list)) and len(p_item) == 3:
                    px, py, pz = map(float, p_item)
                    self.toric.set_noise_params(px, py, pz)
                    if minimum_nbr_of_qubit_errors == 0:
                        self.toric.generate_random_error_biased(px, py, pz)
                    else:
                        self.toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)
                else:
                    p = float(p_item)
                    self.toric.set_noise_params(p / 3.0, p / 3.0, p / 3.0)
                    if minimum_nbr_of_qubit_errors == 0:
                        self.toric.generate_random_error(p)
                    else:
                        self.toric.generate_n_random_errors(minimum_nbr_of_qubit_errors)

                terminal = self.toric.terminal_state(self.toric.current_state)
                steps = 0

                while terminal == 1 and steps < num_of_steps:
                    steps += 1
                    steps_counter += 1

                    if self.decoder_mode == "gnn":
                        action_env, _, g = self.select_action_gnn(epsilon=epsilon)
                        nf = torch.tensor(g["node_features"], dtype=torch.float32, device=self.device)
                        ei = torch.tensor(g["edge_index"], dtype=torch.long, device=self.device)
                        ea = torch.tensor(g["edge_attr"], dtype=torch.float32, device=self.device)
                        nt = torch.tensor(g["node_type"], dtype=torch.long, device=self.device)
                        with torch.no_grad():
                            q = self.policy_net(nf, ei, ea)
                            q = q.masked_fill(~self._valid_action_mask(nt), -1e9)
                            qv = float(q.max().item()) if q.numel() else 0.0
                        self.toric.step(action_env)
                    else:
                        pos, op = self.select_action_grid(epsilon=epsilon)
                        obs = self.toric.get_current_observation()
                        with torch.no_grad():
                            x = convert_from_np_to_tensor(obs[None, ...]).to(self.device)
                            qv = float(self.policy_net(x).max().item())
                        self.toric.step(Action(pos, op))

                    self.toric.commit_next_state()
                    terminal = self.toric.terminal_state(self.toric.current_state)
                    mean_q = incremental_mean(qv, mean_q, steps_counter)

                mean_steps = incremental_mean(steps, mean_steps, j + 1)

                error_corrected[j] = 1.0 - float(self.toric.terminal_state(self.toric.current_state))
                self.toric.eval_ground_state()
                ground_ok[j] = self.toric.ground_state
                logical_success[j] = float(error_corrected[j] == 1.0 and ground_ok[j] is True)

            error_corrected_list[i] = float(np.mean(error_corrected))
            ground_state_list[i] = float(np.mean(ground_ok))
            logical_success_list[i] = float(np.mean(logical_success))
            avg_steps_list[i] = float(np.round(mean_steps, 1))
            mean_q_list[i] = float(np.round(mean_q, 3))

        failure_rate = float(1.0 - np.mean(logical_success_list))
        return (
            error_corrected_list,
            ground_state_list,
            avg_steps_list,
            mean_q_list,
            [],
            logical_success_list,
            prediction_list_p_error,
            failure_rate,
        )
