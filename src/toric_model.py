import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from .util import Action, Perspective


class Toric_code:
    """
    Toric code environment.

    CHANGED vs original repo:
    - Novelty #1: measurement noise p_meas + temporal stacking stack_T (measured syndrome history).
    - Novelty #2: biased noise params (px,py,pz) + optional conditioning planes in observation.
    - Keeps original grid decoder workflow (generate_perspective + generate_memory_entry),
      but generalized to multi-channel observations.
    - Novelty #3: graph builder + mapping (defect node, move_id) -> Action for GNN-style decoder.
    """

    def __init__(self, size: int, p_meas: float = 0.0, stack_T: int = 1, noise_conditioning: bool = False):
        if size % 2 == 0:
            raise ValueError("Toric_code requires an odd system size.")
        if stack_T < 1:
            raise ValueError("stack_T must be >= 1.")
        if p_meas < 0:
            raise ValueError("p_meas must be >= 0.")

        self.system_size = int(size)
        self.p_meas = float(p_meas)
        self.stack_T = int(stack_T)
        self.noise_conditioning = bool(noise_conditioning)

        self.plaquette_matrix = np.zeros((self.system_size, self.system_size), dtype=int)
        self.vertex_matrix = np.zeros((self.system_size, self.system_size), dtype=int)
        self.qubit_matrix = np.zeros((2, self.system_size, self.system_size), dtype=int)

        # true (noise-free) syndrome
        self.current_state = np.stack((self.vertex_matrix, self.plaquette_matrix), axis=0)
        self.next_state = np.stack((self.vertex_matrix, self.plaquette_matrix), axis=0)

        self.ground_state = True

        # Identity=0, X=1, Y=2, Z=3
        self.rule_table = np.array([[0, 1, 2, 3],
                                    [1, 0, 3, 2],
                                    [2, 3, 0, 1],
                                    [3, 2, 1, 0]], dtype=int)

        # Novelty #2: episode noise params (for conditioning)
        self.px, self.py, self.pz = 0.0, 0.0, 0.0

        # Novelty #1: measured syndrome history
        zero_frame = np.zeros((2, self.system_size, self.system_size), dtype=int)
        self._obs_history = deque([zero_frame.copy() for _ in range(self.stack_T)], maxlen=self.stack_T)
        self._next_obs_history = deque([zero_frame.copy() for _ in range(self.stack_T)], maxlen=self.stack_T)

    # -------------------------
    # Noise params (Novelty #2)
    # -------------------------
    def set_noise_params(self, px: float, py: float, pz: float):
        self.px, self.py, self.pz = float(px), float(py), float(pz)

    # -------------------------
    # Error generation
    # -------------------------
    def generate_random_error(self, p_error: float):
        """Depolarizing: total p_error, uniform Pauli type."""
        p_error = float(p_error)
        if p_error < 0 or p_error > 1:
            raise ValueError("p_error must be in [0,1].")

        mask = (np.random.rand(2, self.system_size, self.system_size) < p_error).astype(int)
        pauli = np.random.randint(1, 4, size=(2, self.system_size, self.system_size))
        self.qubit_matrix[:, :, :] = mask * pauli

        # conditioning params for depolarizing
        self.set_noise_params(p_error / 3.0, p_error / 3.0, p_error / 3.0)

        self.syndrom('state')
        self._init_obs_history_from_state(self.current_state)

    def generate_random_error_biased(self, px: float, py: float, pz: float):
        """Biased: X with px, Y with py, Z with pz."""
        px, py, pz = float(px), float(py), float(pz)
        if px < 0 or py < 0 or pz < 0:
            raise ValueError("px,py,pz must be non-negative.")
        if px + py + pz > 1.0 + 1e-12:
            raise ValueError("px+py+pz must be <= 1.")

        r = np.random.rand(2, self.system_size, self.system_size)
        qm = np.zeros_like(r, dtype=int)
        qm[r < px] = 1
        qm[(r >= px) & (r < px + py)] = 2
        qm[(r >= px + py) & (r < px + py + pz)] = 3
        self.qubit_matrix[:, :, :] = qm

        self.set_noise_params(px, py, pz)

        self.syndrom('state')
        self._init_obs_history_from_state(self.current_state)

    def generate_n_random_errors(self, n: int):
        """Fixed number of random Pauli errors."""
        n = int(n)
        if n < 0 or n > 2 * self.system_size ** 2:
            raise ValueError("n out of range.")

        errors = np.random.randint(1, 4, size=n)
        flat = np.zeros(2 * self.system_size ** 2, dtype=int)
        flat[:n] = errors
        np.random.shuffle(flat)
        self.qubit_matrix[:, :, :] = flat.reshape(2, self.system_size, self.system_size)

        # no special conditioning here unless you want (keep as-is)
        self.syndrom('state')
        self._init_obs_history_from_state(self.current_state)

    # -------------------------
    # Novelty #1: measurement noise + stacking
    # -------------------------
    def _measure_syndrome(self, true_state: np.ndarray) -> np.ndarray:
        """Flip each syndrome bit with probability p_meas (bit-flip measurement noise)."""
        if self.p_meas <= 0:
            return true_state.copy()
        flips = (np.random.rand(*true_state.shape) < self.p_meas).astype(int)
        # true_state is 0/1 => XOR is safe
        return (true_state ^ flips).astype(int)

    def _init_obs_history_from_state(self, true_state: np.ndarray):
        meas = self._measure_syndrome(true_state)
        self._obs_history = deque([meas.copy() for _ in range(self.stack_T)], maxlen=self.stack_T)
        self._next_obs_history = deque([meas.copy() for _ in range(self.stack_T)], maxlen=self.stack_T)

    def _stack_history(self, hist: deque) -> np.ndarray:
        """
        Base stacked channels = 2*stack_T.
        If noise_conditioning: append 3 planes (px,py,pz) => channels +3.
        """
        stacked = np.concatenate(list(hist), axis=0).astype(np.float32)  # [2T, d, d]
        if self.noise_conditioning:
            d = self.system_size
            planes = np.array([self.px, self.py, self.pz], dtype=np.float32).reshape(3, 1, 1)
            planes = np.broadcast_to(planes, (3, d, d)).astype(np.float32)
            stacked = np.concatenate([stacked, planes], axis=0)  # [2T+3, d, d]
        return stacked

    def get_current_observation(self) -> np.ndarray:
        """Observation used by the agent (stacked measured syndrome + optional conditioning)."""
        return self._stack_history(self._obs_history)

    def get_next_observation(self) -> np.ndarray:
        """Next observation (after step), based on measured next_state."""
        return self._stack_history(self._next_obs_history)

    def _latest_measured_frame(self, use_next: bool = False) -> np.ndarray:
        """Return the latest measured syndrome frame [2,d,d] (vertex, plaquette)."""
        hist = self._next_obs_history if use_next else self._obs_history
        return hist[-1]

    def _persistence_maps(self, use_next: bool = False):
        """Average defect presence over the history (helps GNN with measurement noise)."""
        hist = self._next_obs_history if use_next else self._obs_history
        T = len(hist)
        v = np.zeros((self.system_size, self.system_size), dtype=float)
        p = np.zeros((self.system_size, self.system_size), dtype=float)
        for frame in hist:
            v += frame[0]
            p += frame[1]
        v /= max(T, 1)
        p /= max(T, 1)
        return v, p

    # -------------------------
    # Dynamics
    # -------------------------
    def step(self, action: Action):
        """
        Apply action on qubit_matrix, compute next_state syndrome, and update _next_obs_history.
        NOTE: current_state is NOT updated here (same as original design); call commit_next_state() after you accept the step.
        """
        qm, row, col = action.position
        add_operator = int(action.action)

        old_operator = int(self.qubit_matrix[qm, row, col])
        new_operator = int(self.rule_table[old_operator, add_operator])
        self.qubit_matrix[qm, row, col] = new_operator

        self.syndrom('next_state')

        # build next history from current history + measured(next_state)
        self._next_obs_history = deque(self._obs_history, maxlen=self.stack_T)
        self._next_obs_history.append(self._measure_syndrome(self.next_state))

    def commit_next_state(self):
        """Commit next_state -> current_state and sync observation history."""
        self.current_state = self.next_state
        self._obs_history = self._next_obs_history

    # -------------------------
    # Syndrome computation
    # -------------------------
    def syndrom(self, which: str):
        """
        Compute vertex/plaquette defects from current qubit_matrix.
        """
        # vertex: from Y or Z
        q0 = self.qubit_matrix[0]
        charge = ((q0 == 2) | (q0 == 3)).astype(int)
        charge_shift = np.roll(charge, 1, axis=0)
        charge0 = ((charge + charge_shift) == 1).astype(int)

        q1 = self.qubit_matrix[1]
        charge = ((q1 == 2) | (q1 == 3)).astype(int)
        charge_shift = np.roll(charge, 1, axis=1)
        charge1 = ((charge + charge_shift) == 1).astype(int)

        vertex_matrix = ((charge0 + charge1) == 1).astype(int)

        # plaquette: from X or Y
        q0 = self.qubit_matrix[0]
        flux = ((q0 == 1) | (q0 == 2)).astype(int)
        flux_shift = np.roll(flux, -1, axis=1)
        flux0 = ((flux + flux_shift) == 1).astype(int)

        q1 = self.qubit_matrix[1]
        flux = ((q1 == 1) | (q1 == 2)).astype(int)
        flux_shift = np.roll(flux, -1, axis=0)
        flux1 = ((flux + flux_shift) == 1).astype(int)

        plaquette_matrix = ((flux0 + flux1) == 1).astype(int)

        if which == 'state':
            self.current_state = np.stack((vertex_matrix, plaquette_matrix), axis=0)
        elif which == 'next_state':
            self.next_state = np.stack((vertex_matrix, plaquette_matrix), axis=0)
        else:
            raise ValueError("which must be 'state' or 'next_state'")

    # -------------------------
    # Terminal + logical checks
    # -------------------------
    def terminal_state(self, state: np.ndarray) -> int:
        """0: terminal (no defects), 1: not terminal."""
        return 0 if np.all(state == 0) else 1

    def eval_ground_state(self):
        """
        True: trivial loops only
        False: non-trivial loop happened (logical failure)
        NOTE: works reliably for odd system sizes (as in original repo).
        """
        z0 = ((self.qubit_matrix[0] == 2) | (self.qubit_matrix[0] == 3)).astype(int)
        z1 = ((self.qubit_matrix[1] == 2) | (self.qubit_matrix[1] == 3)).astype(int)
        x0 = ((self.qubit_matrix[0] == 1) | (self.qubit_matrix[0] == 2)).astype(int)
        x1 = ((self.qubit_matrix[1] == 1) | (self.qubit_matrix[1] == 2)).astype(int)

        self.ground_state = True
        if (np.sum(x0) % 2 == 1) or (np.sum(x1) % 2 == 1) or (np.sum(z0) % 2 == 1) or (np.sum(z1) % 2 == 1):
            self.ground_state = False

    # ============================================================
    # GRID MODE (original-style perspective logic, generalized to multi-channel)
    # ============================================================
    def _rotate_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Rotate 90 degrees counter-clockwise with toric-specific correction for vertex channels.

        Works on:
          - 2 channels (v,p)
          - 2*stack_T channels (stacked v,p frames)
          - plus optional 3 conditioning planes

        IMPORTANT:
        - In the original repo rotate_state() uses np.roll(vertex, 1, axis=0).
          We keep the same rule here for compatibility.
        """
        obs = np.asarray(obs)
        C = obs.shape[0]

        syndrome_C = 2 * self.stack_T
        syndrome_C = min(syndrome_C, C)

        out = []
        num_pairs = syndrome_C // 2
        for t in range(num_pairs):
            v = obs[2 * t + 0]
            p = obs[2 * t + 1]
            rot_p = np.rot90(p)
            rot_v = np.rot90(v)
            rot_v = np.roll(rot_v, 1, axis=0)  # keep original rule (compatibility)
            out.extend([rot_v, rot_p])

        # rotate remaining planes (e.g., conditioning px/py/pz)
        for k in range(syndrome_C, C):
            out.append(np.rot90(obs[k]))

        return np.stack(out, axis=0)

    def generate_perspective(self, grid_shift: int, obs: np.ndarray):
        """
        Create candidate perspectives around defects (original idea).
        CHANGED vs original: now works with stacked+conditioned observations.

        obs shape: [C, d, d]
        """
        obs = np.asarray(obs)
        d = self.system_size
        if obs.shape[1] != d or obs.shape[2] != d:
            raise ValueError("Observation spatial shape mismatch.")

        def mod(x): return x % d

        perspectives = []

        # We detect defects from the *latest measured* syndrome (robust with stack_T)
        latest = self._latest_measured_frame(use_next=False)
        latest_vertex = latest[0]
        latest_plaquette = latest[1]

        # qubit matrix 0 candidates
        for i in range(d):
            for j in range(d):
                if latest_vertex[i, j] == 1 or latest_vertex[mod(i + 1), j] == 1 or \
                   latest_plaquette[i, j] == 1 or latest_plaquette[i, mod(j - 1)] == 1:
                    new_obs = np.roll(obs, grid_shift - i, axis=1)
                    new_obs = np.roll(new_obs, grid_shift - j, axis=2)
                    perspectives.append(Perspective(new_obs, (0, i, j)))

        # qubit matrix 1 candidates
        for i in range(d):
            for j in range(d):
                if latest_vertex[i, j] == 1 or latest_vertex[i, mod(j + 1)] == 1 or \
                   latest_plaquette[i, j] == 1 or latest_plaquette[mod(i - 1), j] == 1:
                    new_obs = np.roll(obs, grid_shift - i, axis=1)
                    new_obs = np.roll(new_obs, grid_shift - j, axis=2)
                    new_obs = self._rotate_observation(new_obs)
                    perspectives.append(Perspective(new_obs, (1, i, j)))

        return perspectives

    def generate_memory_entry(self, action: Action, reward: float, grid_shift: int):
        """
        Store (perspective, centered_action, reward, next_perspective, terminal) like the original repo.

        CHANGED vs original:
        - perspective/next_perspective are now multi-channel (stacked + optional conditioning).
        - terminal is computed from TRUE next_state (not noisy measurement).
        """
        current_obs = self.get_current_observation()
        next_obs = self.get_next_observation()

        def shift(obs, row, col):
            x = np.roll(obs, grid_shift - row, axis=1)
            x = np.roll(x, grid_shift - col, axis=2)
            return x

        qm, row, col = action.position
        op = action.action

        if qm == 0:
            perspective = shift(current_obs, row, col)
            next_perspective = shift(next_obs, row, col)
            centered = Action((0, grid_shift, grid_shift), op)
        else:
            perspective = shift(current_obs, row, col)
            next_perspective = shift(next_obs, row, col)
            perspective = self._rotate_observation(perspective)
            next_perspective = self._rotate_observation(next_perspective)
            centered = Action((1, grid_shift, grid_shift), op)

        terminal = self.terminal_state(self.next_state)
        return perspective, centered, reward, next_perspective, terminal

    # ============================================================
    # GNN MODE (Novelty #3): graph builder + action mapping
    # ============================================================
    def _torus_delta(self, a: int, b: int) -> int:
        d = self.system_size
        x = (b - a) % d
        if x > d // 2:
            x -= d
        return int(x)

    def build_defect_graph(self, k: int = 8, use_next: bool = False):
        """
        Returns dict of numpy arrays:
          node_features: [N, F]
          edge_index: [2, E]
          edge_attr: [E, 3] (dx_norm, dy_norm, dist_norm)
          node_type: [N] (0 vertex, 1 plaquette)
          node_ij: [N,2]
        """
        frame = self._latest_measured_frame(use_next=use_next)
        v_persist, p_persist = self._persistence_maps(use_next=use_next)

        v_coords = np.argwhere(frame[0] == 1)
        p_coords = np.argwhere(frame[1] == 1)

        nodes, types, ijs, persists = [], [], [], []

        for (i, j) in v_coords:
            nodes.append((int(i), int(j)))
            types.append(0)
            ijs.append((int(i), int(j)))
            persists.append(float(v_persist[i, j]))

        for (i, j) in p_coords:
            nodes.append((int(i), int(j)))
            types.append(1)
            ijs.append((int(i), int(j)))
            persists.append(float(p_persist[i, j]))

        N = len(nodes)
        if N == 0:
            F = 7 + (3 if self.noise_conditioning else 0)
            return {
                "node_features": np.zeros((0, F), dtype=np.float32),
                "edge_index": np.zeros((2, 0), dtype=np.int64),
                "edge_attr": np.zeros((0, 3), dtype=np.float32),
                "node_type": np.zeros((0,), dtype=np.int64),
                "node_ij": np.zeros((0, 2), dtype=np.int64),
            }

        d = self.system_size
        two_pi = 2.0 * np.pi

        feats = []
        for idx, (i, j) in enumerate(nodes):
            # torus positional encoding (sin/cos)
            si, ci = np.sin(two_pi * i / d), np.cos(two_pi * i / d)
            sj, cj = np.sin(two_pi * j / d), np.cos(two_pi * j / d)

            t = types[idx]
            onehot = [1.0, 0.0] if t == 0 else [0.0, 1.0]
            persist = persists[idx]

            f = [si, ci, sj, cj] + onehot + [persist]
            if self.noise_conditioning:
                f += [self.px, self.py, self.pz]

            feats.append(f)

        node_features = np.array(feats, dtype=np.float32)
        node_type = np.array(types, dtype=np.int64)
        node_ij = np.array(ijs, dtype=np.int64)

        coords = np.array(nodes, dtype=int)
        dist = np.zeros((N, N), dtype=float)
        for a in range(N):
            for b in range(N):
                if a == b:
                    dist[a, b] = 0.0
                else:
                    di = self._torus_delta(coords[a, 0], coords[b, 0])
                    dj = self._torus_delta(coords[a, 1], coords[b, 1])
                    dist[a, b] = np.sqrt(di * di + dj * dj)

        edge_src, edge_dst, edge_attr = [], [], []

        if N == 1:
            edge_src.append(0)
            edge_dst.append(0)
            edge_attr.append([0.0, 0.0, 0.0])
        else:
            for a in range(N):
                nn = np.argsort(dist[a])[1:min(k + 1, N)]
                for b in nn:
                    di = self._torus_delta(coords[a, 0], coords[b, 0]) / d
                    dj = self._torus_delta(coords[a, 1], coords[b, 1]) / d
                    dn = float(dist[a, b] / d)
                    edge_src.append(a)
                    edge_dst.append(int(b))
                    edge_attr.append([di, dj, dn])

        return {
            "node_features": node_features,
            "edge_index": np.array([edge_src, edge_dst], dtype=np.int64),
            "edge_attr": np.array(edge_attr, dtype=np.float32),
            "node_type": node_type,
            "node_ij": node_ij,
        }

    def map_defect_move_to_action(self, defect_type: int, i: int, j: int, move_id: int) -> Action:
        """
        move_id in [0..11]:
          dir_id = move_id // 3   (0 up, 1 down, 2 left, 3 right)
          pauli_id = move_id % 3  (0 X, 1 Y, 2 Z)
          env Pauli uses 1..3
        """
        d = self.system_size
        dir_id = int(move_id // 3)
        pauli_id = int(move_id % 3)
        pauli = pauli_id + 1

        i = int(i) % d
        j = int(j) % d

        if defect_type == 0:
            # vertex defect
            if dir_id == 0:
                qm, r, c = 0, (i - 1) % d, j
            elif dir_id == 1:
                qm, r, c = 0, i, j
            elif dir_id == 2:
                qm, r, c = 1, i, (j - 1) % d
            else:
                qm, r, c = 1, i, j
        else:
            # plaquette defect
            if dir_id == 0:
                qm, r, c = 1, i, j
            elif dir_id == 1:
                qm, r, c = 1, (i + 1) % d, j
            elif dir_id == 2:
                qm, r, c = 0, i, j
            else:
                qm, r, c = 0, i, (j + 1) % d

        return Action((qm, r, c), pauli)

    # -------------------------
    # Plot (optional)
    # -------------------------
    def plot_toric_code(self, state, title):
        """
        Full toric-code visualization: qubits + Pauli errors + vertex/plaquette defects.

        CHANGED vs previous draft:
        - FIXED: if noise_conditioning=True, last channels are px/py/pz planes.
          So for stacked obs we must take the latest syndrome frame from the first 2*stack_T channels,
          not simply state[-2:].
        """
        os.makedirs("plots", exist_ok=True)

        state = np.asarray(state)
        if state.shape[0] != 2:
            # FIX: handle stacked obs + optional conditioning planes correctly
            syndrome_C = 2 * self.stack_T
            syndrome_C = min(syndrome_C, state.shape[0])  # safety
            state = state[syndrome_C - 2:syndrome_C, :, :]

        x_error_qubits1 = np.where(self.qubit_matrix[0, :, :] == 1)
        y_error_qubits1 = np.where(self.qubit_matrix[0, :, :] == 2)
        z_error_qubits1 = np.where(self.qubit_matrix[0, :, :] == 3)

        x_error_qubits2 = np.where(self.qubit_matrix[1, :, :] == 1)
        y_error_qubits2 = np.where(self.qubit_matrix[1, :, :] == 2)
        z_error_qubits2 = np.where(self.qubit_matrix[1, :, :] == 3)

        vertex_matrix = state[0, :, :]
        plaquette_matrix = state[1, :, :]
        vertex_defect_coordinates = np.where(vertex_matrix)
        plaquette_defect_coordinates = np.where(plaquette_matrix)

        # Keep original-style grid coordinates
        xLine = np.linspace(0, self.system_size, self.system_size)
        x = range(self.system_size)
        X, Y = np.meshgrid(x, x)
        XLine, YLine = np.meshgrid(x, xLine)

        markersize_qubit = 15
        markersize_excitation = 7
        markersize_symbols = 7
        linewidth = 2

        fig, ax = plt.subplots(1, 1)

        ax.plot(XLine, -YLine, 'black', linewidth=linewidth)
        ax.plot(YLine, -XLine, 'black', linewidth=linewidth)

        # add the last two black lines
        ax.plot(XLine[:, -1] + 1.0, -YLine[:, -1], 'black', linewidth=linewidth)
        ax.plot(YLine[:, -1], -YLine[-1, :], 'black', linewidth=linewidth)

        ax.plot(X + 0.5, -Y, 'o', color='black', markerfacecolor='white', markersize=markersize_qubit + 1)
        ax.plot(X, -Y - 0.5, 'o', color='black', markerfacecolor='white', markersize=markersize_qubit + 1)

        # add grey qubits
        ax.plot(X[-1, :] + 0.5, -Y[-1, :] - 1.0, 'o', color='black', markerfacecolor='grey', markersize=markersize_qubit + 1)
        ax.plot(X[:, -1] + 1.0, -Y[:, -1] - 0.5, 'o', color='black', markerfacecolor='grey', markersize=markersize_qubit + 1)

        # all x errors
        ax.plot(x_error_qubits1[1], -x_error_qubits1[0] - 0.5, 'o', color='r', label="x error", markersize=markersize_qubit)
        ax.plot(x_error_qubits2[1] + 0.5, -x_error_qubits2[0], 'o', color='r', markersize=markersize_qubit)
        ax.plot(x_error_qubits1[1], -x_error_qubits1[0] - 0.5, 'o', color='black', markersize=markersize_symbols, marker=r'$X$')
        ax.plot(x_error_qubits2[1] + 0.5, -x_error_qubits2[0], 'o', color='black', markersize=markersize_symbols, marker=r'$X$')

        # all y errors
        ax.plot(y_error_qubits1[1], -y_error_qubits1[0] - 0.5, 'o', color='blueviolet', label="y error", markersize=markersize_qubit)
        ax.plot(y_error_qubits2[1] + 0.5, -y_error_qubits2[0], 'o', color='blueviolet', markersize=markersize_qubit)
        ax.plot(y_error_qubits1[1], -y_error_qubits1[0] - 0.5, 'o', color='black', markersize=markersize_symbols, marker=r'$Y$')
        ax.plot(y_error_qubits2[1] + 0.5, -y_error_qubits2[0], 'o', color='black', markersize=markersize_symbols, marker=r'$Y$')

        # all z errors
        ax.plot(z_error_qubits1[1], -z_error_qubits1[0] - 0.5, 'o', color='b', label="z error", markersize=markersize_qubit)
        ax.plot(z_error_qubits2[1] + 0.5, -z_error_qubits2[0], 'o', color='b', markersize=markersize_qubit)
        ax.plot(z_error_qubits1[1], -z_error_qubits1[0] - 0.5, 'o', color='black', markersize=markersize_symbols, marker=r'$Z$')
        ax.plot(z_error_qubits2[1] + 0.5, -z_error_qubits2[0], 'o', color='black', markersize=markersize_symbols, marker=r'$Z$')

        # defects
        ax.plot(vertex_defect_coordinates[1], -vertex_defect_coordinates[0], 'o', color='blue', label="charge", markersize=markersize_excitation)
        ax.plot(plaquette_defect_coordinates[1] + 0.5, -plaquette_defect_coordinates[0] - 0.5, 'o', color='red', label="flux", markersize=markersize_excitation)

        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')

        fig.savefig(f"plots/graph_{title}.png", bbox_inches="tight")
        plt.close(fig)
