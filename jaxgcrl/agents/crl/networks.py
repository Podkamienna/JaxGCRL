import logging
import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling, orthogonal


class Encoder(nn.Module):
    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = (
        0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    )
    use_relu: bool = False
    use_ln: bool = False

    @nn.compact
    def __call__(self, data: jnp.ndarray, delta: jnp.ndarray = None):
        logging.info("encoder input shape: %s", data.shape)
        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.use_ln:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        x = data
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        x = nn.Dense(self.repr_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class GoalEncoderWithA(nn.Module):
    """Goal encoder that outputs psi and can apply A^delta transformation to get phi."""
    repr_dim: int = 64
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = 0
    use_relu: bool = False
    use_ln: bool = False
    use_A: bool = True
    orthogonal_A: bool = False
    rotational_A: bool = False
    rotational_with_Q: bool = False
    orthogonal_method: str = "cayley"
    Q_orthogonal_method: str = "cayley"
    randomly_initialized_A: bool = False
    # When True with rotational_A, per-block base angles are fixed: i * pi / rope_alpha, i = 1..repr_dim/2.
    fixed_rope_A: bool = False
    rope_alpha: float = 108.0
    # When True with rotational_A, the first half of angle blocks are fixed to 0.
    half_zero_angles: bool = False

    def setup(self):
        assert self.orthogonal_method in ["cayley", "qr", "exp"], "Invalid orthogonal method"
        assert self.Q_orthogonal_method in ["cayley", "qr", "exp"], "Invalid Q orthogonal method"
        assert not (self.rotational_A and self.orthogonal_A), "Cannot have both rotational and orthogonal A"
        
        if self.use_A:
            if self.rotational_A:
                assert self.repr_dim % 2 == 0, "repr_dim must be divisible by 2 when using rotational_A"
                if not self.fixed_rope_A:
                    if self.randomly_initialized_A:
                        # Initialize angles uniformly from 0 to π
                        self.A = self.param("A", lambda rng: jax.random.uniform(rng, (self.repr_dim // 2,), minval=0, maxval=jnp.pi))
                    else:
                        self.A = self.param("A", lambda rng: jnp.zeros(self.repr_dim // 2))
                
                if self.rotational_with_Q:
                    if self.randomly_initialized_A:
                        self.Q = self.param("Q", orthogonal(), (self.repr_dim, self.repr_dim))
                    else:
                        self.Q = self.param("Q", lambda rng: jnp.eye(self.repr_dim))
            else:
                if self.randomly_initialized_A:
                    self.A = self.param("A", orthogonal(), (self.repr_dim, self.repr_dim))
                else:
                    self.A = self.param("A", lambda rng: jnp.eye(self.repr_dim))
        
        self.log_lambda = self.param("log_lambda", lambda rng: jnp.array(0.0))

    def _construct_rotational_matrix(self, A_vector, num_A_applications=None):
        """Construct block-diagonal rotational matrix from angles."""
        n_blocks = len(A_vector)
        repr_dim = n_blocks * 2
        
        if num_A_applications is not None:
            # Batch case
            angle_to_apply = A_vector[None, :] * num_A_applications[:, None]  # (batch, n_blocks)
            batch_size = len(num_A_applications)
            matrix = jnp.zeros((batch_size, repr_dim, repr_dim))
            
            block_starts = jnp.arange(n_blocks) * 2
            cos_vals = jnp.cos(angle_to_apply)  # (batch, n_blocks)
            sin_vals = jnp.sin(angle_to_apply)  # (batch, n_blocks)
            
            batch_indices = jnp.arange(batch_size)[:, None]  # (batch, 1)
            row_indices = block_starts[None, :]  # (1, n_blocks)
            col_indices = block_starts[None, :]  # (1, n_blocks)
            
            matrix = matrix.at[batch_indices, row_indices, col_indices].set(cos_vals)
            matrix = matrix.at[batch_indices, row_indices, col_indices + 1].set(sin_vals)
            matrix = matrix.at[batch_indices, row_indices + 1, col_indices].set(-sin_vals)
            matrix = matrix.at[batch_indices, row_indices + 1, col_indices + 1].set(cos_vals)            
        else:
            angle_to_apply = A_vector
            matrix = jnp.zeros((repr_dim, repr_dim))
            block_starts = jnp.arange(n_blocks) * 2
            cos_vals = jnp.cos(angle_to_apply)
            sin_vals = jnp.sin(angle_to_apply)
            
            matrix = matrix.at[block_starts, block_starts].set(cos_vals)
            matrix = matrix.at[block_starts, block_starts + 1].set(sin_vals)
            matrix = matrix.at[block_starts + 1, block_starts].set(-sin_vals)
            matrix = matrix.at[block_starts + 1, block_starts + 1].set(cos_vals)
        
        return matrix

    def _fixed_rope_angles(self) -> jnp.ndarray:
        half = self.repr_dim // 2
        return jnp.pi * jnp.arange(1, half + 1, dtype=jnp.float32) / self.rope_alpha

    def _apply_angle_constraints(self, A_vector: jnp.ndarray) -> jnp.ndarray:
        """Apply optional constraints to rotational angle blocks."""
        if not self.half_zero_angles:
            return A_vector
        half = A_vector.shape[0]
        mask = (jnp.arange(half) >= (half // 2)).astype(A_vector.dtype)
        return A_vector * mask

    def _get_orthogonal_Q(self):
        """Get orthogonal Q matrix."""
        if self.Q_orthogonal_method == "cayley":
            ss = self.Q - self.Q.T
            I = jnp.eye(self.Q.shape[0])
            Q_ortho = jnp.linalg.solve(I + ss, I - ss)
        elif self.Q_orthogonal_method == "qr":
            Q_ortho, _ = jnp.linalg.qr(self.Q)
        elif self.Q_orthogonal_method == "exp":
            ss = self.Q - self.Q.T
            Q_ortho = jax.scipy.linalg.expm(ss)
        else:
            raise ValueError(f"Invalid Q orthogonal method: {self.Q_orthogonal_method}")
        return Q_ortho

    @nn.compact
    def __call__(self, data: jnp.ndarray, delta: jnp.ndarray = None):
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.use_ln:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        # Input layer
        x = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init)(data)
        x = activation(normalize(x))

        # Residual blocks (depth // 2 for main layers, depth // 2 for residual layers)
        depth_half = self.network_depth // 2
        assert depth_half * 2 == self.network_depth, "Depth must be divisible by 2"
        
        # Create main layers and residual layers
        for i in range(depth_half):
            # Main path: layer -> norm -> activation
            main_out = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init, name=f"layer_{i}")(x)
            main_out = activation(normalize(main_out))
            
            # Residual path: layer -> norm (no activation before adding)
            res_out = nn.Dense(self.network_width, kernel_init=lecun_uniform, bias_init=bias_init, name=f"res_{i}")(main_out)
            res_out = normalize(res_out)
            
            # Residual connection
            x = res_out + x

        # Output layer
        psi = nn.Dense(self.repr_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        # Apply A matrix transformation if enabled
        if self.use_A:
            if self.rotational_A:
                if self.fixed_rope_A:
                    A_vector = self._fixed_rope_angles()
                else:
                    A_vector = self.A
                A_vector = self._apply_angle_constraints(A_vector)
                R = self._construct_rotational_matrix(A_vector, delta)
                if self.rotational_with_Q:
                    Q = self._get_orthogonal_Q()
                    a_to_apply = Q @ R @ Q.T
                else:
                    a_to_apply = R
            elif self.orthogonal_A:
                if self.orthogonal_method == "cayley":
                    ss = self.A - self.A.T
                    I = jnp.eye(self.A.shape[0])
                    a_to_apply = jnp.linalg.solve(I + ss, I - ss)
                elif self.orthogonal_method == "qr":
                    a_to_apply, _ = jnp.linalg.qr(self.A)
                elif self.orthogonal_method == "exp":
                    ss = self.A - self.A.T
                    a_to_apply = jax.scipy.linalg.expm(ss)
                else:
                    raise ValueError(f"Invalid orthogonal method: {self.orthogonal_method}")
            else:
                a_to_apply = self.A

            # Apply power operation if delta is provided and not rotational
            if not self.rotational_A and delta is not None:
                if isinstance(delta, jnp.ndarray) and delta.ndim > 0:
                    # Batch case: compute A^n for each n in delta
                    def compute_A_power(d):
                        d_int = int(d)
                        if d_int == 0:
                            return jnp.eye(self.repr_dim)
                        elif d_int == 1:
                            return a_to_apply
                        else:
                            # Compute A^d by repeated multiplication
                            result = a_to_apply
                            for _ in range(d_int - 1):
                                result = result @ a_to_apply
                            return result
                    
                    a_to_apply = jax.vmap(compute_A_power)(delta.astype(jnp.int32))
                else:
                    # Scalar case
                    delta_int = int(delta)
                    if delta_int == 0:
                        a_to_apply = jnp.eye(self.repr_dim)
                    elif delta_int == 1:
                        pass  # Already set
                    else:
                        result = a_to_apply
                        for _ in range(delta_int - 1):
                            result = result @ a_to_apply
                        a_to_apply = result

            # Apply transformation: phi = A @ psi (or A^delta @ psi)
            # if isinstance(a_to_apply, jnp.ndarray) and a_to_apply.ndim == 3:
                # Batch case: (batch, repr_dim, repr_dim)
            phi = jnp.einsum('brr,br->br', a_to_apply, psi)
            # else:
            #     # Single matrix case: (repr_dim, repr_dim)
            #     phi = a_to_apply @ psi
        else:
            phi = psi

        return phi, psi


class Actor(nn.Module):
    action_size: int
    network_width: int = 256
    network_depth: int = 4
    skip_connections: int = (
        0  # 0 for no skip connections, >= 0 means the frequency of skip connections (every X layers)
    )
    use_relu: bool = False
    use_ln: bool = False
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.use_ln:
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        logging.info("actor input shape: %s", x.shape)
        for i in range(self.network_depth):
            x = nn.Dense(self.network_width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)

            if self.skip_connections:
                if i == 0:
                    skip = x
                if i > 0 and i % self.skip_connections == 0:
                    x = x + skip
                    skip = x

        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std
