import flax.linen as nn
import jax
import jax.numpy as jnp


def energy_fn(name, x, y):
    if name == "norm":
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-6)
    elif name == "dot":
        return jnp.sum(x * y, axis=-1)
    elif name == "cosine":
        return jnp.sum(x * y, axis=-1) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-6)
    elif name == "l2":
        return -jnp.sum((x - y) ** 2, axis=-1)
    elif name == "l1":
        return -jnp.sum(jnp.abs(x - y), axis=-1)
    else:
        raise ValueError(f"Unknown energy function: {name}")


def _A_angles_for_logging(config, g_encoder_params):
    """Base rotation angles for logging (rotational_A only)."""
    if not (config.get("use_A") and config.get("rotational_A")):
        return None
    if config.get("fixed_rope_A"):
        half = config["repr_dim"] // 2
        A_angles = jnp.pi * jnp.arange(1, half + 1, dtype=jnp.float32) / config["rope_alpha"]
    else:
        params = g_encoder_params.get("params")
        if params is None or "A" not in params:
            return None
        A_angles = params["A"]

    if config.get("half_zero_angles"):
        half = A_angles.shape[0]
        mask = (jnp.arange(half) >= (half // 2)).astype(A_angles.dtype)
        A_angles = A_angles * mask
    return A_angles


def _add_A_rotation_metrics(metrics: dict, A_angles) -> None:
    """Log summary stats for per-block base rotation angles θ (radians)."""
    if A_angles is None:
        return
    metrics["A_angles_mean_rad"] = jnp.mean(A_angles)
    metrics["A_angles_std_rad"] = jnp.std(A_angles)
    metrics["A_angles_min_rad"] = jnp.min(A_angles)
    metrics["A_angles_max_rad"] = jnp.max(A_angles)
    metrics["A_angles_mean_deg"] = jnp.degrees(jnp.mean(A_angles))
    metrics["A_angles_std_deg"] = jnp.degrees(jnp.std(A_angles))
    metrics["A_angles_min_deg"] = jnp.degrees(jnp.min(A_angles))
    metrics["A_angles_max_deg"] = jnp.degrees(jnp.max(A_angles))


def _A_dense_matrix_for_logging(config, g_encoder_params):
    """Raw A parameter when it is a full matrix (non-rotational goal encoder)."""
    if not config.get("use_A") or config.get("rotational_A"):
        return None
    params = g_encoder_params.get("params")
    if params is not None and "A" in params:
        return params["A"]
    return None


def _add_A_dense_matrix_metrics(metrics: dict, A_mat) -> None:
    if A_mat is None:
        return
    metrics["A_matrix_mean"] = jnp.mean(A_mat)
    metrics["A_matrix_mean_abs"] = jnp.mean(jnp.abs(A_mat))
    metrics["A_matrix_frobenius"] = jnp.linalg.norm(A_mat)


def contrastive_loss_fn(name, logits):
    if name == "fwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    elif name == "bwd_infonce":
        critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=0))
    elif name == "sym_infonce":
        critic_loss = -jnp.mean(
            2 * jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1) - jax.nn.logsumexp(logits, axis=0)
        )
    elif name == "binary_nce":
        critic_loss = -jnp.mean(jax.nn.sigmoid(logits))
    else:
        raise ValueError(f"Unknown contrastive loss function: {name}")
    return critic_loss


def update_actor_and_alpha(config, networks, transitions, training_state, key):
    def actor_loss(actor_params, critic_params, log_alpha, transitions, key):
        obs = transitions.observation  # expected_shape = self.batch_size, obs_size + goal_size
        state = obs[:, : config["state_size"]]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, config["goal_indices"]]
        observation = jnp.concatenate([state, goal], axis=1)

        means, log_stds = networks["actor"].apply(actor_params, observation)
        stds = jnp.exp(log_stds)
        x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
        action = nn.tanh(x_ts)
        log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
        log_prob -= 2 * (jnp.log(2.0) - x_ts - nn.softplus(-2.0 * x_ts))
        log_prob = log_prob.sum(-1)  # dimension = B

        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )
        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        delta = transitions.extras["delta"]
        phi, psi = networks["g_encoder"].apply(g_encoder_params, goal, delta)
        
        A_angles = _A_angles_for_logging(config, g_encoder_params)
        A_matrix_param = _A_dense_matrix_for_logging(config, g_encoder_params)

        # Use phi (A^delta @ psi) for matching with state-action representation
        qf_pi = energy_fn(config["energy_fn"], sa_repr, phi)

        actor_loss = jnp.mean(jnp.exp(log_alpha) * log_prob - qf_pi)

        return actor_loss, (log_prob, A_angles, A_matrix_param)

    def alpha_loss(alpha_params, log_prob):
        alpha = jnp.exp(alpha_params["log_alpha"])
        alpha_loss = alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - config["target_entropy"]))
        return jnp.mean(alpha_loss)

    (actor_loss, (log_prob, A_angles, A_matrix_param)), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
        training_state.actor_state.params,
        training_state.critic_state.params,
        training_state.alpha_state.params["log_alpha"],
        transitions,
        key,
    )
    new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

    alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss)(training_state.alpha_state.params, log_prob)
    new_alpha_state = training_state.alpha_state.apply_gradients(grads=alpha_grad)

    training_state = training_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)

    metrics = {
        "entropy": -log_prob,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "log_alpha": training_state.alpha_state.params["log_alpha"],
    }
    
    _add_A_rotation_metrics(metrics, A_angles)
    _add_A_dense_matrix_metrics(metrics, A_matrix_param)

    return training_state, metrics


def update_critic(config, networks, transitions, training_state, key):
    def critic_loss(critic_params, transitions, key):
        sa_encoder_params, g_encoder_params = (
            critic_params["sa_encoder"],
            critic_params["g_encoder"],
        )

        state = transitions.observation[:, : config["state_size"]]
        action = transitions.action

        sa_repr = networks["sa_encoder"].apply(sa_encoder_params, jnp.concatenate([state, action], axis=-1))
        goal = transitions.observation[:, config["state_size"] :]
        delta = transitions.extras["delta"]
        phi, psi = networks["g_encoder"].apply(g_encoder_params, goal, delta)
        
        A_angles = _A_angles_for_logging(config, g_encoder_params)
        A_matrix_param = _A_dense_matrix_for_logging(config, g_encoder_params)

        # Use phi (A^delta @ psi) for contrastive learning
        g_repr = phi

        # InfoNCE
        logits = energy_fn(config["energy_fn"], sa_repr[:, None, :], g_repr[None, :, :])
        critic_loss = contrastive_loss_fn(config["contrastive_loss_fn"], logits)

        # logsumexp regularisation
        logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
        critic_loss += config["logsumexp_penalty_coeff"] * jnp.mean(logsumexp**2)

        I = jnp.eye(logits.shape[0])
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return critic_loss, (logsumexp, I, correct, logits_pos, logits_neg, A_angles, A_matrix_param)

    (loss, (logsumexp, I, correct, logits_pos, logits_neg, A_angles, A_matrix_param)), grad = jax.value_and_grad(
        critic_loss, has_aux=True
    )(training_state.critic_state.params, transitions, key)
    new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
    training_state = training_state.replace(critic_state=new_critic_state)

    metrics = {
        "categorical_accuracy": jnp.mean(correct),
        "logits_pos": logits_pos,
        "logits_neg": logits_neg,
        "logsumexp": logsumexp.mean(),
        "critic_loss": loss,
    }
    
    _add_A_rotation_metrics(metrics, A_angles)
    _add_A_dense_matrix_metrics(metrics, A_matrix_param)

    return training_state, metrics
