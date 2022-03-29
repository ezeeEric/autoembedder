import tensorflow as tf


def get_optimizer(name: str, learning_rate: float) -> tf.keras.optimizers.Optimizer:
    name = name.lower()

    if name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise NotImplementedError(f"Unknown optimizer {name}")

    print(f"Selected Optimizer: {optimizer}")
    return optimizer