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


def get_loss(name: str) -> tf.keras.losses.Loss:
    name = name.lower()

    if name == "mse":
        loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
    elif name == "bce":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif name == "cce":
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        raise NotImplementedError(f"Unknown Loss Metric {name}.")

    print(f"Selected Loss Metric: {loss}")
    return loss


def get_metrics(names: list[str]) -> list:
    # TODO put metrics selection logic here
    return list(names)


def compile_model(
    model: tf.keras.Model,
    config: dict,
) -> None:

    optimizer = get_optimizer(
        name=config["optimizer"], learning_rate=config["learning_rate"]
    )
    loss = get_loss(name=config["loss"])
    metrics = get_metrics(names=config["metrics"])
    # explicitely setting run_eagerly=True is necessary in tf 2.0 when dealing
    # with custom layers and losses
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        run_eagerly=True,
    )
