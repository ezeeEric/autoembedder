import tensorflow as tf
import pandas as pd

y_true = [
    [0.85600763, -0.85807727, -0.75218486],
    [-0.72037966, -0.15377929, -0.78046557],
    [0.35826603, 0.0319686, -0.54466821],
]
y_pred = [
    [8.1048226e-03, -1.8281912e-02, -2.0242289e-02],
    [-6.2456918e-03, -1.3126648e-02, -7.6899203e-03],
    [-1.2688854e-04, -1.6250038e-04, -5.1886946e-05],
    [-3.5913135e-03, -1.0775279e-02, -5.6126826e-03],
    [0.35826603, 0.0319686, -0.54466821],
]


cosine_loss = tf.keras.losses.CosineSimilarity(
    axis=-1, reduction=tf.keras.losses.Reduction.NONE
)
for y in y_pred:
    dist = cosine_loss(y_true, y)
    print(dist)
    print(tf.math.argmin(dist))
