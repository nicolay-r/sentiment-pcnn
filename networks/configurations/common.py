import tensorflow as tf


class CommonSettings:

    test_on_epochs = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
    optimiser = tf.train.AdadeltaOptimizer(
        learning_rate=0.1,
        epsilon=10e-6,
        rho=0.95)



