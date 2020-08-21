import copy
import os.path as osp

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras


def dist_logp(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    dist = tfp.distributions.MultivariateNormalDiag(
        loc=mu, scale_diag=sigma)
    log_p = dist.log_prob(y_true)
    loss = -tf.reduce_mean(log_p)
    return loss


def dist_mse(y_true, y_pred):
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    mse = tf.reduce_mean(tf.square(tf.reduce_mean(mu - y_true)))
    return mse


def stats(y_true, y_pred):
    return tf.reduce_mean(y_pred, keepdims=False) - tf.reduce_mean(y_true, keepdims=False)


class CNPModel:
    def __init__(self, state_dims, action_dims, name='cnp_model'):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.name = name

    def build_model(self, encoder_hiddens, decoder_hiddens):
        context_x = keras.layers.Input(name='context_x',
                                       shape=(None, self.state_dims + self.action_dims))
        context_y = keras.layers.Input(name='context_y',
                                       shape=(None, self.state_dims))
        query_x = keras.layers.Input(name='query_x', shape=(self.state_dims + self.action_dims,))

        # ------------------ encoder --------------------- #
        x = keras.layers.Concatenate(name='context_concat', axis=-1)([context_x, context_y])
        for i, units in enumerate(encoder_hiddens[:-1], start=1):
            x = keras.layers.Dense(name=f'dense_{i}',
                                   units=units,
                                   activation='tanh')(x)
        encodes = keras.layers.Dense(name=f'dense_{len(encoder_hiddens)}',
                                     units=encoder_hiddens[-1],
                                     activation='linear')(x)
        # ------------------ aggregator ------------------ #
        aggregates = keras.layers.Lambda(lambda logit: tf.reduce_mean(logit, axis=1, keepdims=False),
                                         name='avg_aggregates')(encodes)

        # ------------------ decoder ------------------ #
        x = keras.layers.Concatenate(name='query_concat', axis=-1)([aggregates, query_x])
        for i, units in enumerate(decoder_hiddens[:-1], start=len(encoder_hiddens) + 1):
            x = keras.layers.Dense(name=f'dense_{i}',
                                   units=units,
                                   activation='tanh')(x)

        mu = keras.layers.Dense(name='mu', units=self.state_dims)(x)
        log_sigma = keras.layers.Dense(name=f'log_sigma', units=self.state_dims)(x)
        sigma = keras.layers.Lambda(lambda logit: 0.1 + 0.9 * tf.nn.softplus(logit),
                                    name='sigma')(log_sigma)  # bound variance
        dist_concat = keras.layers.Concatenate(name='dist_concat', axis=-1)([mu, sigma])

        # ------------------ model ----------------------- #
        self.model = keras.models.Model(inputs={'context_x': context_x,
                                                'context_y': context_y,
                                                'query_x': query_x},
                                        outputs={'mu': mu,
                                                 'sigma': sigma,
                                                 'dist_concat': dist_concat},
                                        name=self.name)
        return self.model.summary()

    def train(self, generator, epochs=100, ):
        self.model.compile(
            # run_eagerly=True,
            optimizer=keras.optimizers.Adam(5e-4),
            loss={'dist_concat': dist_logp},
            metrics={'dist_concat': dist_mse, 'mu': stats},
        )

        train_data = copy.deepcopy(generator)
        train_data.train = True
        vali_data = copy.deepcopy(generator)
        vali_data.train = False

        self.model.fit(
            train_data, epochs=epochs, steps_per_epoch=1000,
            validation_data=vali_data, validation_steps=20,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_dist_concat_loss',
                    patience=5, mode='auto',
                    restore_best_weights=True),
            ]
        )
        self.model.save(osp.abspath(osp.join(osp.dirname(__file__), f'Checkpoints/{self.name}')))

    def load_model(self):
        self.model = keras.models.load_model(
            osp.abspath(osp.join(osp.dirname(__file__), f'Checkpoints/{self.name}')),
            custom_objects={'dist_logp': dist_logp, 'dist_mse': dist_mse, 'stats': stats}
        )
        return self.model.summary()

    def set_context(self, context_x, context_y):
        # check shape
        assert context_x.shape[0] == context_y.shape[0], "Number of contexts not match"
        assert context_y.shape[-1] == self.state_dims
        assert context_x.shape[-1] == self.state_dims + self.action_dims

        self.context_x = np.expand_dims(context_x, axis=0)
        self.context_y = np.expand_dims(context_y, axis=0)

    def predict(self, query_x):
        query_x = query_x.reshape(-1, self.state_dims + self.action_dims)
        batch_size = query_x.shape[0]
        if batch_size > 1:
            query = {
                'context_x': np.repeat(self.context_x, batch_size, axis=0),
                'context_y': np.repeat(self.context_y, batch_size, axis=0),
                'query_x': query_x
            }
        else:
            query = {
                'context_x': self.context_x,
                'context_y': self.context_y,
                'query_x': query_x
            }
        target_y = self.model.predict(query)
        target_y = {name: pred for name, pred in zip(self.model.output_names, target_y)}
        return target_y


def gen_context(env, num_context_points=15):
    # ------------------- Generate Context ------------------------- #
    # Get context points
    context_x = []
    context_y = []

    while len(context_x) < num_context_points:
        obs = env.reset()
        for i in range(100):
            action = env.action_space.sample()
            new_obs, rew, done, info = env.step(action)
            delta = new_obs - obs
            context_x.append(np.concatenate([obs, np.array([action])]))
            context_y.append(delta)
            obs = new_obs
            if done:
                break

    idx = np.arange(len(context_x))
    np.random.shuffle(idx)
    context_x = np.stack(context_x, axis=0)[idx[:num_context_points]]
    context_y = np.stack(context_y, axis=0)[idx[:num_context_points]]
    return context_x, context_y
