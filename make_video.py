import gym
from gym.wrappers import Monitor
from tensorflow.keras.models import load_model
import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import tensorflow as tf


ARTIFACT_DIRECTORY = './'


def preview_model(model_version, episodes=10):
    env = Monitor(gym.make('LunarLander-v2'), ARTIFACT_DIRECTORY + 'video', force=True)
    model_file = f'{ARTIFACT_DIRECTORY}{model_version}.h5'
    model = load_model(model_file)

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = np.argmax(model.predict(state.reshape(1, -1))[0])
            new_state, _, done, _ = env.step(action)
            state = new_state
    env.close()

preview_model(320)