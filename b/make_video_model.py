import numpy as np
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tensorflow.keras.models import load_model


ARTIFACT_DIRECTORY = './'
MODEL_VERSIONS = [350, 360]


def preview_model(model_version, filename, episodes=10):
    env = gym.make('LunarLander-v2')
    model_file = f'{ARTIFACT_DIRECTORY}models/{model_version}.h5'
    model = load_model(model_file)

    vid = VideoRecorder(env, path=ARTIFACT_DIRECTORY + f'video/{filename}.mp4')
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            frame = env.render(mode='rgb_array')
            vid.capture_frame()
            action = np.argmax(model.predict(state.reshape(1, -1))[0])  # type: ignore
            new_state, _, done, _ = env.step(action)
            state = new_state
    env.close()


for v in MODEL_VERSIONS:
    preview_model(v, filename=f'LunarLander-v2-{v}')
