import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder

ARTIFACT_DIRECTORY = './'


def preview_random_agent(episodes=10):
    env = gym.make('LunarLander-v2')

    vid = VideoRecorder(env, path=ARTIFACT_DIRECTORY + 'video/LunarLander-v2-Random.mp4')
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            frame = env.render(mode='rgb_array')
            vid.capture_frame()
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
    env.close()


preview_random_agent()
