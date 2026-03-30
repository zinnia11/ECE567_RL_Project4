import gym
import numpy as np
import os

from .image_task import ImageTask

try:
    import gymnasium as gymnasium
except ImportError:
    gymnasium = None


def _minihack_wrapper_classes(gym_mod):
    """Build MiniHack-specific wrappers against either gym or gymnasium."""

    class MiniHackObsWrapper(gym_mod.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym_mod.spaces.Box(
                low=0, high=255, dtype=np.uint8, shape=(84, 84, 3)
            )

        def observation(self, obs):
            obs = obs["pixel_crop"]
            obs = np.pad(obs, [(2, 2), (2, 2), (0, 0)])
            return obs

    # from https://github.com/MiniHackPlanet/MiniHack/blob/e9c8c20fb2449d1f87163314f9b3617cf4f0e088/minihack/scripts/venv_demo.py#L28
    class MiniHackMakeVecSafeWrapper(gym_mod.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.basedir = os.getcwd()

        def step(self, action: int):
            os.chdir(self.env.env._vardir)
            x = self.env.step(action)
            os.chdir(self.basedir)
            return x

        def reset(self, **kwargs):
            os.chdir(self.env.env._vardir)
            x = self.env.reset(**kwargs)
            os.chdir(self.basedir)
            return x

        def close(self):
            os.chdir(self.env.env._vardir)
            self.env.close()
            os.chdir(self.basedir)

        def seed(self, core=None, disp=None, reseed=False):
            os.chdir(self.env.env._vardir)
            self.env.seed(core, disp, reseed)
            os.chdir(self.basedir)

    return MiniHackMakeVecSafeWrapper, MiniHackObsWrapper


if gymnasium is not None:

    class _OldGymStepResetWrapper(gymnasium.Wrapper):
        """Map Gymnasium step/reset API to pre-v0.26 gym (obs only; done = term | trunc)."""

        def reset(self, **kwargs):
            obs, _info = self.env.reset(**kwargs)
            return obs

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return obs, reward, terminated or truncated, info

else:
    _OldGymStepResetWrapper = None


# Ref: https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/polybeast/config.yaml#L48
# https://github.com/facebookresearch/nle/blob/b85184f65426e8a7a63b3fdbb1dead135e01e6cc/nle/env/tasks.py#L41
def make_minihack(
    env_name,
    observation_keys=["pixel_crop"],
    reward_win=1,
    reward_lose=0,
    penalty_time=0.0,
    penalty_step=-0.001,  # MiniHack uses different than -0.01 default of NLE
    penalty_mode="constant",
    character="mon-hum-neu-mal",
    savedir=None,  # save_tty=False -> savedir=None, see https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/common/envs/tasks.py#L168
    **kwargs,
):
    try:
        import minihack  # noqa: F401 — registers env IDs (Gymnasium in current MiniHack)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "MiniHack is not installed. Install NLE, then MiniHack "
            "(continual_rl/docs/BENCHMARK_INSTALL.md; "
            "https://github.com/MiniHackPlanet/MiniHack)."
        ) from e

    env_id = f"MiniHack-{env_name}"
    make_kw = dict(
        observation_keys=observation_keys,
        reward_win=reward_win,
        reward_lose=reward_lose,
        penalty_time=penalty_time,
        penalty_step=penalty_step,
        penalty_mode=penalty_mode,
        character=character,
        savedir=savedir,
        **kwargs,
    )

    # Current MiniHack registers with Gymnasium; old installs used gym only.
    if gymnasium is not None:
        from gymnasium.error import NameNotFound as GymnasiumNameNotFound

        try:
            env = gymnasium.make(env_id, **make_kw)
            env = _OldGymStepResetWrapper(env)
            MakeVec, Obs = _minihack_wrapper_classes(gymnasium)
        except GymnasiumNameNotFound:
            env = gym.make(env_id, **make_kw)
            MakeVec, Obs = _minihack_wrapper_classes(gym)
    else:
        env = gym.make(env_id, **make_kw)
        MakeVec, Obs = _minihack_wrapper_classes(gym)

    env = MakeVec(env)
    env = Obs(env)
    return env


def get_single_minihack_task(task_id, action_space_id, env_name, num_timesteps, eval_mode=False, **kwargs):
    return ImageTask(
        task_id=task_id,
        action_space_id=action_space_id,
        env_spec=lambda: make_minihack(env_name, **kwargs),
        num_timesteps=num_timesteps,
        time_batch_size=1,  # no framestack
        eval_mode=eval_mode,
        image_size=[84, 84],
        grayscale=False,
    )
