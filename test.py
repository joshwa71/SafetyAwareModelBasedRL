import gym
from gym.wrappers.transform_observation import TransformObservation
from gym.wrappers.transform_reward import TransformReward
from gym.spaces.box import Box
import numpy as np
import safety_gym
"""An observation wrapper that augments observations by pixel values."""

import collections
import copy

import numpy as np

from gym import Wrapper, spaces
from gym import ObservationWrapper
from gym.envs.registration import register
import matplotlib.pyplot as plt


STATE_KEY = 'state'


class PixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values."""

# Pixel observation wrapper based on OpenAI Gym implementation.
# The MIT License

# Copyright (c) 2016 OpenAI (https://openai.com)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.


    def __init__(self,
                 env,
                 pixels_only=True,
                 render_kwargs=None,
                 pixel_keys=('pixels', )):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            pixels_only: If `True` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If `False`, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs: Optional `dict` containing keyword arguments passed
                to the `self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the `OrderedDict` of observations.
                Defaults to 'pixels'.

        Raises:
            ValueError: If `env`'s observation spec is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If `env`'s observation already contains any of the
                specified `pixel_keys`.
        """

        super(PixelObservationWrapper, self).__init__(env)

        if render_kwargs is None:
            render_kwargs = {}

        for key in pixel_keys:
            render_kwargs.setdefault(key, {})

            render_mode = render_kwargs[key].pop('mode', 'rgb_array')
            assert render_mode == 'rgb_array', render_mode
            render_kwargs[key]['mode'] = 'rgb_array'

        wrapped_observation_space = env.observation_space

        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = set([STATE_KEY])
        elif isinstance(wrapped_observation_space,
                        (spaces.Dict, collections.MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only:
            # Make sure that now keys in the `pixel_keys` overlap with
            # `observation_keys`
            overlapping_keys = set(pixel_keys) & set(invalid_keys)
            if overlapping_keys:
                raise ValueError("Duplicate or reserved pixel keys {!r}."
                                 .format(overlapping_keys))

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict()
            self.observation_space.spaces[STATE_KEY] = wrapped_observation_space

        # Extend observation space with pixels.

        pixels_spaces = {}
        for pixel_key in pixel_keys:
            render_kwargs[pixel_key]["mode"] ="offscreen"
            pixels = self.env.sim.render(**render_kwargs[pixel_key])[::-1, :, :]

            if np.issubdtype(pixels.dtype, np.integer):
                low, high = (0, 255)
            elif np.issubdtype(pixels.dtype, np.float):
                low, high = (-float('inf'), float('inf'))
            else:
                raise TypeError(pixels.dtype)

            pixels_space = spaces.Box(
                shape=pixels.shape, low=low, high=high, dtype=pixels.dtype)
            pixels_spaces[pixel_key] = pixels_space

        self.observation_space.spaces.update(pixels_spaces)

        self._env = env
        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys
        self.buttons = None
        self.COLOR_BUTTON = np.array([1, .5, 0, 1])
        self.COLOR_GOAL = np.array([0, 1, 0, 1])

    def observation(self, observation):
        pixels = self.render(camera_id=2, mode='rgb_array', width=image_size, height=image_size)
        obs = {'image': pixels}
        return obs

gym.logger.set_level(40)



env = gym.make('Safexp-PointGoal2-v0')
observation = env.reset()
image_size = 64
wrapped_env = PixelObservationWrapper(env, render_kwargs={'pixels': {'camera_name': "vision", 'mode': 'rgb_array', 'width':image_size,'height':image_size}})
#wrapped_env = make_safety('Safexp-PointGoal2-v0', image_size, use_pixels=True, action_repeat=1)
#print(env.action_space)
#observation = wrapped_env.reset()
#print(observation)
for i in range(100000):
    #pixels = wrapped_env.render(camera_id=2, mode='rgb_array', width=image_size, height=image_size)
    #print(pixels)
    env.render()
    next_observation, reward, done, info = env.step(env.action_space.sample())
    #print(next_observation)
    #plt.imshow(next_observation['image'])
    #plt.show()

    