import supersuit as ss # this package help us with preprocessing 

def preprocess_env(env):

    # help us with not lossing any information about env
    env = ss.max_observation_v0(env, 2)

    # ask agent for a new decission every 4 frame
    env = ss.frame_skip_v0(env, 4)

    # resize the image to 84x84 px
    env = ss.resize_v1(env, x_size=84, y_size=84)

    # reduce 3 color channels to just 1 channel
    env = ss.color_reduction_v0(env, mode='full')

    env = ss.frame_stack_v1(env, 4)

    return env

