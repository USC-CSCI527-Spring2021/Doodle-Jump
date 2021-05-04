import os
import math
def formulate_reward(reward_type, reward_reason, spring_touch=False, monster_touch=False, score=0):
    """
        - Params
            - reward_type: type of reward set by agent (baseline is default)
            - reward_reason: reason for calling the reward function
                (did agent die, did agent get stuck, did score increment)
            - score_inc: was score incremented (bool)
            - spring_touch: was spring touched (bool)
            - monster_touch: was monster touched (bool)
        - Returns:
            - A reward value based on type and reason
        - to be called to assign a reward value to the agent
    """

    reward = None
    if reward_type == 1:
        # using baseline rewards
        if reward_reason == "DEFAULT":
            reward = 0
        if reward_reason == "DEAD":
            reward = -2
        if reward_reason == "STUCK":
            reward = -2
        if reward_reason == "SCORED":
            reward = 3

    elif reward_type == 2:
        # version 2 discourages agent standing at one place
        if reward_reason == "DEFAULT":
            reward = -1
        if reward_reason == "DEAD":
            reward = -2
        if reward_reason == "STUCK":
            reward = -2
        if reward_reason == "SCORED":
            reward = 3

    elif reward_type == 3:
        # version 3 reward takes into account monster and spring
        if reward_reason == "DEFAULT":
            reward = -1
        if reward_reason == "DEAD":
            reward = -2
        if reward_reason == "STUCK":
            reward = -2
        if reward_reason == "SCORED":
            reward = 3
            if spring_touch:
                reward += 3
            if monster_touch:
                reward -= 4

    elif reward_type == 4:
        # version 4 dynamic reward
        if reward_reason == "DEFAULT":
            reward = -1
        if reward_reason == "DEAD":
            reward = -2
        if reward_reason == "STUCK":
            reward = -2
        if reward_reason == "SCORED":
            reward = 3 + math.log(score)
            if spring_touch:
                reward += 3
            if monster_touch:
                reward -= 4
    elif reward_type == 5:
        # version 5 - agent not penalised for no points scored
        if reward_reason == "DEFAULT":
            reward = 0
        if reward_reason == "DEAD":
            reward = -2
        if reward_reason == "STUCK":
            reward = -2
        if reward_reason == "SCORED":
            reward = 3 + math.log(score)
            if spring_touch:
                reward += 3
            if monster_touch:
                reward -= 4
    elif reward_type == 6:
        # version 6 - same as type 5 but high penalty for dying/stuck
        if reward_reason == "DEFAULT":
            reward = 0
        if reward_reason == "DEAD":
            reward = -20
        if reward_reason == "STUCK":
            reward = -20
        if reward_reason == "SCORED":
            reward = 3 + math.log(score)
            if spring_touch:
                reward += 3
            if monster_touch:
                reward -= 4

    return reward
