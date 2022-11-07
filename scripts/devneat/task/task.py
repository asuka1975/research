from .space_feed4 import SpaceFeed4
from .cartpole import CartPole, CartPole1, BrokenPoleCartPole, DoublePoleCartPole, DoublePoleCartPoleGeometricMean, BrokenPoleCartPoleGeometricMean

def create_task(task_config):
    return globals()[task_config["name"]](task_config["setting"])