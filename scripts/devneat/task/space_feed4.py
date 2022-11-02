from random import random
import math
import copy

from numba import jit, njit

SMELLONLY = 1
MOVEONLY = 2
PERFECT = 3

@njit('f8(f8, f8, f8)')
def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

@njit('f8(f8, f8, f8, f8)')
def distance(p1x, p1y, p2x, p2y):
    dx = p1x - p2x
    dy = p1y - p2y
    return dx * dx + dy * dy

@njit('f8(f8, f8, f8, f8)')
def distance2(p1x, p1y, p2x, p2y):
    return math.sqrt(distance(p1x, p1y, p2x, p2y))

class SpaceFeed4:
    def __init__(self, config) -> None:
        self.width = config["width"]
        self.height = config["height"]
        self.num_feeds = config["num_feeds"]
        self.num_poisons = config["num_poisons"]
        self.agent_velocity = config["agent_velocity"]
        self.eat_radius = config["eat_radius"]
        self.dead_threashold = config["dead_threshold"]
        self.space_type = config["type"]
        if "max_steps" in config:
            self.max_steps = config["max_steps"]
        self.has_max_steps = hasattr(self, "max_steps")
        if "judge_steps" in config:
            self.judge_steps = config["judge_steps"]
        self.has_judge_steps = hasattr(self, "judge_steps")

        self.position = (random() * self.width, random() * self.height)
        self.feeds = [(random() * self.width, random() * self.height, 1) for _ in range(self.num_feeds)] + \
                    [(random() * self.width, random() * self.height, 2) for _ in range(self.num_poisons)]
        self.eaten_poisons = 0
        self.eaten_feeds = 0
        self.fitness_value = 0

        self.step = 1
        
    def update(self, inputs):
        if self.space_type == SMELLONLY:
            if self.step % self.judge_steps == 0:
                i = min(range(len(self.feeds)), key=lambda i: distance(*self.feeds[i][:2], *self.position))
                feed = self.feeds.pop(i)
                if inputs[0] > 0:
                    self.eaten_poisons = feed[2] - 1
                    self.eaten_feeds = 1 - (feed[2] - 1)
        elif self.space_type == MOVEONLY:
            i = min(range(len(self.feeds)), key=lambda i: distance(*self.feeds[i][:2], *self.position))
            d = math.sqrt(inputs[1] * inputs[1] + inputs[2] * inputs[2])
            if d > 0:
                x, y = inputs[1] / d, inputs[2] / d
                self.position = (
                    clamp(self.position[0] + self.agent_velocity * x, 0, self.width),
                    clamp(self.position[1] + self.agent_velocity * y, 0, self.height)
                )
            if distance2(*self.position, *self.feeds[i][:2]) < self.eat_radius:
                self.feeds.pop(i)
        else:
            i = min(range(len(self.feeds)), key=lambda i: distance(*self.feeds[i][:2], *self.position))
            d = math.sqrt(inputs[1] * inputs[1] + inputs[2] * inputs[2])
            if d > 0:
                x, y = inputs[1] / d, inputs[2] / d
                self.position = (
                    clamp(self.position[0] + self.agent_velocity * x, 0, self.width),
                    clamp(self.position[1] + self.agent_velocity * y, 0, self.height)
                )
            if distance2(*self.position, *self.feeds[i][:2]) < self.eat_radius and inputs[0] > 0:
                feed = self.feeds.pop(i)
                if inputs[0] > 0:
                    self.eaten_poisons = feed[2] - 1
                    self.eaten_feeds = 1 - (feed[2] - 1)
        self.step += 1

    def get_output(self):
        i = min(range(len(self.feeds)), key=lambda i: distance(*self.feeds[i][:2], *self.position))
        outputs = [0] * 4
        if SMELLONLY & self.space_type == SMELLONLY:
            outputs[0] = self.feeds[i][2]
        if MOVEONLY & self.space_type == MOVEONLY:
            d = distance2(*self.position, *self.feeds[i][:2])
            x, y = self.feeds[i][0] - self.position[0], self.feeds[i][1] - self.position[1]
            outputs[1] = d
            if not math.isclose(d, 0, abs_tol=0.0001):
                outputs[2] = x / d
                outputs[3] = y / d
        return outputs


    def state(self):
        return copy.deepcopy({
            "position" : { "x" : self.position[0], "y" : self.position[1] },
            "feeds" : self.feeds,
            "eaten_feeds" : self.eaten_feeds,
            "eaten_poisons" : self.eaten_poisons,
            "input_switch" : False,
            "output_switch" : False
        })

    def finish(self):
        return self.dead() or len(self.feeds) == 0 or (self.has_max_steps and self.step >= self.max_steps)

    def fitness(self):
        if self.space_type & SMELLONLY == SMELLONLY:
            return -100 if self.eaten_poisons >= self.dead_threashold else self.eaten_feeds
        else:
            return self.num_feeds - len(self.feeds)

    def max_fitness(self):
        return self.num_feeds

    def min_fitness(self):
        if self.space_type == MOVEONLY:
            return 0
        return -100

    def dead(self):
        return self.eaten_poisons >= self.dead_threashold

    def num_inputs(self):
        return 4
    
    def num_outputs(self):
        return 3
