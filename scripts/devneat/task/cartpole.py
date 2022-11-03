from math import cos, sin, pi, fmod
from random import random, randint

import numpy as np
from numpy.linalg import solve
import numba

@numba.njit('Tuple((f8, f8))(f8, f8, f8, f8, f8, f8)')
def solve2(a1, b1, a2, b2, c1, c2):
    th_dot2 = 0
    if abs(a1) < 0.00001:
        th_dot2 = c1 / b1
    else:
        th_dot2 = (a2 * c1 - a1 * c2) / (a2 * b1 - a1 * b2)
    return (c2 - b2 * th_dot2) / a2, th_dot2

@numba.njit('Tuple((f8, f8, f8, f8, f8, f8))(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)')
def update2(dt, g, cweight, pweight, plength, j, resistance, th, th_dot, th_dot2, x, x_dot, x_dot2, f):
    x_dot2, th_dot2 = solve2(
        pweight * plength * cos(th), pweight * plength ** 2 + j,                                              # a1, b1
        cweight + pweight, pweight * plength * cos(th) + pweight * plength ** 2 + j,                          # a2, b2
        pweight * g * plength * sin(th) - pweight * plength * x_dot * th_dot * sin(th) - resistance * th_dot, # c1
        f - resistance * x_dot + pweight * plength * (th_dot ** 2) * sin(th)                                  # c2
    )

    x_dot += x_dot2 * dt
    x += x_dot * dt
    th_dot += th_dot2 * dt
    th += th_dot * dt
    while th < 0:
        th += 2 * pi
    th = np.fmod(th, 2 * pi)

    return x, x_dot, x_dot2, th, th_dot, th_dot2

@numba.njit('Tuple((f8, f8, f8))(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)')
def solve3(a11, a12, a13, a21, a22, a23, a31, a32, a33, b1, b2, b3):
    return (
        (b1 * a22 * a33 + b3 * a12 * a23 + b2 * a13 * a32 - b1 * a23 * a32 - b2 * a12 * a33 - b3 * a13 * a22) / (a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 - a12 * a21 * a33 - a13 * a22 * a31),
        (b2 * a11 * a33 + b1 * a23 * a31 + b3 * a13 * a21 - b3 * a11 * a23 - b1 * a21 * a33 - b2 * a13 * a31) / (a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 - a12 * a21 * a33 - a13 * a22 * a31),
        (b3 * a11 * a22 + b2 * a12 * a31 + b1 * a21 * a32 - b2 * a11 * a32 - b3 * a12 * a21 - b1 * a22 * a31) / (a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - a11 * a23 * a32 - a12 * a21 * a33 - a13 * a22 * a31)
    )

@numba.njit('Tuple((f8, f8, f8, f8, f8, f8, f8, f8, f8))(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)')
def update3(dt, g, cweight, p1weight, p1length, p2weight, p2length, j1, j2, resistance, th1, th1_dot, th1_dot2, th2, th2_dot, th2_dot2, x, x_dot, x_dot2, f):
    x_dot2, th1_dot2, th2_dot2 = solve3(
        cweight + p1weight + p2weight, (p1weight + p2weight) * p1length * cos(th1), p2weight * p2length * cos(th1),
        (p1weight + p2weight) * p1length * cos(th1), p1weight * p1length ** 2 + p2weight * p1length ** 2 + j1, p2weight * p1length * p2length * cos(th1 - th2),
        p2weight * p2length * cos(th2), p2weight * p1length * p2length * cos(th1 - th2), p2weight * p2length ** 2 + j2,
        
        (p1weight + p2weight) * p1length * (th1_dot ** 2) * sin(th1) + p2weight * p2length * (th2_dot ** 2) * sin(th2) + f - resistance * x_dot,
        p1weight * g * p1length * sin(th1) - p2weight * p1length * p2length * (th2_dot ** 2) * sin(th1 - th2) - resistance * th1_dot,
        p2weight * g * p2length * sin(th2) + p2weight * p1length * p2length * (th1_dot ** 2) * sin(th1 - th2) - resistance * th2_dot
    )

    x_dot += x_dot2 * dt
    x += x_dot * dt
    th1_dot += th1_dot2 * dt
    th1 += th1_dot * dt
    th2_dot += th2_dot2 * dt
    th2 += th2_dot * dt
    while th1 < 0:
        th1 += 2 * pi
    th1 = np.fmod(th1, 2 * pi)
    while th2 < 0:
        th2 += 2 * pi
    th2 = np.fmod(th2, 2 * pi)

    return x, x_dot, x_dot2, th1, th1_dot, th1_dot2, th2, th2_dot, th2_dot2

class CartPole:
    def __init__(self, config):
        pass

class CartPole:
    def __init__(self, config) -> None:
        self.dt = config["dt"]
        self.g = config["gravity"]
        self.cart_weight = config["cart_weight"]
        self.pole_weight = config["pole_weight"]
        self.pole_length = config["pole_length"]
        self.j = self.pole_weight * self.pole_length ** 2 # moment of inertia
        self.resistance = config["resistance"]
        self.theta = random() * (config["initial_theta"][1] - config["initial_theta"][0]) + config["initial_theta"][0]
        self.theta_dot = 0
        self.theta_dot2 = 0
        self.x = 0
        self.x_dot = 0
        self.x_dot2 = 0
        self.num_steps = config["num_steps"]
        self.steps = 0
        self.fitness_value = 0

    def update(self, inputs):
        f = inputs[0]
        self.x, self.x_dot, self.x_dot2, self.theta, self.theta_dot, self.theta_dot2 = update2(self.dt, self.g, self.cart_weight, self.pole_weight, self.pole_length, self.j, self.resistance, self.theta, self.theta_dot, self.theta_dot2, self.x, self.x_dot, self.x_dot2, f)

        self.fitness_value += 1 if cos(self.theta) > 5 / 6 else 0
        self.steps += 1

    def get_output(self):
        return [self.x, self.x_dot, cos(self.theta), sin(self.theta), self.theta_dot]

    def state(self):
        return {
            "cart_position" : self.x,
            "pole_angle" : self.theta
        }

    def label(self):
        return {
            -1 : "cart position",
            -2 : "cart velocity",
            -3 : "pole angle",
            -4 : "pole angular velocity",
            0 : "force"
        }

    def finish(self):
        return self.steps >= self.num_steps or not (-1 < self.x < 1)

    def fitness(self):
        if not (-1 < self.x < 1):
            return -100
        return self.fitness_value

    def max_fitness(self):
        return self.num_steps

    def min_fitness(self):
        return -100

    def num_inputs(self):
        return 5

    def num_outputs(self):
        return 1

class CartPole1:
    def __init__(self, config) -> None:
        self.dt = config["dt"]
        self.g = config["gravity"]
        self.cart_weight = config["cart_weight"]
        pole_rate = random() * 0.4 + 0.3
        self.pole1_weight = config["pole_weight"] * pole_rate
        self.pole1_length = config["pole_length"] * pole_rate
        self.pole2_weight = config["pole_weight"] * (1 - pole_rate)
        self.pole2_length = config["pole_length"] * (1 - pole_rate)
        self.j1 = self.pole1_weight * self.pole1_length ** 2 # moment of inertia
        self.j2 = self.pole2_weight * self.pole2_length ** 2
        self.resistance = config["resistance"]
        self.theta1 = random() * 2 * pi
        self.theta1_dot = 0
        self.theta1_dot2 = 0
        self.theta2 = self.theta1
        self.theta2_dot = 0
        self.theta2_dot2 = 0
        self.x = 0
        self.x_dot = 0
        self.x_dot2 = 0
        self.num_steps = config["num_steps"]
        self.steps = 0
        self.fitness_value = 0

    def update(self, inputs):
        f = inputs[0]

        left = np.array([
            [self.cart_weight + self.pole1_weight + self.pole2_weight, (self.pole1_weight + self.pole2_weight) * self.pole1_length * cos(self.theta1), self.pole2_weight * self.pole2_length * cos(self.theta1)],
            [(self.pole1_weight + self.pole2_weight) * self.pole1_length * cos(self.theta1), self.pole1_weight * self.pole1_length ** 2 + self.pole2_weight * self.pole1_length ** 2 + self.j1, self.pole2_weight * self.pole1_length * self.pole2_length * cos(self.theta1 - self.theta2)],
            [self.pole2_weight * self.pole2_length * cos(self.theta2), self.pole2_weight * self.pole1_length * self.pole2_length * cos(self.theta1 - self.theta2), self.pole2_weight * self.pole2_length ** 2 + self.j2]
        ])
        right = np.array([
            (self.pole1_weight + self.pole2_weight) * self.pole1_length * (self.theta1_dot ** 2) * sin(self.theta1) + self.pole2_weight * self.pole2_length * (self.theta2_dot ** 2) * sin(self.theta2) + f - self.resistance * self.x_dot,
            self.pole1_weight * self.g * self.pole1_length * sin(self.theta1) - self.pole2_weight * self.pole1_length * self.pole2_length * (self.theta2_dot ** 2) * sin(self.theta1 - self.theta2) - self.resistance * self.theta1_dot,
            self.pole2_weight * self.g * self.pole2_length * sin(self.theta2) + self.pole2_weight * self.pole1_length * self.pole2_length * (self.theta1_dot ** 2) * sin(self.theta1 - self.theta2) - self.resistance * self.theta2_dot
        ])
        self.x_dot2, self.theta1_dot2, self.theta2_dot2 = solve(left, right)

        self.x_dot += self.x_dot2 * self.dt
        self.x += self.x_dot * self.dt
        self.theta1_dot += self.theta1_dot2 * self.dt
        self.theta1 += self.theta1_dot * self.dt
        self.theta2_dot += self.theta2_dot2 * self.dt
        self.theta2 += self.theta2_dot * self.dt
        while self.theta1 < 0:
            self.theta1 += 2 * pi
        self.theta1 = fmod(self.theta1, 2 * pi)
        while self.theta2 < 0:
            self.theta2 += 2 * pi
        self.theta2 = fmod(self.theta2, 2 * pi)

        self.fitness_value += (self.pole1_length if cos(self.theta1) > 5 / 6 else 0) + (self.pole2_length if cos(self.theta2) > 5 / 6 else 0)
        self.steps += 1
        if not (-1 < self.cart_position < 1):
            self.steps = self.num_steps

    def get_output(self):
        return [self.x, self.x_dot, self.theta1, self.theta1_dot, self.theta2, self.theta2_dot]

    def state(self):
        return {
            "pole1_length" : self.pole1_length,
            "pole2_length" : self.pole2_length,
            "cart_position" : self.x,
            "pole1_angle" : self.theta1,
            "pole2_angle" : self.theta2
        }

    def finish(self):
        return self.steps >= self.num_steps or not (-1 < self.x < 1)

    def fitness(self):
        if not (-1 < self.x < 1):
            return -100
        return self.fitness_value

    def max_fitness(self):
        return self.num_steps

    def min_fitness(self):
        return -100

    def num_inputs(self):
        return 6

    def num_outputs(self):
        return 1


class BrokenPoleCartPole:
    def __init__(self, config):
        self.dt = config["dt"]
        self.g = config["gravity"]
        self.cart_weight = config["cart_weight"]
        self.resistance = config["resistance"]

        self.pole_weight = config["pole_weight"]
        self.pole_length = config["pole_length"]
        self.j = self.pole_weight * self.pole_length ** 2 # moment of inertia
        self.theta = random() * (config["initial_theta"][1] - config["initial_theta"][0]) + config["initial_theta"][0]
        self.theta_dot = 0
        self.theta_dot2 = 0
        self.x = 0
        self.x_dot = 0
        self.x_dot2 = 0

        # broken states
        self.pole_rate = random() * 0.4 + 0.3
        self.pole1_weight = config["pole_weight"] * self.pole_rate
        self.pole1_length = config["pole_length"] * self.pole_rate
        self.pole2_weight = config["pole_weight"] * (1 - self.pole_rate)
        self.pole2_length = config["pole_length"] * (1 - self.pole_rate)
        self.j1 = self.pole1_weight * self.pole1_length ** 2 # moment of inertia
        self.j2 = self.pole2_weight * self.pole2_length ** 2
        self.theta1 = random() * 2 * pi
        self.theta1_dot = 0
        self.theta1_dot2 = 0
        self.theta2 = self.theta1
        self.theta2_dot = 0
        self.theta2_dot2 = 0

        self.num_steps = config["num_steps"]
        self.steps = 0
        self.fitness_value = 0

        self.broken = False
        self.break_step = randint(*config["break_step"])

    def update(self, inputs):
        f = inputs[0]
        if not self.broken:
            self.x, self.x_dot, self.x_dot2, self.theta, self.theta_dot, self.theta_dot2 = update2(self.dt, self.g, self.cart_weight, self.pole_weight, self.pole_length, self.j, self.resistance, self.theta, self.theta_dot, self.theta_dot2, self.x, self.x_dot, self.x_dot2, f)
            self.fitness_value += 1 if cos(self.theta) > 5 / 6 else 0
        else:
            self.x, self.x_dot, self.x_dot2, self.theta1, self.theta1_dot, self.theta1_dot2, self.theta2, self.theta2_dot, self.theta2_dot2 = update3(self.dt, self.g, self.cart_weight, self.pole1_weight, self.pole1_length, self.pole2_weight, self.pole2_length, self.j1, self.j2, self.resistance, self.theta1, self.theta1_dot, self.theta1_dot2, self.theta2, self.theta2_dot, self.theta2_dot2, self.x, self.x_dot, self.x_dot2, f)
            self.fitness_value += (self.pole_rate if cos(self.theta1) > 5 / 6 else 0) + (1 - self.pole_rate if cos(self.theta2) > 5 / 6 else 0)

        self.steps += 1

        if self.break_step == self.steps:
            self.theta1 = self.theta
            self.theta1_dot = self.theta_dot
            self.theta1_dot2 = self.theta_dot2
            self.theta2 = self.theta
            self.broken = True

    def get_output(self):
        if self.broken:
            return [self.x, self.x_dot, self.theta1, self.theta1_dot, self.theta2, self.theta2_dot]
        else:
            return [self.x, self.x_dot, self.theta, self.theta_dot, 0, 0]

    def state(self):
        if self.broken:
            return {
                "broken" : True,
                "pole1_length" : self.pole1_length,
                "pole2_length" : self.pole2_length,
                "cart_position" : self.x,
                "pole1_angle" : self.theta1,
                "pole2_angle" : self.theta2
            }
        else:
            return {
                "broken" : False,
                "cart_position" : self.x,
                "pole_angle" : self.theta
            }

    def label(self):
        return {
            -1 : "cart position",
            -2 : "cart velocity",
            -3 : "pole1 angle",
            -4 : "pole1 angular velocity",
            -5 : "pole2 angle",
            -6 : "pole2 angular velocity",
            0 : "force"
        }

    def finish(self):
        return self.steps >= self.num_steps or not (-1 < self.x < 1)

    def fitness(self):
        if not (-1 < self.x < 1):
            return self.fitness_value - 100
        return self.fitness_value

    def max_fitness(self):
        return self.num_steps

    def min_fitness(self):
        return -100

    def num_inputs(self):
        return 6

    def num_outputs(self):
        return 1