from typing import Tuple
import random

type Coord = Tuple[int, int, int, int]

SCREEN_SIZE = (600, 600)
MOVE = 10
BALL_SPEED = (4, 4)

class Game:
    pos_x_p1 = SCREEN_SIZE[0] // 10
    pos_x_p2 = SCREEN_SIZE[0] // 10 * 9
    dim_x = 10
    dim_y = 40
    dim = 12
    time_limit = 60
    ball_y_max_speed = BALL_SPEED[1]

    def __init__(self):
        self.score_p1 = 0
        self.score_p2 = 0
        self.pos_y_p1 = SCREEN_SIZE[1] // 2
        self.pos_y_p2 = SCREEN_SIZE[1] // 2
        self.reset_ball()
        self.clock = 0

    def reset_ball(self):
        self.pos_x_ball = SCREEN_SIZE[0] // 2
        self.pos_y_ball = SCREEN_SIZE[1] // 2
        self.vx_ball = BALL_SPEED[0]
        self.vy_ball = BALL_SPEED[1]
        self.vy_ball = random.random() * (BALL_SPEED[1] - 1) + 1
        if (random.random() < 0.5):
            self.vy_ball = -self.vy_ball

    def change_dir_ball(self, coef=None):
        if coef == None:
            coef = self.vy_ball / self.ball_y_max_speed
        self.vx_ball = -self.vx_ball
        if self.vx_ball < 0:
            self.pos_x_ball -= self.dim_x * 1.01
        else:
            self.pos_x_ball += self.dim_x * 1.01
        self.vy_ball = coef * self.ball_y_max_speed
    
    def change_dir_ball_env_collision(self):
        self.vx_ball = self.vx_ball
        self.vy_ball = -self.vy_ball

    def update_score_p1(self):
        self.score_p1 += 1

    def update_score_p2(self):
        self.score_p2 += 1

    def rectangle_overlap(self, coord1: Coord, coord2: Coord):
        if max (coord1[0], coord2[0]) > min (coord1[2], coord2[2]):
            return False
        if max (coord1[1], coord2[1]) > min (coord1[3], coord2[3]):
            return False
        return True
    
    def get_player1_coords(self) -> Coord:
        return (self.pos_x_p1, self.pos_y_p1, self.pos_x_p1 + self.dim_x, self.pos_y_p1 + self.dim_y)
    
    def get_player2_coords(self) -> Coord:
        return (self.pos_x_p2, self.pos_y_p2, self.pos_x_p2 + self.dim_x, self.pos_y_p2 + self.dim_y)
    
    def get_ball_coords(self) -> Coord:
        return (self.pos_x_ball, self.pos_y_ball, self.pos_x_ball + self.dim, self.pos_y_ball + self.dim)
    
    def check_player_collision(self):
        if self.rectangle_overlap(self.get_player1_coords(), self.get_ball_coords()):
            relative_pos_on_the_paddle = ((self.pos_y_ball + self.dim / 2) - (self.pos_y_p1 + self.dim_y / 2)) / (self.dim_y / 2)
            self.change_dir_ball(relative_pos_on_the_paddle)
            return 1

        if self.rectangle_overlap(self.get_player2_coords(), self.get_ball_coords()):
            relative_pos_on_the_paddle = ((self.pos_y_ball + self.dim / 2) - (self.pos_y_p2 + self.dim_y / 2)) / (self.dim_y / 2)
            self.change_dir_ball(relative_pos_on_the_paddle)
            return 2
        return 0

    def snap_back_to_screen_p1(self):
        self.pos_x_p1 = min (max (0, self.pos_x_p1), SCREEN_SIZE[0] - self.dim_x)
        self.pos_y_p1 = min (max (0, self.pos_y_p1), SCREEN_SIZE[0] - self.dim_y)

    def snap_back_to_screen_p2(self):
        self.pos_x_p2 = min (max (0, self.pos_x_p2), SCREEN_SIZE[0] - self.dim_x)
        self.pos_y_p2 = min (max (0, self.pos_y_p2), SCREEN_SIZE[0] - self.dim_y)

    def snap_back_to_screen(self):
        self.snap_back_to_screen_p1()
        self.snap_back_to_screen_p2()

    def step(self, move_p1, move_p2, max_points=1):
        reward = 0
        done = False

        # making sure that there players are moving in a valid way
        if move_p1 < 0:
            move_p1 = -1
        elif move_p1 > 0:
            move_p1 = 1
        if move_p2 < 0:
            move_p2 = -1
        elif move_p2 > 0:
            move_p2 = 1

        # moving the players
        self.pos_y_p1 += MOVE * move_p1
        self.pos_y_p2 += MOVE * move_p2

        # moving the ball
        self.pos_x_ball += self.vx_ball
        self.pos_y_ball += self.vy_ball

        # ceiling collsion
        if self.pos_y_ball < 0 or self.pos_y_ball + self.dim > SCREEN_SIZE[1]:
            self.change_dir_ball_env_collision()

        # wall collision
        if self.pos_x_ball < 0:
            reward = 1
            self.update_score_p2()
            self.reset_ball()
            if max (self.score_p1, self.score_p2) >= max_points:
                done = True
            
        if self.pos_x_ball + self.dim > SCREEN_SIZE[0]:
            reward = -1
            self.update_score_p1()
            self.reset_ball()
            if max (self.score_p1, self.score_p2) >= max_points:
                done = True

        if (self.check_player_collision() == 2):
            reward += 10
        return reward, done

    def get_score(self):
        return f"{self.score_p1} - {self.score_p2}"
    
    def get_state(self):
        return {
            "pos_x_p1": self.pos_x_p1,
            "pos_y_p1": self.pos_y_p1,
            "pos_x_p2": self.pos_x_p2,
            "pos_y_p2": self.pos_y_p2,
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "pos_ball_x": self.pos_x_ball,
            "pos_ball_y": self.pos_y_ball,
            "dim": self.dim,
            "velocity_ball_x": self.vx_ball,
            "velocity_ball_y": self.vy_ball,
            "score_p1": self.score_p1,
            "score_p2": self.score_p2,
            "clock": self.clock,
            "time_limit": self.time_limit
        }