import pygame
import random
from enum import Enum
from collections import namedtuple #as it like as dictionary in python
import numpy as np # for handle arrays and math function 

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum): 
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 50

class SnakeGameAI:

    def __init__(self, w=640, h=480): #let's say it's like constructor
        self.w = w #self = this
        self.h = h
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self): #after evey lose it came here (game state)
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action): # convert method public by remove _ before name of method that becouse we need it to use by user 
        self.frame_iteration += 1
        
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # move by agent {remeber that Action may be [1,0,0], ...}
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # check if game over or find big number of iteration without find food
        reward = 0 #inital reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake): #if collision hsppen or dnske itrste huge number of iterations without any inhansment
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # place new food or just move 
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # return cureent reward, game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None): # also here we consider danger
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]: # out point is now in snake body
            return True   # [1:] remeber that mean it find all index from 1 index to final index (work as for loop)

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
                           # Right=1     Down=2          Left=3           Up=4
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change (go Straight)
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4 # when change form up to right 4 -> 1, 5%4 = 1  
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y) #new position for the head