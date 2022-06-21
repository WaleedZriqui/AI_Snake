import torch
import random
import numpy as np
from collections import deque # deque is data structure (Double Queue help adding and removing from both sides)
from game import SnakeGameAI, Direction, Point # SnakeGameAI is class, Direction and point are out of class 
from model import Linear_QNet, QTrainer
from helper import plot

# Constant Parameters
MAX_MEMORY = 100_000 # 100 000 item in memory
BATCH_SIZE = 1000 
LR = 0.001 # laering rate

class Agent:

    def __init__(self):   # as we say it's constructor 
        self.n_games = 0 # number of games
        self.epsilon = 0 # control randomness 
        self.gamma = 0.9 # {discount rate < 1} (needed when use Bellman equation)
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 256, 3) # input_size = 11, output_size = 3, hidden_size = didn't care
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # lr is defined in above  


    def get_state(self, game):
        head = game.snake[0] # first item from list
        point_l = Point(head.x - 20, head.y) # 20 is number block size 
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # current directin <boolean>
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or # goining right and point right has a collision
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    # Store everything in memory (Which is deque here)
    def remember(self, state, action, reward, next_state, done): # done here is as game over state
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    # when agent lose will all info here 
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # check if we already have more than 1000 items 
            mini_sample = random.sample(self.memory, BATCH_SIZE) # return list of tuples
        else:
            mini_sample = self.memory # take hole the memory
            
        # train_step for every state, action, reward, next_state, done
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)
    
    # Short memory what agent play current 
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # decide when make a random move or specific move (tradeoff exploration / exploitation)
        # in the begining random move, but then we nedd to get that from train
        self.epsilon = 80 - self.n_games # lests say 80 game random (when )
        final_move = [0,0,0] # at the begining 
        if random.randint(0, 200) < self.epsilon: # (.randint() returns an integer number selected element from the specified range)
            move = random.randint(0, 2) # then make random move (2 is included)
            final_move[move] = 1 # random move (make random move by make index of move index)
        else:
            state0 = torch.tensor(state, dtype=torch.float) # tensor([0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.])
            prediction = self.model(state0) # in Pytorch this will automatically excute forward function inside class model
            move = torch.argmax(prediction).item() # torch.argmax() -> return index of maximium value, .item() to convert it from tensor to number
            final_move[move] = 1

        return final_move


def train(): # This function is global (doesn't belong to agent class) 
    plot_scores = [] # save scores
    plot_mean_scores = [] # save average scores
    total_score = 0 # value of total score
    record = 0 # best record
    agent = Agent()
    game = SnakeGameAI()
    
    while True: # training loop
    
        state_old = agent.get_state(game) # get old state

        final_move = agent.get_action(state_old) # get move

        # perform move and get new state
        reward, done, score = game.play_step(final_move) # done = game over 
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done: # game over = true
            
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record: # when current score large than high record 
                record = score
                agent.model.save()

            # print information
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score) # all scores 
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__': #used to execute some code only if the file was run directly, and not imported
    train()