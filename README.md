main idea: Project teches it's self how to play snake 

Represintaion By me Waleed and Ahmad: 

We start by the game with pygame, 
then we bulid agent and deeplaering algorithm with pyTorch. 
Also we use a basic of Reinforcement Learning.

After run project: python agent.py
this will start trainign our egent, and u well also sow that there is 
a game screen (Black screen) and score screen (Score vs. number of games)

At start, game will play to learn our agent 
rules are clearrly 2 things 1) Find the food 
                            2) Don't hit the border  

Our snake will know nothing about the game and will make random moves in 
enviermonet. When u lock to the score screen u will see chart changed 
every time snake play and lose, as we say at each game snake will move 
in random way and every game it learns more and more and become better 
and better.

At first few games we wont see a lot of improvment (it need around 800 
to 100 games untill AI will good game stratigey) the time it take will 
be around 10 minetus.

So after play game 10 minetus look to the score screen, u will find score 
become high for snake, not perfect but it's get in better and better.

--------------------------------------------------------------------------

We will divide this project to 4 Main Parts:- 
A) Lets start with Theory of Reinforcement learning 
B) Implement the actual game (envierment) using pygame
C) Implement the agent 
D) Implement the actual model with pytorch

** So let's start with Theory (Reinforcement Learning) 

This definination we take from Wikipedia

"Reinforcement learning (RL) is an area of machine learning concerned with 
how software agents try to take actions in an environment in order to 
maximize the notion of cumulative reward".

Also "RL is teaching a software agent how to behave in an environment by
telling it how good it's doing".

So from this definination we find that it tech (Agent) wich is our 
computer palyer, and (envierment) which is our game in this case, 
finally we find (cumulative rewards) which tell agent who good it doing 
that try to find best next action. 

To train the agent there are a lot of different abroches and not all of 
them envolve Deep learning, But in our case we use Deep Learning
"This approach extends reinforcement learning by using a deep neural 
network to predict the actions"

((See Conclusion pic in this file))

1- implement the game (will have game loop every game will do a play_step 
which get an action and move the snake then after move it return the 
current reward, also if game is over or not and current score).

2- implement the agent (it basicly put every thing together, so it must 
know about the game and model, so we will store of them inside agent. 
Also it will implement a Training loop, find the state then based in the 
state we callculate next action using model.predict(), by this new action 
we do a next play_step(action) method and will have return values, 
form this information we calculate a new sate then remeber every thing 
{will have old state, new state, game_over state, and score} and then 
train our model by model.train() ).

3- implement the Model (we called it Linear_QNet(DQN) which is not 
complecated it's just a feed forward neural net with a few linear layers 
which need to train the model by {old state, new state, game_over state, 
and score} then asked about next action).

**************************************************************************

Now lets speek about the variabe and what we spesific a static deta mean: 

1) Reward 
 - snake eat food: +10 points
 - snake game over: -10 points
 - else action: 0 points 

2) Action {Which will determine next move} 
(snake will have three choice to move Straight <up, down>, right and left) 
 - [1,0,0] -> Go straight
 - [0,1,0] -> right turn 
 - [0,0,1] -> left turn 

3) State {Tell snake information about the enviermont}
 [danger straight, danger right, danger left,
  direction left, direction right, direction up, direction down,
  food left, food right, food up, food down]

all of these will be boolean values and inside one array State
for the direction only one of them will have value 1 another will 0
 
Ex: [0,0,0,
     0,1,0,0
     0,1,0,1] 
Suppose that snake is go to right now -->, and have above value
** snake know that there isn't any danger straight, right, and left
** snake know that it's move right now 
** snake know food in right and down of it 


!!! See Screenshot_1 
u will have a state of 11 inputs then u find 3 outputs by using neural 
network, at the output layer u will have 3 values as us see in picture 
[5.0, 2.7, 0.1] so to choose the Action u should make max value and make 
it 1 and onvert another to 0 [1, 0, 0] Go Straight


Now how we find the result?
Bu using train model (Deep Q Learning) 
Q value stands for Quality of action <each action should emprove qulaity of snake>

0. Init Q Value (= init model) {random parametet}
1. Choose action (model.predict(state)) 
<or random move at begining when snake didn't know anything> <----
2. Perform action <next move>                                     |
3. Measure reward                                                 |
4. Update Q value (then train model) {repet to step number 1} ____|

The important thing here is for step 1 to detrmine random move or by
predict value (there is a trade off)
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

In step 4 to update Q value We use a Bellman Equation:

NewQ = Q + alfa [R + Gama maxQ` - Q]
alfa: learning rate
R: Reward for action at this state
maxQ`: maximum expected future reward from all possible action from new state 

in conclosion:
Q = model.predict(state0)
Qnew = R + Gama . max( Q(state1) )

So go back to pic 2 
 - state = get_state(game) <state 0>
 - new_state = get_sate(game) <state 1> 

We also will have a Loss function:
loss = (Qnew - Q)^2     => mean squared error 
-----------------------------------------------------------------------

** So Now start implement all 3 Classes and that start is game class** 

In game class we take a snake game by human and make alot of changes 
to the class to handle that action will come from agent


-----------------------------------------------------------------------

!!!!!! Our implrmntation in not complete since snake still meanwhile learning 
at some moments eat it self and loses, also sometimes stuck inside a loop and 
eat it self !!!!!!!!! 
 
We make it working for about 2 hours 
after make 4000 game it score 92 as maximum score <not bad>


