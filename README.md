# Doodle-Jump-RL
## **USC - CSCI 527** Under Prof. Micheal Zyda
Game Bot Reinforcement Learning and Adversarial Attack 
## [Jump to our webpage](https://pranavmswamy.github.io/doodle-jump-website/)
<img src="game\assets\giphy_end.gif" width="150" height="150"/>

## Getting Started
1. Clone the repository
2. Install the requirements `pip install -r requirements.txt`

## Usage
* Train/Test **DQN & DRQN** Model
```
python deepQAgent.py

  -h, --help            show this help message and exit
  --macos               select model to train the agent
  --human               playing the game manually without agent
  --test                playing the game with a trained agent
  -d {EASY,MEDIUM,HARD}, --difficulty {EASY,MEDIUM,HARD}
                        select difficulty of the game
  -m {dqn,drqn,resnet,mobilenet,mnasnet}, --model {dqn,drqn,resnet,mobilenet,mnasnet}
                        select model to train the agent
  -p MODEL_PATH, --model_path MODEL_PATH
                        path to weights of an earlier trained model
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        set learning rate for training the model
  -g GAMMA, --gamma GAMMA
                        set discount factor for q learning
  --max_memory MAX_MEMORY
                        Buffer memory size for long training
  --store_frames        store frames encountered during game play by agent
  --batch_size BATCH_SIZE
                        Batch size for long training
  --reward_type {1,2,3,4,5,6}
                        types of rewards formulation
  --exploration EXPLORATION
                        number of games to explore
  --channels CHANNELS   set the image channels for preprocessing
  --height HEIGHT       set the image height post resize
  --width WIDTH         set the image width post resize
  --server              when training on server add this flag
  --seed SEED           change seed value for creating game randomness
  --max_games MAX_GAMES
                        set the max number of games to be played by the agent
  --explore {epsilon_g,epsilon_g_decay_exp,epsilon_g_decay_exp_cur}
                        select the exploration vs exploitation tradeoff
  --decay_factor DECAY_FACTOR
                        set the decay factor for exploration
  --epsilon EPSILON     set the epsilon value for exploration
  --attack              use fast fgsm attack to manipulate the input state
  --attack_eps ATTACK_EPS
                        epsilon value for the fgsm attack
```
* Train/Test **A2C** Model
```
python a2cAgent.py --options
```
* Train/Test **PPO** Model
```
python ppoAgent.py --options
```

## Folder Tree
```
Root
|   .gitignore
|   a2cAgent.py
|   deepQAgent.py
|   helper.py
|   ppoAgent.py
|   README.md
|   requirements.txt
|   rewards.py
|
+---game
|   |   doodlejump.py
|   |   LICENSE
|   |   __init__.py
|   |
|   \---assets
|
\---model
    |   a2cTrainer.py
    |   deepQTrainer.py
    |   networks.py
    |   ppoTrainernew.py
```
