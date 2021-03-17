import os
import datetime
import argparse
import random
import numpy as np
import cv2
import torch
from collections import deque
from game.doodlejump import DoodleJump
from model import Deep_QNet, Deep_RQNet, QTrainer
from helper import write_model_params
from torch.utils.tensorboard import SummaryWriter


class Agent:
    def __init__(self, args):
        self.n_games = 0
        self.epsilon = 0
        self.ctr = 1
        seed = args.seed
        self.exploration = args.exploration
        os.environ['PYTHONHASHSEED'] = str(seed)
        # Torch RNG
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Python RNG
        np.random.seed(seed)
        random.seed(seed)
        self.store_frames = args.store_frames
        self.image_h = args.height
        self.image_w = args.width
        self.memory = deque(maxlen=args.max_memory)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        if args.model=="dqn":
            self.model = Deep_QNet()
        elif args.model=="drqn":
            self.model = Deep_RQNet()
        if args.model_path or args.test:
            self.model.load_state_dict(torch.load(args.model_path))
        self.trainer = QTrainer(model=self.model, lr=self.lr, gamma=self.gamma, device=self.device)

    def get_state(self, game):
        state = game.getCurrentFrame()
        img = cv2.cvtColor(cv2.resize(state, (self.image_w, self.image_h)), cv2.COLOR_BGR2GRAY)
        M = cv2.getRotationMatrix2D((self.image_w / 2, self.image_h / 2), 270, 1.0)
        img = cv2.warpAffine(img, M, (self.image_h, self.image_w))

        if self.store_frames:
            os.makedirs("./image_dump", exist_ok=True)
            cv2.imwrite("./image_dump/"+str(self.ctr)+".jpg", img)
            self.ctr+=1
        state = np.expand_dims(img, axis=0)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.model.train()
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.model.train()
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, test_mode=False):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = self.exploration - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon and not test_mode:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            self.model.eval()
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(game, args, writer):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK']='True'

    sum_rewards = 0
    sum_short_loss = 0
    total_score = 0
    record = 0
    loop_ctr = 0
    agent = Agent(args)
    dummy_input = torch.randn(1, 1, args.height, args.width).to(agent.device)
    writer.add_graph(agent.model, dummy_input)
    print("Now playing")

    while agent.n_games != args.max_games:
        loop_ctr += 1
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.playStep(final_move)
        state_new = agent.get_state(game)
        sum_rewards += reward
        writer.add_scalar('Reward/curr_reward', reward, loop_ctr)

        # train short memory
        short_loss = agent.train_short_memory(state_old, final_move, reward, state_new, [done])
        writer.add_scalar('Game/Short_Episodes', loop_ctr, loop_ctr)
        sum_short_loss += short_loss

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if loop_ctr%25 == 0:
            writer.add_scalar('Loss/Short_train', sum_short_loss/loop_ctr, loop_ctr)
            writer.add_scalar('Reward/mean_reward', sum_rewards/loop_ctr, loop_ctr)

        if done:
            # train long memory, plot result
            game.gameReboot()
            agent.n_games += 1
            long_loss = agent.train_long_memory()
            writer.add_scalar('Loss/Long_train', long_loss, agent.n_games)
            writer.add_scalar('Game/Episodes', agent.n_games, agent.n_games)

            if score > record:
                record = score
                # agent.model.save()
                agent.model.save(model_folder_path="./model"+hyper_params+dstr)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            writer.add_scalar('Score/High_Score', record, agent.n_games)

            total_score += score
            mean_score = total_score / agent.n_games
            writer.add_scalars('Score', {'Curr_Score':score, 'Mean_Score': mean_score}, agent.n_games)
            write_model_params(agent.model, agent.n_games, writer)

    writer.add_hparams(hparam_dict=vars(args),
                        metric_dict={'long_loss_loss': long_loss,
                                     'mean_short_loss': sum_short_loss/loop_ctr,
                                     'mean_reward': sum_rewards/loop_ctr,
                                     'high_score': record,
                                     'mean_score': mean_score
                                     })

def test(game, args):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    record = 0
    agent = Agent(args)
    print("Now playing")
    
    while agent.n_games != args.max_games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.playStep(final_move)
        if done:
            game.gameReboot()
            if score > record:
                record = score
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RL Agent for Doodle Jump')
    parser.add_argument("--macos", action="store_true", help="select model to train the agent")
    parser.add_argument("--human", action="store_true", help="playing the game manually without agent")
    parser.add_argument("--test", action="store_true", help="playing the game with a trained agent")
    parser.add_argument("-d", "--difficulty", type=str, default="EASY", choices=["EASY", "MEDIUM", "HARD"], help="select difficulty of the game")
    parser.add_argument("-m", "--model", type=str, default="dqn", choices=["dqn", "drqn"], help="select model to train the agent")
    parser.add_argument("-p", "--model_path", type=str, help="path to weights of an earlier trained model")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="set learning rate for training the model")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help="set discount factor for q learning")
    parser.add_argument("--max_memory", type=int, default=10000, help="Buffer memory size for long training")
    parser.add_argument("--store_frames", action="store_true", help="store frames encountered during game play by agent")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for long training")
    parser.add_argument("--reward_type", type=int, default=1, choices=[1, 2, 3, 4], help="types of rewards formulation")
    parser.add_argument("--exploration", type=int, default=40, help="number of games to explore")
    parser.add_argument("--height", type=int, default=80, help="set the image height post resize")
    parser.add_argument("--width", type=int, default=80, help="set the image width post resize")
    parser.add_argument("--server", action="store_true", help="when training on server add this flag")
    parser.add_argument("--seed", type=int, default=42, help="change seed value for creating game randomness")
    parser.add_argument("--max_games", type=int, default=1000, help="set the max number of games to be played by the agent")
    args = parser.parse_args()
    
    game = DoodleJump(difficulty=args.difficulty, server=args.server, reward_type=args.reward_type)

    if args.human:
        game.run()
    elif args.test:
        test(game, args)
    else:
        hyper_params = "_d_"+args.difficulty+"_m_"+args.model+"_lr_"+str(args.learning_rate)+"_g_"+str(args.gamma)+"_mem_"+str(args.max_memory)+"_batch_"+str(args.batch_size)
        arg_dict = vars(args)

        dstr = datetime.datetime.now().strftime("_dt-%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir="model"+hyper_params+dstr)
        writer.add_text('Model Parameters: ', str(arg_dict), 0)
        
        train(game, args, writer)
            
