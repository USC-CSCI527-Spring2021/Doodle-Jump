import torch
import numpy as np
import torch.nn as nn
import cv2
import random
import os
import datetime
import argparse
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from game.doodlejump import DoodleJump
from model.networks import ActorCritic
from model.a2cTrainer import A2CLearner

def mish(input):
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

def t(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float() #.reshape(6400, 1)

class Runner():
    def __init__(self, game):
        self.game = game
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.mean_reward = 0
        self.mean_score = 0
        self.episode_rewards = []
        self.total_score = 0
        ''''''
        self.n_games = 0
        self.epsilon = 0
        self.ctr = 1
        seed = args.seed
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
        self.image_c = args.channels
        # self.memory = deque(maxlen=args.max_memory)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.record = 0
    
    def reset(self):
        self.episode_reward = 0
        self.done = False
        self.state = None
        self.game.gameReboot()

    def preprocess(self, state):
        
        # resize the image and then rotate
        img = cv2.resize(state, (self.image_w, self.image_h))
        M = cv2.getRotationMatrix2D((self.image_w / 2, self.image_h / 2), 270, 1.0)
        img = cv2.warpAffine(img, M, (self.image_h, self.image_w))

        if self.store_frames:
            os.makedirs("./image_dump", exist_ok=True)
            cv2.imwrite("./image_dump/"+str(self.ctr)+".jpg", img)
            self.ctr+=1
            
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        if self.image_c == 1:
            # convert the image to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # normalize the image with imagenet mean and std values
            img = ((img/255.0) - np.mean(imagenet_mean))/np.mean(imagenet_std)
        else:
            # normalize the image with imagenet mean and std values
            img = ((img/255.0) - imagenet_mean)/imagenet_std
            # change the shape from WxHxC to CxHxW for pytorch tensor
            img = img.transpose((2, 0, 1))
        
        # Add a axis for converting image to shape: 1xCxHxW
        img = np.expand_dims(img, axis=0)
        return img

    def get_state(self):
        state = self.game.getCurrentFrame()
        return self.preprocess(state)


    def run(self, max_steps, memory=None):
        loop_ctr = 0
        if not memory or len(memory) > args.max_memory:
            memory = []
        for _ in range(max_steps):
            if self.done: self.reset()
            state_old = self.get_state()
            dists = actorcritic(t(state_old).to(self.device))[0]
            actions = dists.sample().detach().cpu().data.numpy()
            actions_clipped = np.clip(actions, -1, 1) #self.env.action_space.low.min(), env.action_space.high.max())
            
            final_move = [0,0,0]
            final_move[np.argmax(actions_clipped)] = 1
            reward, self.done, score = self.game.playStep(final_move)
            
            next_state = self.get_state()
            memory.append((actions, reward, self.state, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward
            self.mean_reward = self.episode_reward/self.steps

            writer.add_scalar("Reward/mean_reward", self.mean_reward, global_step=self.steps)
            
            if self.done:
                self.n_games += 1
                learner.lastlaegam = 0
                self.episode_rewards.append(self.episode_reward)
                if len(self.episode_rewards) % 10 == 0:
                    print("episode:", len(self.episode_rewards), ", episode reward:", self.episode_reward)
                writer.add_scalar("Reward/episode_reward", self.episode_reward, global_step=self.steps)

                if score > self.record:
                    self.record = score
                    # save the best model yet
                    actorcritic.save(file_name="a2c_model_best.pth", model_folder_path="./model"+hyper_params+dstr)
                    # actorcritic.save(file_name="critic_model_best.pth", model_folder_path="./model"+hyper_params+dstr)
                
                if self.n_games%100 == 0:
                    # save model per 100 games
                    actorcritic.save(file_name="a2c_model_"+str(self.n_games)+".pth", model_folder_path="./model"+hyper_params+dstr)
                    # actorcritic.save(file_name="critic_model_"+str(self.n_games)+".pth", model_folder_path="./model"+hyper_params+dstr)

                print('Game', self.n_games, 'Score', score, 'Record:', self.record)
                writer.add_scalar('Score/High_Score', self.record, self.n_games)

                self.total_score += score
                self.mean_score = self.total_score / agent.n_games
                writer.add_scalar('Score/Mean_Score', self.mean_score, self.n_games)

        return memory


def test(game, args):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
    record = 0
    agent = Runner(game)
    print("Now testing")
    
    while agent.n_games != args.max_games:
        state_old = agent.get_state()
        dists = actorcritic(t(state_old).to(agent.device))[0]
        actions = dists.sample().detach().cpu().data.numpy()
        actions_clipped = np.clip(actions, -1, 1) #self.env.action_space.low.min(), env.action_space.high.max())
        
        final_move = [0,0,0]
        final_move[np.argmax(actions_clipped)] = 1
        reward, done, score = game.playStep(final_move)
        # final_move = agent.get_action(state_old, test_mode=True)
        # reward, done, score = game.playStep(final_move)
        if done:
            agent.n_games += 1
            game.gameReboot()
            if score > record:
                record = score
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent for Doodle Jump')
    parser.add_argument("--macos", action="store_true", help="select model to train the agent")
    parser.add_argument("--human", action="store_true", help="playing the game manually without agent")
    parser.add_argument("--test", action="store_true", help="playing the game with a trained agent")
    parser.add_argument("-d", "--difficulty", type=str, default="EASY", choices=["EASY", "MEDIUM", "HARD"], help="select difficulty of the game")
    parser.add_argument("-m", "--model", type=str, default="a2c", choices=["a2c"], help="select model to train the agent")
    parser.add_argument("-p", "--model_path", type=str, help="path to weights of an earlier trained model")
    # parser.add_argument("-cp", "--critic_path", type=str, help="path to weights of an earlier trained model")
    parser.add_argument("-alr", "--actor_lr", type=float, default=4e-4, help="set learning rate for training the model")
    parser.add_argument("-clr", "--critic_lr", type=float, default=4e-3, help="set learning rate for training the model")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help="set discount factor for q learning")
    parser.add_argument("--max_memory", type=int, default=10000, help="Buffer memory size for long training")
    parser.add_argument("--store_frames", action="store_true", help="store frames encountered during game play by agent")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for long training")
    parser.add_argument("--reward_type", type=int, default=1, choices=[1, 2, 3, 4], help="types of rewards formulation")
    parser.add_argument("--exploration", type=int, default=40, help="number of games to explore")
    parser.add_argument("--channels", type=int, default=1, help="set the image channels for preprocessing")
    parser.add_argument("--height", type=int, default=80, help="set the image height post resize")
    parser.add_argument("--width", type=int, default=80, help="set the image width post resize")
    parser.add_argument("--server", action="store_true", help="when training on server add this flag")
    parser.add_argument("--seed", type=int, default=42, help="change seed value for creating game randomness")
    parser.add_argument("--max_games", type=int, default=1000, help="set the max number of games to be played by the agent")
    args = parser.parse_args()

    game = DoodleJump(difficulty=args.difficulty, server=args.server, reward_type=args.reward_type)
    agent = Runner(game)
    # env = gym.make("Pendulum-v0")
    hyper_params = "_d_"+args.difficulty+"_m_"+args.model+"_alr_"+str(args.actor_lr)+"_clr_"+str(args.critic_lr)+"_g_"+str(args.gamma)+"_mem_"+str(args.max_memory)+"_batch_"+str(args.batch_size)
    dstr = datetime.datetime.now().strftime("_dt-%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(log_dir="./model"+hyper_params+dstr)
    arg_dict = vars(args)
    writer.add_text('Model Parameters: ', str(arg_dict), 0)

    # config
    state = agent.get_state() #env.observation_space.shape[0]
    n_actions = 3 #env.action_space.shape[0]
    
    actorcritic = ActorCritic(state.shape[0], n_actions, activation=Mish).to(agent.device)
    # critic = Critic(state.shape[0], activation=Mish).to(agent.device)
    if (args.model_path) or args.test:
        actorcritic.load_state_dict(torch.load(args.model_path))
        # critic.load_state_dict(torch.load(args.critic_path))

    if args.test:
        test(game, args)
    else:

        learner = A2CLearner(actorcritic, agent.device, gamma=args.gamma, entropy_beta=0,
                    actor_lr=args.actor_lr, critic_lr=args.critic_lr, max_grad_norm=0.5, batch_size = args.batch_size)
        # runner = Runner(env)
        ###########
        steps_on_memory = 16
        episodes = 500
        episode_length = 200
        total_steps = (episode_length*episodes)//steps_on_memory
        # record = 0
        while agent.n_games != args.max_games:
            memory = agent.run(steps_on_memory)
            learner.learn(memory, agent.steps, writer, discount_rewards=False)
        
        writer.add_hparams(hparam_dict=vars(args),
                            metric_dict={'mean_reward': agent.mean_reward,
                                            'high_score': agent.record,
                                            'mean_score': agent.mean_score
                                            })