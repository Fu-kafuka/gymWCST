# Wisconsin Card Sorting Inspired Task (WCST) Gym Environment

import gymnasium as gym
from gymnasium import spaces

import cv2
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from enum import IntEnum
import random
import os

class color(IntEnum):
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3

class shape(IntEnum):
    CIRCLE = 0
    TRIANGLE = 1
    SQUARE = 2
    STAR = 3

class number(IntEnum):
    ONE = 0
    TWO = 1
    THREE = 2
    FOUR = 3

class question(IntEnum):
    COLOR = 0
    SHAPE = 1
    NUMBER = 2

class WCST(gym.Env):
    metadata = {'render.modes': ['human']}
    eplenmax = 60

    height = 256
    width = 512

    shapesize = 30
    fixedChoice = True
    consecutiveCorrectMax = 6

    def __init__(self, fixedChoice=True):
        self.action_space = spaces.Discrete(4) # 4択の選択肢

        # observation_space = [4枚のカード(色、形、数), 問題カード(色、形、数), 結果通知]を一行に並べたもの (非視覚モード)
        # self.observation_space = spaces.Tuple([spaces.Box(0,3,(4,3),int ),spaces.Box(0,3,(4,),int ), spaces.Discrete(2)]) # 4種類のカード(色、形、数) 数は固定 + 結果通知
        self.observation_space = spaces.Box(-1,3, (5*3+1,) ,int ) 
        self.reward_range = (-1, 1)
        self.fixedChoice = fixedChoice


    def reset(self, seed=None):
        self.lastCorrect = False
        self.seed(seed)
        self.questionMode = random.choice(list(question))
        self.consecutiveCorrect = 0
        
        self.obs = self.calculateNext()
        self.obs[-1] = 0
        self.eplen = 0
        returnObs = np.array(list(self.obsflatten(self.obs)))
        return returnObs, {}

    def step(self, action):
        # 正解を取得 -> index

        if self.questionMode == question.NUMBER:
            key = self.obs[1][0]
            index = self.indexSearch(key, self.obs[0], 0)
        elif self.questionMode == question.SHAPE:
            key = self.obs[1][1]
            index = self.indexSearch(key, self.obs[0], 1)
        elif self.questionMode == question.COLOR:
            key = self.obs[1][2]
            index = self.indexSearch(key, self.obs[0], 2)

        # 正解かどうか判定
        if action == index:
            self.lastCorrect = True
            self.consecutiveCorrect += 1
            reward = 1
        else:
            self.lastCorrect = False
            self.consecutiveCorrect = 0
            reward = -1
        
        if self.eplen < self.eplenmax:
            done = False
            truncated = False
        else:
            done = True
            truncated = True

        self.eplen += 1

        # 連続正解時の問題モード切替
        if self.consecutiveCorrect >= self.consecutiveCorrectMax:
            self.questionMode = random.choice(list(question))
            self.consecutiveCorrect = 0

        # return処理
        self.obs = self.calculateNext()
        if self.lastCorrect:
            self.obs[-1] = 1
        else:
            self.obs[-1] = -1

        returnObs = np.array(list(self.obsflatten(self.obs)))
        return returnObs, reward, done, truncated, {}
        
    def render(self, mode='human'): # 描画
        img = np.full((self.height, self.width, 3),128,dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (130, 130), (255, 255, 255), thickness=-1)
        cv2.rectangle(img, (140, 30), (240, 130), (255, 255, 255), thickness=-1)
        cv2.rectangle(img, (250, 30), (350, 130), (255, 255, 255), thickness=-1)
        cv2.rectangle(img, (360, 30), (460, 130), (255, 255, 255), thickness=-1)
        cv2.putText(img, "0", (75, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, "1", (185, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, "2", (295, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img, "3", (405, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 図形の描画
        self.renderShape(img, self.obs[0][0][1], self.obs[0][0][2], self.obs[0][0][0], 80, 80)
        self.renderShape(img, self.obs[0][1][1], self.obs[0][1][2], self.obs[0][1][0], 190, 80)
        self.renderShape(img, self.obs[0][2][1], self.obs[0][2][2], self.obs[0][2][0], 300, 80)
        self.renderShape(img, self.obs[0][3][1], self.obs[0][3][2], self.obs[0][3][0], 410, 80)

        # 問題の描画
        cv2.rectangle(img, (360, 150), (460, 250), (255, 255, 255), thickness=-1) 
        self.renderShape(img, self.obs[1][1], self.obs[1][2], self.obs[1][0], 410, 200)

        # 結果の描画
        if self.obs[-1] == 1:
            cv2.putText(img, "Correct", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif self.obs[-1] == -1:
            cv2.putText(img, "Incorrect", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.pause(0.01)
        #cv2.imshow('WCST', img)
        #cv2.waitKey(0)

    def renderShape(self, img, shape, color, shapeNumber, x, y): # 図形の描画の座標制御
        if shapeNumber == number.ONE:
            self.renderShapeIndivisual(img, shape, color, x, y)
        elif shapeNumber == number.TWO:
            self.renderShapeIndivisual(img, shape, color, x , y - 30)
            self.renderShapeIndivisual(img, shape, color, x , y + 30)
        elif shapeNumber == number.THREE:
            self.renderShapeIndivisual(img, shape, color, x + 30 , y - 30)
            self.renderShapeIndivisual(img, shape, color, x      , y)
            self.renderShapeIndivisual(img, shape, color, x - 30 , y + 30)
        elif shapeNumber == number.FOUR:
            self.renderShapeIndivisual(img, shape, color, x + 30 , y - 30)
            self.renderShapeIndivisual(img, shape, color, x - 30 , y - 30)
            self.renderShapeIndivisual(img, shape, color, x + 30 , y + 30)
            self.renderShapeIndivisual(img, shape, color, x - 30 , y + 30)
            

    def renderShapeIndivisual(self, img, randerShape, color, x, y): # 個別の図形の描画
        if randerShape == shape.CIRCLE:
            cv2.circle(img, (x, y), self.shapesize//2, self.getRenderColor(color), thickness=-1)
            cv2.circle(img, (x, y), self.shapesize//2, self.getRenderColor((0,0,0)))
        elif randerShape == shape.TRIANGLE:
            points = np.array([[int(x - self.shapesize / 2) , int(y - self.shapesize / 2)], [int(x + self.shapesize / 2), int(y - self.shapesize / 2)], [x, int(y + self.shapesize / 2)]])
            cv2.fillConvexPoly(img, points, self.getRenderColor(color))
            cv2.polylines(img, [points], True, self.getRenderColor((0,0,0)))
        elif randerShape == shape.SQUARE:
            cv2.rectangle(img, (int(x - self.shapesize / 2), int(y - self.shapesize / 2)), (int(x + self.shapesize / 2), int(y + self.shapesize / 2)), self.getRenderColor(color), thickness=-1)
            cv2.rectangle(img, (int(x - self.shapesize / 2), int(y - self.shapesize / 2)), (int(x + self.shapesize / 2), int(y + self.shapesize / 2)), self.getRenderColor((0,0,0)))
        elif randerShape == shape.STAR:
            angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
            points = np.array([[int(self.shapesize / 2 * np.cos(angle) + x), int(self.shapesize / 2 * np.sin(angle) + y)] for angle in angles])
            cv2.fillConvexPoly(img, points, self.getRenderColor(color)) # 星形の描画ができない
            cv2.polylines(img, [points], True, self.getRenderColor((0,0,0)))

    def getRenderColor(self, shapeColor): # 色の取得
        if shapeColor == color.RED:
            return (0, 0, 255)
        elif shapeColor == color.GREEN:
            return (0, 255, 0)
        elif shapeColor == color.BLUE:
            return (255, 0, 0)
        elif shapeColor == color.YELLOW:
            return (0, 255, 255)
        else:
            return (0, 0, 0)

    def close(self):
        cv2.destroyAllWindows()

    def calculateNext(self): # 次のカードを計算
        obs = [[0] * 3 for i in range(4)]
        tempNumber = list(number)
        tempShape = list(shape)
        tempColor = list(color)
        
        if self.fixedChoice == False: # 変動選択肢モード
            obs[0][0] = random.choice(tempNumber)
            obs[0][1] = random.choice(tempShape)
            obs[0][2] = random.choice(tempColor)

            tempNumber.remove(obs[0][0])
            tempShape.remove(obs[0][1])
            tempColor.remove(obs[0][2])

            obs[1][0] = random.choice(tempNumber)
            obs[1][1] = random.choice(tempShape)
            obs[1][2] = random.choice(tempColor)

            tempNumber.remove(obs[1][0])
            tempShape.remove(obs[1][1])
            tempColor.remove(obs[1][2])

            obs[2][0] = random.choice(tempNumber)
            obs[2][1] = random.choice(tempShape)
            obs[2][2] = random.choice(tempColor)

            tempNumber.remove(obs[2][0])
            tempShape.remove(obs[2][1])
            tempColor.remove(obs[2][2])

            obs[3][0] = tempNumber[0]
            obs[3][1] = tempShape[0]
            obs[3][2] = tempColor[0]
        else: # 固定選択肢モード
            obs[0][0] = number.ONE
            obs[0][1] = shape.CIRCLE
            obs[0][2] = color.RED

            obs[1][0] = number.TWO
            obs[1][1] = shape.TRIANGLE
            obs[1][2] = color.GREEN
            
            obs[2][0] = number.THREE
            obs[2][1] = shape.SQUARE
            obs[2][2] = color.BLUE

            obs[3][0] = number.FOUR
            obs[3][1] = shape.STAR
            obs[3][2] = color.YELLOW

        obs = [obs, [random.choice(list(number)), random.choice(list(shape)), random.choice(list(color))], 0]

        return obs

    def indexSearch(self, key, array, column):
        for i in range(len(array)):
            if array[i][column] == key:
                return i
        return -1

    
    def obsflatten(self, obs):
        for el in obs:
            if isinstance(el, list):
                yield from self.obsflatten(el)
            else:
                el = int(el)
                yield el

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return True    
        
    

#テスト
env = WCST()
#print(env.observation_space)
env.seed(1)
env.reset()
while True:
    env.render()
    action = input()
    if action == "q":
        break
    obser, rew, done, _ = env.step(int(action))
    print(obser)
    print(rew)
    if done:
        break

# from sb3_contrib import RecurrentPPO
# from stable_baselines3.common.evaluation import evaluate_policy

# env = WCST()
# model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)

# if os.path.exists("ppo_wcst"):
#   model.load("ppo_wcst")

# for i in range(100):
#   model.learn(total_timesteps=10000)
#   mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#   print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
#   model.save("ppo_wcst")


