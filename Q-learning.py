import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os


# set the rows and columns length
BOARD_ROWS = 4
BOARD_COLS = 6

# initalise start, win and lose states
START = (0, 0)
WIN_STATE = (3, 5)

# class state defines the board and decides reward, end and next position


class State:
    def __init__(self, state=START):
        self.state = state
        self.isEnd = False

    def getReward(self):
        if self.state == WIN_STATE:
            return 100
        else:
            return 0

    def isEndFunc(self):
        if self.state == WIN_STATE:
            self.isEnd = True

    def nxtPosition(self, action):
        if action == 0:
            nxtState = (self.state[0] - 1, self.state[1])  # up
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1])  # down
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1)  # left
        else:
            nxtState = (self.state[0], self.state[1] + 1)  # right

        if (nxtState[0] >= 0) and (nxtState[0] < BOARD_ROWS) and (nxtState[1] >= 0) and (nxtState[1] < BOARD_COLS):
            return nxtState

        return self.state


# class agent to implement reinforcement learning through grid
class Agent:
    def __init__(self):
        # inialise states and actions
        self.states = []
        self.actions = [0, 1, 2, 3]  # up, down, left, right
        self.State = State()
        self.alpha = 0.8
        self.epsilon = 1
        self.isEnd = self.State.isEnd

        # array to retain reward values for plot
        self.plot_reward = []

        # initalise Q values as a dictionary for current and new
        self.Q = {}
        self.new_Q = {}
        self.rewards = 0

        # initalise all Q values across the board to 0, print these values
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(len(self.actions)):
                    self.Q[(i, j, k)] = 0
                    self.new_Q[(i, j, k)] = 0
        # print(self.Q)

    # method to choose action with Epsilon greedy policy, and move to next state

    def Action(self):
        # random value vs epsilon
        rnd = random.random()
        # set arbitraty low value to compare with Q values to find max
        mx_nxt_reward = -10
        action = None

        # find max Q value over actions
        if rnd > self.epsilon:
            # iterate through actions, find Q-value and choose best
            for k in self.actions:
                i, j = self.State.state
                nxt_reward = self.Q[(i, j, k)]

                if nxt_reward >= mx_nxt_reward:
                    action = k
                    mx_nxt_reward = nxt_reward
        # else choose random action
        else:
            action = np.random.choice(self.actions)

        # select the next state based on action chosen
        position = self.State.nxtPosition(action)
        return position, action

    # Q-learning Algorithm
    def Q_Learning(self, episodes):
        df_factor_path = "discount_factor.csv"
        df_factor = pd.read_csv(df_factor_path, index_col="Grid 座標")
        x = 0
        # iterate through best path for each episode
        while x < episodes:
            # check if state is end
            if self.isEnd:
                # get current rewrard and add to array for plot
                reward = self.State.getReward()
                self.rewards += reward
                self.plot_reward.append(self.rewards)

                # get state, assign reward to each Q_value in state
                i, j = self.State.state
                for a in self.actions:
                    self.new_Q[(i, j, a)] = round(reward, 3)

                # reset state
                self.State = State()
                self.isEnd = self.State.isEnd

                # set rewards to zero and iterate to next episode
                self.rewards = 0
                x += 1
            else:
                mx_nxt_value = -10
                next_state, action = self.Action()
                i, j = self.State.state
                reward = self.State.getReward()
                # add reward to rewards for plot
                self.rewards += reward

                # iterate through actions to find max Q value for action based on next state action
                for a in self.actions:
                    now_index = str(i) + "_" + str(j)
                    df = df_factor.loc[now_index, "discount factor"]
                    df = df.split("/")

                    nxtStateAction = (next_state[0], next_state[1], a)
                    q_value = (1 - self.alpha) * self.Q[(i, j, action)] + self.alpha * (
                        reward + float(df[a]) * self.Q[nxtStateAction]
                    )

                    if q_value >= mx_nxt_value:
                        mx_nxt_value = q_value

                # next state is now current state, check if end state
                self.State = State(state=next_state)
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
                self.new_Q[(i, j, action)] = round(mx_nxt_value, 3)
            self.Q = self.new_Q.copy()
        # print(self.Q)

    def plot(self, episodes):
        plt.plot(self.plot_reward)
        plt.show()

    def showValues(self, arr):
        outArr = [[0 for _ in range(0, BOARD_COLS)] for _ in range(0, BOARD_ROWS)]
        for i in range(0, BOARD_ROWS):
            print("-----------------------------------------------------------------------------------------------------------")
            out = "| "
            for j in range(0, BOARD_COLS):
                mx_nxt_value = -10
                for a in self.actions:
                    nxt_value = self.Q[(i, j, a)]
                    if nxt_value >= mx_nxt_value:
                        mx_nxt_value = nxt_value
                out += str(mx_nxt_value).ljust(6) + " | "
                outArr[i][j] = str(mx_nxt_value)
            print(out)
        print("-----------------------------------------------------------------------------------------------------------")

        if START[0] - 1 < 0:
            arr[0] = -1
        else:
            arr[0] = outArr[START[0] - 1][START[1]]

        if START[0] + 1 >= BOARD_ROWS:
            arr[1] = -1
        else:
            arr[1] = outArr[START[0] + 1][START[1]]
        
        if START[1] - 1 < 0:
            arr[2] = -1
        else:
            arr[2] = outArr[START[0]][START[1] - 1]

        if START[1] + 1 >= BOARD_COLS:
            arr[3] = -1
        else:
            arr[3] = outArr[START[0]][START[1] + 1]

        print(arr)

if __name__ == "__main__":
    # create agent for 15,000 episdoes implementing a Q-learning algorithm plot and show values.
    episodes = 10000
    ag = Agent()

    filename = "q_table"
    q_value = np.zeros((4), dtype=np.float64)

    print(START, WIN_STATE)

    ag.Q_Learning(episodes)
    ag.showValues(q_value)
    np.save(os.path.join(filename), q_value)