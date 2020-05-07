import numpy as np
import cv2
import random
import matplotlib.pyplot as plot


class Tic_Tac_Toe:
    grid=np.zeros((3,3))
    turn=1
    row_sum=np.zeros((3,1))
    col_sum = np.zeros((3, 1))
    dia_sum=np.zeros((2, 1))
    screen=np.zeros((3,3))
    moves_made=0
    possible_actions=[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]

    def reset(self):
        self.grid = np.zeros((3, 3))
        self.turn = 1
        self.moves_made=0
        self.row_sum = np.zeros((3, 1))
        self.col_sum = np.zeros((3, 1))
        self.dia_sum = np.zeros((2, 1))
        self.possible_actions=[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
        self.make_screen()

    def make_move(self,action):
        if(self.grid[action[0],action[1]]==0):
            self.grid[action[0],action[1]]=self.turn
            self.update_sums(action)
        self.moves_made+=1

    def update_sums(self,action):
        self.row_sum[action[0]]+=self.turn
        self.col_sum[action[1]]+=self.turn
        if(action[0]==action[1]):
            self.dia_sum[0]+=self.turn
        if(action[0]==(2-action[1])):
            self.dia_sum[1]+= self.turn

    def game_status(self):
        if((self.row_sum==3*self.turn).any() or (self.col_sum==3*self.turn).any()  or (self.dia_sum==3*self.turn).any() ):
            return(1)
        return(0)

    def make_screen(self):
        self.screen=np.ones((600,600,3))*255
        self.screen[5:-5,5:-5,:]=0
        self.screen[195:205,:,:]=255
        self.screen[395:405,:,:]=255
        self.screen[:,195:205, :] = 255
        self.screen[:,395:405, :] = 255

    def display_move(self,action):
        self.screen[action[1]*200+5:(action[1]+1)*200-5,action[0]*200+5:(action[0]+1)*200-5,self.turn]=255

    def no_callback(self,event,x,y,flags,params):
        pass

    def get_action_screen_pvp(self,event,x,y,flags,params):
        if(event==cv2.EVENT_LBUTTONDOWN):
            action=[int(x/200),int(y/200)]
            if(self.grid[action[0],action[1]]==0):
                self.display_move(action)
                self.make_move(action)
                cv2.imshow("Tic-Tac-Toe", self.screen)
                if (self.game_status()):
                    if (self.turn == 1):
                        print("Player 1 Won")
                    else:
                        print("Player 2 Won")
                    cv2.setMouseCallback("Tic-Tac-Toe", self.no_callback)
                elif(self.moves_made==9):
                    print("Draw")
                    cv2.setMouseCallback("Tic-Tac-Toe", self.no_callback)
                self.turn *= -1

    def get_action_screen_pve(self,event,x,y,flags,params):
        if(event==cv2.EVENT_LBUTTONDOWN):
            action=[int(x/200),int(y/200)]
            if(self.grid[action[0],action[1]]==0):
                self.display_move(action)
                self.make_move(action)
                self.possible_actions.remove(action)
                cv2.imshow("Tic-Tac-Toe", self.screen)
                if (self.game_status()):
                    if (self.turn == 1):
                        print("Player 1 Won")
                    else:
                        print("Player 2 Won")
                    cv2.setMouseCallback("Tic-Tac-Toe", self.no_callback)
                elif(self.moves_made==9):
                    print("Draw")
                    cv2.setMouseCallback("Tic-Tac-Toe", self.no_callback)
                else:
                    self.turn *= -1
                    action = self.get_action(model,self.get_state())
                    actions = np.array(self.possible_actions)
                    q_values = model.predict(np.concatenate((np.tile(self.get_state(), (9 - self.moves_made, 1)), actions), axis=1))
                    print(actions,q_values)
                    self.possible_actions.remove(action)
                    self.make_move(action)
                    self.display_move(action)
                    cv2.imshow("Tic-Tac-Toe", self.screen)
                    if (self.game_status()):
                        if (self.turn == 1):
                            print("Player 1 Won")
                        else:
                            print("Player 2 Won")
                        cv2.setMouseCallback("Tic-Tac-Toe", self.no_callback)
                        pass
                    self.turn*=-1

    def get_state(self):
        return(self.grid.reshape(-1)*self.turn)

    def get_action(self,model,state):
        actions=np.array(self.possible_actions)
        q_values=model.predict(np.concatenate((np.tile(state, (9-self.moves_made, 1)), actions), axis=1))
        action=q_values==q_values.max()
        action=list(actions[action.reshape(-1)][0])
        return(action)


    def get_random_action(self):
        action=random.choice(self.possible_actions)
        return(action)

    def run_game_pvp(self):
        self.reset()
        cv2.imshow("Tic-Tac-Toe",self.screen)
        cv2.setMouseCallback("Tic-Tac-Toe", self.get_action_screen_pvp)
        cv2.waitKey(0)

    def run_game_pve(self,model):
        self.model=model
        self.reset()
        cv2.imshow("Tic-Tac-Toe", self.screen)
        action = self.get_action(model, self.get_state())
        self.possible_actions.remove(action)
        self.make_move(action)
        self.display_move(action)
        cv2.imshow("Tic-Tac-Toe", self.screen)
        self.turn *= -1
        cv2.setMouseCallback("Tic-Tac-Toe", self.get_action_screen_pve)
        cv2.waitKey(0)

    def run_game_eve(self,model):
        self.reset()
        cv2.imshow("Tic-Tac-Toe", self.screen)
        cv2.waitKey(500)
        while(True):
            action = self.get_action(model, self.get_state())
            print(
                model.predict(
                    np.expand_dims(np.concatenate((self.get_state(), [x / 2 for x in action]), axis=0), axis=0)))
            self.possible_actions.remove(action)
            self.make_move(action)
            self.display_move(action)
            if (self.game_status()):
                if (self.turn == 1):
                    print("Player 1 Won")
                else:
                    print("Player 2 Won")
                break
            self.turn *= -1
            cv2.imshow("Tic-Tac-Toe", self.screen)
            cv2.waitKey(500)


    def train(self,model_offline,episodes,refresh_model,gamma,epsilon_decay,epsilon_min,batch_size,buffer_size):
        model_offline.compile(optimizer=rmsprop(learning_rate=0.01), loss='mean_squared_error')
        model_online=models.clone_model(model_offline)
        model_online.compile(optimizer=rmsprop(learning_rate=0.01), loss='mean_squared_error')
        model_offline.set_weights(model_online.get_weights())
        epsilon = 1-epsilon_min
        experiances=[]
        history=[]
        for episode in range(buffer_size):
            self.get_experiance(model_online,epsilon+epsilon_min,experiances)
            self.learn(model_online,model_offline,experiances,batch_size,gamma)
        for episode in range(buffer_size,episodes+1):
            self.get_experiance(model_online,epsilon+epsilon_min,experiances)
            del experiances[:self.moves_made]
            epsilon*=epsilon_decay
            loss=self.learn(model_online,model_offline,experiances,batch_size,gamma)
            history.append(loss[0])
            if(episode%refresh_model==0):
                model_offline.set_weights(model_online.get_weights())
                print("Episode: ", episode, " Epsilon: ", epsilon+epsilon_min)
        plot.plot(history[2000:])
        plot.show()

    def get_experiance(self,model,epsilon,experiances):
        self.reset()
        while (True):
            state = self.get_state()
            if (np.random.random() >epsilon):
                action = self.get_action(model, state)
            else:
                action = self.get_random_action()
            self.make_move(action)
            self.display_move(action)
            reward = self.game_status()
            state_action = np.concatenate((state, [x/2 for x in action]), axis=0)
            self.possible_actions.remove(action)
            if (reward != 0):
                experiances.append([state_action,reward])
                break
            elif (self.moves_made == 9):
                experiances.append([state_action,reward])
                break
            self.turn *= -1
            state = self.get_state()
            action = self.get_action(model, state)
            next_state_action = np.concatenate((state, [x/2 for x in action]), axis=0)
            experiances.append([state_action,reward,next_state_action])


    def learn(self,online_model,offline_model,experiances,batch_size,gamma):
        experiance_batch=random.choices(experiances,k=batch_size)
        state_batch=[]
        target_batch=[]
        for experiance in experiance_batch:
            if(len(experiance)==2):
                state_batch.append(experiance[0])
                target_batch.append(experiance[1])
            else:
                target_batch.append(experiance[1]-gamma*(offline_model.predict(np.array([experiance[2]])))[0,0])
                state_batch.append(experiance[0])
        history=online_model.fit(np.array(state_batch),np.array(target_batch),verbose=0)
        return (history.history['loss'])





from keras.optimizers import rmsprop
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(50, input_dim=11, activation='sigmoid'))
model.add(layers.Dense(20, activation='sigmoid'))
model.add(layers.Dense(1, activation='linear'))

Game=Tic_Tac_Toe()
#Game.train_model_e_greedy(model,10000,100,.95,.99995)
Game.train(model_offline=model,episodes=10000,refresh_model=500,gamma=1,epsilon_decay=.9995,epsilon_min=0.05,batch_size=500,buffer_size=500)
while(True):
    Game.run_game_pve(model)
