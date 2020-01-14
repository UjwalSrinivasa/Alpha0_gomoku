
import random
import numpy as np

from collections import defaultdict
from collections import deque

from game import Board, Game
from nneural_network import neural_net_p

from montecarlo import monte_tree_search




class train_net():
    def __init__(self):
        
        self.board_width = 11
        self.board_height = 11
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.learn_rate = 0.009
        self.lr_multiplier = 1.0  
        self.temp = 1.0 
        self.n_playout = 400 
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 100 
        self.data_buffer = deque(maxlen=self.buffer_size)        
        self.play_batch_size = 1 
        self.epochs = 50 
        self.kl_targ = 0.02
        self.check_freq = 1000
        self.game_batch_num = 50000000
        self.best_win_ratio = 0.0
    
        
        
        self.pv_net = neural_net_p(self.board_width, self.board_height, self.n_in_row) 
        self.mcts_player =monte_tree_search(self.pv_net.pv_fn, 
                                      c_puct=self.c_puct, 
                                      n_playout=self.n_playout, is_selfplay=1)

    def rotate_flip(self, play_data):
        e_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1,2,3,4]:
                
                e_state = np.array([np.rot90(s,i) for s in state])
                e_mcts_p = np.rot90(np.flipud(mcts_porb.reshape(self.board_height, self.board_width)), i)
                e_data.append((e_state, np.flipud(e_mcts_p).flatten(), winner))
                
                e_state = np.array([np.fliplr(s) for s in e_state])
                e_mcts_p = np.fliplr(e_mcts_p)
                e_data.append((e_state, np.flipud(e_mcts_p).flatten(), winner))
        return e_data
                
    def t_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            
            play_data = self.rotate_flip(play_data) 
            self.episode_len = len(play_data) / 8
            self.data_buffer.extend(play_data)
                        
    def nn_update(self, verbose=False):
        """update the policy-value net"""
        small_b = random.sample(self.data_buffer, self.batch_size)
        b_state = [data[0] for data in small_b]
        b_mcts_p = [data[1] for data in small_b]
        winner_batch = [data[2] for data in small_b]            
        
        old_probs, old_v = self.pv_net.u_p_value(b_state)
        
        l_list = []
        e_list = []
        for i in range(self.epochs): 
            loss, entropy = self.pv_net.train_step(b_state, 
                                             b_mcts_p, 
                                             winner_batch,
                                             self.learn_rate*self.lr_multiplier)
            
            l_list.append(loss)
            e_list.append(entropy)
            
            new_probs, new_v = self.pv_net.u_p_value(b_state)
            kbc = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kbc > self.kbc_targ * 4:  
                break
        
        if kbc > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kbc < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
            
        if verbose:
            explained_var_old = (1 -
                                 np.var(np.array(winner_batch) - old_v.flatten()) /
                                 np.var(np.array(winner_batch)))
            explained_var_new = (1 -
                                 np.var(np.array(winner_batch) - new_v.flatten()) /
                                 np.var(np.array(winner_batch)))
            
            print(("kbc: {:.3f}, "
                   "lr_multiplier: {:.3f}\n"
                   "loss: {:.3f}, "
                   "entropy: {:.3f}\n"
                   "explained old: {:.3f}, "
                   "explained new: {:.3f}\n"
                   ).format(kbc,
                            self.lr_multiplier,
                            np.mean(l_list),
                            np.mean(e_list),
                            explained_var_old,
                            explained_var_new))        

        
   
    
    def run(self):
        
        try:
            for i in range(self.game_batch_num):  
                self.t_data(self.play_batch_size)
             
                if len(self.data_buffer) > self.batch_size:
                    print("#### batch i:{}, episode_len:{} ####\n".format(i+1, self.episode_len))
                    for i in range(5):
                        verbose = i % 5 == 0
                        self.nn_update(verbose)                    
                
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    self.pv_net.saver.save(self.pv_net.sess, self.pv_net.model_file)
                    
                    print('*****win ration: {:.2f}%\n'.format(win_ratio*100))
                    
                    if win_ratio > self.best_win_ratio: 
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        self.pv_net.saver.save(self.pv_net.sess, self.pv_net.model_file) 
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 100
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            self.pv_net.saver.save(self.pv_net.sess, self.pv_net.model_file)
            print('\n\rquit')
    
if __name__ == '__main__':
    
    training_net = train_net()
    training_net.run()    
