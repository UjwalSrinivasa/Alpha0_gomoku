import numpy as np
import tkinter

class Board(object):
    """
    board for the game
    """

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {} 
        self.n_in_row = int(kwargs.get('n_in_row', 5)) 
        self.players = [1, 2] 
        
    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row)
        self.current_player = self.players[start_player]   
        self.availables = list(range(self.width * self.height)) 
        self.states = {} 
        self.last_move = -1

    def mtl(self, move):
        """       
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        x = move  // self.width
        y = move  %  self.width
        return [x, y]

    def ltm(self, location):
        if(len(location) != 2):
            return -1
        x = location[0]
        y = location[1]
        move = x * self.width + y
        if(move not in range(self.width * self.height)):
            return -1
        return move

    def current_state(self): 
        """return the board state from the perspective of the current player
        shape: 4*width*height"""
        
        s_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            m_curr = moves[players == self.current_player]
            m_oppo = moves[players != self.current_player]                           
            s_state[0][m_curr // self.width, m_curr % self.height] = 1.0
            s_state[1][m_oppo // self.width, m_oppo % self.height] = 1.0   
            s_state[2][self.last_move //self.width, self.last_move % self.height] = 1.0    
        if len(self.states)%2 == 0:
            s_state[3][:,:] = 1.0

        return s_state[:,::-1,:]

    def dm(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1] 
        self.last_move = move

    def has_a_winner(self):
        wi = self.width
        he = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(wi * he)) - set(self.availables))
        if(len(moved) < self.n_in_row*2 - 1):
            return False, -1

        for m in moved:
            x = m // wi
            y = m % wi
            player = states[m]

            if (y in range(wi - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (x in range(he - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * wi, wi))) == 1):
                return True, player

            if (y in range(wi - n + 1) and x in range(he - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (wi + 1), wi + 1))) == 1):
                return True, player

            if (y in range(n - 1, wi) and x in range(he - n + 1) and
                len(set(states.get(i, -1) for i in range(m, m + n * (wi - 1), wi - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):#            
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

class Point:

    def __init__(self, x, y):
        self.x = x;
        self.y = y;
        self.pixel_x = 30 + 30 * self.x
        self.pixel_y = 30 + 30 * self.y

class Game(object):
    """
    game server
    """
    def __init__(self, board, **kwargs):
        self.board = board
    
    def click1(self, event): 

        current_player = self.board.get_current_player()
        if current_player == 1:
            i = (event.x) // 30
            j = (event.y) // 30
            ri = (event.x) % 30
            rj = (event.y) % 30
            i = i-1 if ri<15 else i
            j = j-1 if rj<15 else j
            move = self.board.ltm((i, j))
            if move in self.board.availables:
                self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='yellow')
                self.board.dm(move)

    def run(self):
        current_player = self.board.get_current_player()
        
        end, winner = self.board.game_end()
        
        if current_player == 2 and not end:
            player_in_turn = self.players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.dm(move)
            i, j = self.board.mtl(move)
            self.cv.create_oval(self.chess_board_points[i][j].pixel_x-10, self.chess_board_points[i][j].pixel_y-10, self.chess_board_points[i][j].pixel_x+10, self.chess_board_points[i][j].pixel_y+10, fill='brown')
                
        end, winner = self.board.game_end()
        
        if end:
            if winner != -1:
                self.cv.create_text(self.board.width*15+15, self.board.height*30+30, text="Game over. Winner is {}".format(self.players[winner]))
                self.cv.unbind('<Button-1>')
            else:
                self.cv.create_text(self.board.width*15+15, self.board.height*30+30, text="Game end. Tie")

            return winner
        else:
            self.cv.after(100, self.run)
        
    def graphic(self, board, player1, player2):
        """
        Draw the board and show game info
        """
        width = board.width
        height = board.height
        
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        self.players = {p1: player1, p2:player2}
        
        window = tkinter.Tk()
        self.cv = tkinter.Canvas(window, height=height*30+60, width=width*30 + 30, bg = 'green')
        self.chess_board_points = [[None for i in range(height)] for j in range(width)]
        
        for i in range(width):
            for j in range(height):
                self.chess_board_points[i][j] = Point(i, j);
        for i in range(width):  
            self.cv.create_line(self.chess_board_points[i][0].pixel_x, self.chess_board_points[i][0].pixel_y, self.chess_board_points[i][width-1].pixel_x, self.chess_board_points[i][width-1].pixel_y)
        
        for j in range(height):  
            self.cv.create_line(self.chess_board_points[0][j].pixel_x, self.chess_board_points[0][j].pixel_y, self.chess_board_points[height-1][j].pixel_x, self.chess_board_points[height-1][j].pixel_y)        
        
        self.button = tkinter.Button(window, text="start game!", command=self.run)
        self.cv.bind('<Button-1>', self.click1)
        self.cv.pack()
        self.button.pack()
        window.mainloop()
               
    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        start a game between two players
        """
        if start_player not in (0,1):
            raise Exception('start_player should be 0 (player1 first) or 1 (player2 first)')
        self.board.init_board(start_player)

        if is_shown:
            self.graphic(self.board, player1, player2)
        else:
            p1, p2 = self.board.players
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)
            players = {p1: player1, p2:player2}
            while(1):
                current_player = self.board.get_current_player()
                print(current_player)
                player_in_turn = players[current_player]
                move = player_in_turn.get_action(self.board)
                self.board.dm(move)
                if is_shown:
                    self.graphic(self.board, player1.player, player2.player)
                end, winner = self.board.game_end()
                if end:
                    return winner   

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree
        store the self-play data: (state, mcts_probs, z)
        """
        self.board.init_board()        
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []        
        while(1):
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
           
            self.board.dm(move)      
            end, winner = self.board.game_end()
            if end:
                
                winners_z = np.zeros(len(current_players))  
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                
                player.reset_player() 
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
            