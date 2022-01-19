import random
from BaseAI import BaseAI
from Grid import Grid
import math

global maxDepth
maxDepth = 3
border_weight = 0.15
cont_trap_weight = 0.01
next_step_weight = 0.05

class PlayerAI(BaseAI):
    def __init__(self) -> None:
        super().__init__()
        self.pos = None
        self.player_num = None

    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        depth = 0
        current_board = grid.clone()
        bestMove = self.MaximizeMove(current_board, depth, -2, 2)
        print("Best moves", bestMove)
        return bestMove[0]

    def getTrap(self, grid: Grid) -> tuple:
        """
       YOUR CODE GOES HERE
       The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
       It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions,
       taking into account the probabilities of it landing in the positions you want.
       Note that you are not required to account for the probabilities of it landing in a different cell.
       You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
       """

        depth = 0
        current_board = grid.clone()
        bestTrap = self.maximizeTrap(current_board, depth, -math.inf, math.inf)
        print("bestTrap", bestTrap)
        return bestTrap[0]

    def EVAL(self, state, player_num):
        score = len(state.get_neighbors(state.find(player_num),
                                        only_available=True)) / 8
        score += (-border_weight) if (self.is_border(state.find(player_num)) == -1) else (border_weight)
        return score

    def chance(self, state, a, b, depth):
        #Easiet to think of for our trap. We just want to pass this with minimize to see the actualy chance of what we want working out and factor
        # that in to our final decision. Still need to work this out completely
        if maxDepth == depth:
            return self.EVAL2(state)
        return (1 - 0.05*(self.manhattan_distance(self.player.getPosition(), self.intended_position) - 1)) * self.MinimizeMove(state, depth + 1, a, b)

    def is_border(pos): 
        #left border 0, bottom border = 1, right border = 2, top border = 3
        #topleft = 4, topright = 5, bottomleft = 6, bottomright = 7
        if pos[0] == 0 and pos[1] == 0:
            return 4
        elif pos[0] == 0 and pos[1] == 6:
            return 5
        elif pos[0] == 6 and pos[1] == 0:
            return 6
        elif pos[0] == 6 and pos[1] == 6:
            return 7
        elif pos[1] == 0:
            return 0
        elif pos[0] == 6:
            return 1
        elif pos[1] == 6:
            return 2
        elif pos[0] == 0:
            return 3
        else:
            return -1

    def MaximizeMove(self, state, depth, alpha, beta):
        global maxDepth

        maxChild, maxUtility = None, -math.inf

        if depth >= maxDepth:
            return (None, self.EVAL(state, self.getPlayerNum()))

        player = state.find(self.player_num)

        for child in state.get_neighbors(player, only_available=True):
            new_grid = state.clone()
            movement = new_grid.move(child, self.player_num)
            _, utility = self.MinimizeMove(movement, depth + 1, alpha, beta)

            if utility > maxUtility:
                maxChild, maxUtility = child, utility
            movement.map[child] = 0

            if utility > maxUtility:
                maxChild, maxUtility = child, utility

            alpha = max(alpha, maxUtility)
            if alpha >= beta:
                break

        return maxChild, maxUtility

    def MinimizeMove(self, state, depth, alpha, beta):
        global maxDepth

        minChild, minUtility = None, math.inf

        if depth >= maxDepth:
            return None, self.EVAL(state, self.getPlayerNum())

        for child in state.getAvailableCells():
            new_grid = state.trap(child)
            _, utility = self.MaximizeMove(new_grid, depth + 1, alpha, beta)

            if utility < minUtility:
                minChild, minUtility = child, utility
            new_grid.map[child] = 0

            beta = min(beta, utility)
            if beta <= alpha:
                break

        return minChild, minUtility

    def EVAL2(self, state, player_num):
        # imporved score
        # print(state.print_grid())
        # print(player_num)
        # print(state.find(player_num))
        score = 0
        opponent = state.find(3 - player_num)
        available_immediate = len(state.get_neighbors(opponent, only_available=True))
        
        #opponent has no more moves, game won
        if(available_immediate == 0):
            return math.inf
        
        score += (-available_immediate)
        score += (-self.cont_trap(state, player_num)*cont_trap_weight)
        
        #one step look ahead
        for child in state.get_neighbors(opponent, only_available=True):
            score += (self.EVAL2(child, player_num)*next_step_weight)
        
        return score


    #check to see if traps are in a line
    def cont_trap(self, state, player_num):
        map = state.getMap()
        traps = []
        for i in range(len(map)):
            for j in range(len(map[i])):
                #add all trap positions
                if map[i][j] == -1:
                    traps.append((i,j))
        
        #finding longest continuous trap wall
        maxLen = 0
        for trap in traps:
            temp = self.longest_trap_len(trap, map, 0)
            if temp > maxLen: 
                maxLen = temp
        return maxLen


    def longest_trap_len(self, trap, map, length):
        #left border 0, bottom border = 1, right border = 2, top border = 3
        #topleft = 4, topright = 5, bottomleft = 6, bottomright = 7
        left_len = 0
        right_len = 0
        down_len = 0
        up_len = 0
        lengths = []

        #border cases (edge cases)
        if self.is_border(trap) != -1:
            border = self.is_border(trap) 
            #top left
            if border == 4:
                if not self.check_trap_down(trap, map) and not self.check_trap_right(trap, map):
                    return length
                else:
                    if self.check_trap_down(trap, map):
                        down_len = self.longest_trap_len((trap[0]+1,trap[1]), map, length + 1)
                    if self.check_trap_right(trap, map):
                        right_len = self.longest_trap_len((trap[0], trap[1]+1), map, length + 1)
            #top right
            elif border == 5:
                if not self.check_trap_down(trap, map) and not self.check_trap_left(trap, map):
                    return length
                else:
                    if self.check_trap_down(trap, map):
                        down_len = self.longest_trap_len((trap[0]+1,trap[1]), map, length + 1)
                    if self.check_trap_left(trap, map):
                        left_len = self.longest_trap_len((trap[0], trap[1]-1), map, length + 1)
            #bottom left
            elif border == 6:
                if not self.check_trap_up(trap, map) and not self.check_trap_right(trap, map):
                    return length
                else:
                    if self.check_trap_up(trap, map):
                        up_len = self.longest_trap_len((trap[0]-1, trap[1]), map, length + 1)
                    if self.check_trap_right(trap, map):
                        right_len = self.longest_trap_len((trap[0], trap[1]+1), map, length + 1)

            #bottom right
            elif border == 7:
                if not self.check_trap_up(trap, map) and not self.check_trap_left(trap, map):
                    return length
                else:
                    if self.check_trap_up(trap, map):
                        up_len = self.longest_trap_len((trap[0]-1, trap[1]), map, length + 1)
                    if self.check_trap_left(trap, map):
                        left_len = self.longest_trap_len((trap[0], trap[1]-1), map, length + 1)
            
            #left
            elif border == 0:
                if not self.check_trap_right(trap, map) and not self.check_trap_up(trap, map) \
                    and not self.check_trap_down(trap, map):
                    return length
                else:
                    if self.check_trap_down(trap, map):
                        down_len = self.longest_trap_len((trap[0]+1,trap[1]), map, length + 1)
                    if self.check_trap_up(trap, map):
                        up_len = self.longest_trap_len((trap[0]-1, trap[1]), map, length + 1)
                    if self.check_trap_right(trap, map):
                        right_len = self.longest_trap_len((trap[0], trap[1]+1), map, length + 1)
                        
            #bottom
            elif border == 1:
                if not self.check_trap_up(trap, map) and not self.check_trap_left(trap, map) \
                    and not self.check_trap_right(trap, map):
                    return length
                else:
                    if self.check_trap_up(trap, map):
                        up_len = self.longest_trap_len((trap[0]-1, trap[1]), map, length + 1)
                    if self.check_trap_left(trap, map):
                        left_len = self.longest_trap_len((trap[0], trap[1]-1), map, length + 1)
                    if self.check_trap_right(trap, map):
                        right_len = self.longest_trap_len((trap[0], trap[1]+1), map, length + 1)

            #right
            elif border == 2:
                if not self.check_trap_left(trap, map) and not self.check_trap_up(trap, map) \
                    and not self.check_trap_down(trap, map):
                    return length
                else:
                    if self.check_trap_down(trap, map):
                        down_len = self.longest_trap_len((trap[0]+1,trap[1]), map, length + 1)
                    if self.check_trap_up(trap, map):
                        up_len = self.longest_trap_len((trap[0]-1, trap[1]), map, length + 1)
                    if self.check_trap_left(trap, map):
                        left_len = self.longest_trap_len((trap[0], trap[1]-1), map, length + 1)

            #top
            elif border == 3:
                if not self.check_trap_down(trap, map) and not self.check_trap_left(trap, map) \
                    and not self.check_trap_right(trap, map):
                    return length
                else:
                    if self.check_trap_up(trap, map):
                        up_len = self.longest_trap_len((trap[0]-1, trap[1]), map, length + 1)
                    if self.check_trap_left(trap, map):
                        left_len = self.longest_trap_len((trap[0], trap[1]-1), map, length + 1)
                    if self.check_trap_right(trap, map):
                        right_len = self.longest_trap_len((trap[0], trap[1]+1), map, length + 1)
            lengths.append(up_len, down_len, right_len, left_len)
            return max(lengths)
        #no more traps around (normal case)
        elif not self.check_trap_down(trap, map) and not self.check_trap_up(trap, map)\
            and not self.check_trap_left(trap, map) and not self.check_trap_right(trap, map):
            return length
        #more traps around
        else:
            if self.check_trap_down(trap, map):
                down_len = self.longest_trap_len((trap[0]+1,trap[1]), map, length + 1)
            if self.check_trap_up(trap, map):
                up_len = self.longest_trap_len((trap[0]-1, trap[1]), map, length + 1)
            if self.check_trap_left(trap, map):
                left_len = self.longest_trap_len((trap[0], trap[1]-1), map, length + 1)
            if self.check_trap_right(trap, map):
                right_len = self.longest_trap_len((trap[0], trap[1]+1), map, length + 1)
            
            lengths.append(up_len, down_len, right_len, left_len)
            return max(lengths)

    def check_trap_left(self, trap, map):
        return map[trap[0]][trap[1]-1] == -1
    def check_trap_right(self, trap, map):
        return map[trap[0][trap[1]+1]] == -1
    def check_trap_down(self, trap, map):
        return map[trap[0]+1][trap[1]] == -1
    def check_trap_up(self, trap, map):
        return map[trap[0]-1][trap[1]] == -1

    def maximizeTrap(self,  state, depth, alpha, beta):
        opponent = state.find(3 - self.player_num)
        neighbors = state.get_neighbors(opponent, only_available=True)
        #print("(in max) opponent position in grid = :", opponent, "  neighbors = ", neighbors)
        #print("depth at maximize = ", depth)
        if depth == 3 or len(neighbors) == 0:
            # print("depth = ", depth, " len(neighbors) = ", len(neighbors), " isMax = ", isMaximizer)
            # print("eval score = ", self.EVAL2(state, self.getPlayerNum()))
            return None, self.EVAL2(state, self.getPlayerNum())

        maxChild, maxUtility = None, -8  # The less neighbors the better
        for child in neighbors:
            if len(neighbors) == 1 and state.find(3 - self.player_num) != child:
                new_grid = state.trap(child)
                # print("after trapping")
                # print(new_grid.print_grid())
                _, utility = self.minimizeTrap(new_grid, depth + 1, alpha, beta)
                new_grid.map[child] = 0

                if utility > maxUtility:
                    maxChild, maxUtility = child, utility

                alpha = max(alpha, maxUtility)
                if alpha >= beta:
                    break
            else:
                # print("deep child neighbors = ", state.get_neighbors(child, only_available=True))
                # print("opp position", opp_position)
                if len(state.get_neighbors(child, only_available=True)) != 0:
                    for deep_child in state.get_neighbors(child, only_available=True):
                        # Basically we are placing a trap 2 cells away from player 2
                        #print(state.find(2))
                        #print(deep_child)
                        if state.find(3 - self.player_num) != deep_child:
                            new_grid = state.trap(deep_child)
                            # print("after trapping")
                            # print(new_grid.print_grid())
                            _, utility = self.minimizeTrap(new_grid, depth + 1, alpha, beta)
                            new_grid.map[deep_child] = 0

                    if utility > maxUtility:
                        maxChild, maxUtility = child, utility

                    alpha = max(alpha, maxUtility)
                    if alpha >= beta:
                        break
                else:
                    #game over since opponent has zero neighbors
                    break
        return maxChild, maxUtility

    def minimizeTrap(self,  state, depth, alpha, beta):
        opponent = state.find(3 - self.player_num)
        neighbors = state.get_neighbors(opponent, only_available=True)
        #print("(in min)opponent position in grid = :", opponent, "  neighbors = ", neighbors)
        #print("depth at minimize = ", depth)
        if depth == 3 or len(neighbors) == 0:
            # print("depth = ", depth, " len(neighbors) = ", len(neighbors), " isMax = ", isMaximizer)
            # print("eval score = ", self.EVAL2(state, self.getPlayerNum()))
            return None, self.EVAL2(state, self.getPlayerNum())

        minChild, minUtility = None, -1
        child = random.choice(neighbors)
        new_grid = state.move(child, 3 - self.player_num)
        # print("after moving 2")
        # print(new_grid.print_grid())
        (_, utility) = self.maximizeTrap(new_grid, depth + 1, alpha, beta)

        if utility < minUtility:
            minChild, minUtility = child, utility

        if state.find(3 - self.player_num) != child:
            new_grid.map[child] = 0
            # print("aftering setting 0")
            # print(new_grid.print_grid())

        beta = min(beta, utility)
        if alpha >= beta:
            return minChild, minUtility

        return minChild, minUtility