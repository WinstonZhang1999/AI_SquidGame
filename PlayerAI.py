import random

import numpy as np

from BaseAI import BaseAI
from Utils import manhattan_distance
from Grid import Grid
import math
import time

global maxDepth
'''
maxDepth = 7
border_weight = 100

# 0.01, 0.75
allocated_time = 2.4
weight_fr = 0.75
#global cont_trap_weight
next_step_weight = 0.5
min_traps_for_good_traps_to_activate = 7
min_moves_for_good_move_eval = 2
really_bad_position_score = -100
next_step_weight_move = 0.5
max_points_secondround = 1000

max_points = 10000
'''
maxDepth = 7
border_weight = 100
available_cell_weight = 4

# 0.01, 0.75
allocated_time = 2.4
weight_fr = 0.75
# global cont_trap_weight
next_step_weight = 0.5
min_traps_for_good_traps_to_activate = 8
min_moves_for_good_move_eval = 2
really_bad_position_score = -100
next_step_weight_move = 0.5
max_points_secondround = 1000

max_points = 10000


# TODO:
# add trap selection (only 3 left), done
# add move one step lookahead
# minimize move add selection as well




class PlayerAI(BaseAI):

    startTime = 0
    allocated_time = 0

    def __init__(self) -> None:
        super().__init__()
        self.N = 7
        self.pos = None
        self.player_num = None
        self.cont_trap_weight = 1

    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def outOfTime(self):
        #print("time:", time.time() - self.startTime)
        if time.time() - self.startTime > self.allocated_time:
            print("out of time", time.time() - self.startTime)
            self.inTime = False
            return True
        return False

    def getMove(self, grid: Grid) -> tuple:
        self.N = grid.dim
        depth = 0
        current_board = grid.clone()
        bestMove = self.MaximizeMove(current_board, depth, -2, 2, (self.player_num))
        #print(bestMove)

        self.startTime = time.time()
        self.inTime = True
        self.allocated_time = allocated_time
        # print("Best moves", bestMove)
        while self.inTime:
           if bestMove is None or bestMove[0] is None:
               # bestMove[0] = grid.get_neighbors(grid.find(self.player_num), only_available=True)[0]
               # print("none bestMove", bestMove[0])
               return random.choice(grid.get_neighbors(grid.find(self.player_num), only_available=True))
           else:
               return bestMove[0]
               # return random.choice(grid.get_neighbors(grid.find(self.player_num), only_available=True))

    def getTrap(self, grid: Grid) -> tuple:
        """
       YOUR CODE GOES HERE
       The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
       It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions,
       taking into account the probabilities of it landing in the positions you want.
       Note that you are not required to account for the probabilities of it landing in a different cell.
       You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
       """
        self.cont_trap_weight = (len(grid.getAvailableCells()) / (64))* weight_fr
        #self.cont_trap_weight = ((len(grid.getAvailableCells())*len(grid.getAvailableCells())) / (49*49)) * weight_fr
        depth = 0
        current_board = grid.clone()
        bestTrap = self.maximizeTrap(current_board, depth, -math.inf, math.inf, None)
        #print("bestTrap", bestTrap)

        self.startTime = time.time()
        self.inTime = True
        self.allocated_time = allocated_time
        # print("Best moves", bestMove)
        while self.inTime:
            if bestTrap is None or bestTrap[0] is None:
                # print("none bestTrap", bestTrap)
                return random.choice(grid.getAvailableCells())
            else:
               '''
               opponent = grid.find(3 - self.player_num)
               potential_traps = grid.get_neighbors(opponent, only_available=True)
               best_trap = random.choice(potential_traps)
               return best_trap
               '''
               # print("best trap:", bestTrap[0])
               return bestTrap[0]

    def EVAL(self, state, player_num):
        available_moves = state.get_neighbors(state.find(player_num), only_available=True) * available_cell_weight
        available_moves_len = len(available_moves)
        score = 0
        score += available_moves_len / 8
        score += (-border_weight) if (self.is_border(state.find(player_num)) == -1) else (border_weight)

        for child in available_moves:
            temp_state = state.clone()
            temp_state.move(child, player_num)
            score += ((self.EVAL_SECONDTIME_FR(temp_state, player_num) * next_step_weight_move))/len(available_moves)
            

        if available_moves_len <= min_moves_for_good_move_eval:
            return really_bad_position_score
        else:
            return score

    def EVAL_SECONDTIME_FR(self, state, player_num):
        available_moves = state.get_neighbors(state.find(player_num), only_available=True) * available_cell_weight
        available_moves_len = len(available_moves)
        score = 0
        score += available_moves_len / 8
        score += (-border_weight) if (self.is_border(state.find(player_num)) == -1) else (border_weight)

        if available_moves_len <= min_moves_for_good_move_eval:
            return really_bad_position_score
        else:
            return score

    # def exp_value(self, state, player_num):
    #
    #     v = 0
    #     for child in state.get_neighbors(state.find(player_num)):
    #         p = self.probability(child)
    #         v += p * value(child)
    #     return v
    def chance(self, state, a, b, depth, intended_position):
        if maxDepth == depth:
            return self.EVAL2(state)
        asdf = self.minimizeTrap(state, depth + 1, a, b)
        # print("--- inside chance ----")
        # print("asdf utility = ", asdf)
        # print(" returning = ", (asdf[0], (1 - 0.05 * manhattan_distance(self.getPosition(), intended_position) - 1) * float(asdf[1])))
        return (asdf[0], (1 - 0.05 * manhattan_distance(self.getPosition(), intended_position) - 1) * float(asdf[1]))

    def is_border(self, pos):
        # left border 0, bottom border = 1, right border = 2, top border = 3
        # topleft = 4, topright = 5, bottomleft = 6, bottomright = 7
        border = self.N - 1
        if pos[0] == 0 and pos[1] == 0:
            return 4
        elif pos[0] == 0 and pos[1] == border:
            return 5
        elif pos[0] == border and pos[1] == 0:
            return 6
        elif pos[0] == border and pos[1] == border:
            return 7
        elif pos[1] == 0:
            return 0
        elif pos[0] == border:
            return 1
        elif pos[1] == border:
            return 2
        elif pos[0] == 0:
            return 3
        else:
            return -1

    def find_cleanest_path(self, state, pos):

        possible_moves = []
        top = self.get_path(state, pos, self.look_top)
        #print(" top is ", top)
        if top[1] != None: possible_moves.append(top)
        top_right = self.get_path(state, pos, self.look_top_right)
        #print("top right is ", top_right)
        if top_right[1] != None: possible_moves.append(top_right)
        right = self.get_path(state, pos, self.look_right)
        #print("right is ", right)
        if right[1] != None: possible_moves.append(right)
        bottom_right = self.get_path(state, pos, self.look_bottom_right)
        #print("bottm right is ", bottom_right)
        if bottom_right[1] != None: possible_moves.append(bottom_right)
        bottom = self.get_path(state, pos, self.look_bottom)
        #print("bottom is ", bottom)
        if bottom[1] != None: possible_moves.append(bottom)
        bottom_left = self.get_path(state, pos, self.look_bottom_left)
        #print("bottom left ", bottom_left)
        if bottom_left[1] != None: possible_moves.append(bottom_left)
        left = self.get_path(state, pos, self.look_left)
        #print("left is ", left)
        if left[1] != None: possible_moves.append(left)
        top_left = self.get_path(state, pos, self.look_top_left)
        #print("top left is ", top_left)
        if top_left[1] != None: possible_moves.append(top_left)

        modified_possible_moves = [list(move[1]) for move in possible_moves]
        move_scores = [move[0] for move in possible_moves]

        for i, move in enumerate(modified_possible_moves):
            neighbors = len(state.get_neighbors(tuple(move), only_available=True))
            move_scores[i] += (neighbors-1) # -1 to not include itself
            modified_possible_moves[i] = (move_scores[i], tuple(move))
        #print(" ????? ", modified_possible_moves)
        max_distance = 0
        next_move = None
        equal_moves = []
        for move in modified_possible_moves:
            if move[0] > max_distance:
                max_distance = move[0]
                next_move = move[1]

        modified_possible_moves.sort(key=lambda a: a[0], reverse=True)
        #print("sorted ???? ", modified_possible_moves)
        modified_possible_moves = x = [i[1] for i in modified_possible_moves]
        #print("---- moves with scores =  ", modified_possible_moves)
        # print("next move = ", next_move, " max_distance = ", max_distance)
        return  modified_possible_moves[:3]  # next_move, max_distance

    def get_path(self, state, pos, path):
        distance = 0
        suggested_move = path(pos)
        while True:
            next = path(pos)
            if next == None:
                break
            else:
                value = state.getCellValue(next)
                if value == 0:
                    distance += 1
                    pos = next
                else:
                    break
        return distance, suggested_move

    def look_top(self, pos):
        next = (pos[0] - 1, pos[1])
        if next[0] < 0:
            return None
        return next

    def look_top_right(self, pos):
        next = (pos[0] - 1, pos[1] + 1)
        if next[0] < 0 or next[1] > 6:
            return None
        return next

    def look_right(self, pos):
        next = (pos[0], pos[1] + 1)
        if next[1] > 6:
            return None
        return next

    def look_bottom_right(self, pos):
        next = (pos[0] + 1, pos[1] + 1)
        if next[0] > 6 or next[1] > 6:
            return None
        return next

    def look_bottom(self, pos):
        next = (pos[0] + 1, pos[1])
        if next[0] > 6:
            return None
        return next

    def look_bottom_left(self, pos):
        next = (pos[0] + 1, pos[1] - 1)
        if next[0] > 6 or next[1] < 0:
            return None
        return next

    def look_left(self, pos):
        next = (pos[0], pos[1] - 1)
        if next[1] < 0:
            return None
        return next

    def look_top_left(self, pos):
        next = (pos[0] - 1, pos[1] - 1)
        if next[0] < 0 or next[1] < 0:
            return None
        return next

    def next_move(self, state, pos):
        return self.find_cleanest_path(state, pos)

    def MaximizeMove(self, state, depth, alpha, beta, player_num):
        # print("entering maximize -> grid  ----------------")
        # print(state.print_grid())
        maxChild, maxUtility = None, -math.inf

        opp_moves = len(state.get_neighbors(state.find(3 - player_num), only_available=True))
        player_moves = len(state.get_neighbors(state.find(player_num), only_available=True))

        if depth == maxDepth or opp_moves == 0 or player_moves == 0 or self.outOfTime():
            return None, self.EVAL(state, player_num)

        player = state.find(player_num)
        # print('player loc in maximize:', player)
        for child in state.get_neighbors(player, only_available=True):

            # print("current child = ", child)
            # print("next move options = ", self.next_move(state, player))
            if child in self.next_move(state, player):
                # print("THIS CHILD YES")
                state.map[player] = 0
                state.map[child] = player_num
                _, utility = self.MinimizeMove(state, depth + 1, alpha, beta, player_num)
                # print("utility = ", utility)
                # print("max utility =  ", maxUtility)
                # print("grid after self.minimize returns")
                # print(state.print_grid())

                if utility >= maxUtility:
                    maxChild, maxUtility = child, utility
                # state.map[child] = 0
                # print("new max utility = ", maxUtility)
                alpha = max(alpha, maxUtility)
                # print("---- alpha", alpha)
                if alpha >= beta:
                    # print("break ----------------------------------------")
                    break
        # print("max utility = ", maxUtility)
        # print("-------max child = ", maxChild)
        return maxChild, maxUtility

    def MinimizeMove(self, state, depth, alpha, beta, player_num):
        
        # print("entering minimize -> grid  ----------------")
        # print(state.print_grid())

        minChild, minUtility = None, math.inf
        opp_moves = len(state.get_neighbors(state.find(3 - player_num), only_available=True))
        player_moves = len(state.get_neighbors(state.find(player_num), only_available=True))
        if depth == maxDepth or opp_moves == 0 or player_moves == 0 or self.outOfTime():
            return None, self.EVAL(state, player_num)

        for child in state.getAvailableCells():
            # new_grid = state.trap(child)
            if state.map[child] == player_num:
                continue
            if manhattan_distance(state.find(player_num), child) <= 5:
                state.map[child] = -1
                _, utility = self.MaximizeMove(state, depth + 1, alpha, beta, player_num)
                # print("utility = ", utility)
                # print("min utility =  ", minUtility)
                # print("grid after self.maximize returns")
                # print(state.print_grid())

                if utility < minUtility:
                    minChild, minUtility = child, utility

                beta = min(beta, utility)
                if alpha >= beta:
                    break
            # print("min utility = ", minUtility)
            # print("--------min child = ", minChild)
        return minChild, minUtility

    def EVAL2(self, state, player_num, trap):
        score = 0
        opponent = state.find(3 - player_num)
        available_immediate = len(state.get_neighbors(opponent, only_available=True))

        # opponent has no more moves, game won
        if (available_immediate == 0):
            return max_points

        score += (-available_immediate)
        score += (-self.cont_trap(state, player_num) * self.cont_trap_weight)

        # one step look ahead
        for child in state.get_neighbors(opponent, only_available=True):
            new_grid = state.clone()
            movement = new_grid.trap(child)
            score += (self.EVAL2_SECONDTIME_FR(movement, player_num) * next_step_weight)
        # print("eval 2 score = ", score)
        
        score = (1 - 0.05 * (manhattan_distance(state.find(player_num), trap) - 1))*score
        return score

    def EVAL2_SECONDTIME_FR(self, state, player_num):
        score = 0
        opponent = state.find(3 - player_num)
        available_immediate = len(state.get_neighbors(opponent, only_available=True))

        # opponent has no more moves, game won
        if (available_immediate == 0):
            return max_points_secondround

        score += (-available_immediate)
        score += (-self.cont_trap(state, player_num) * self.cont_trap_weight)

        return score

    # check to see if traps are in a line
    def cont_trap(self, state, player_num):
        map = state.getMap()
        traps = []
        for i in range(len(map)):
            for j in range(len(map[i])):
                # add all trap positions
                if map[i][j] == -1:
                    traps.append((i, j))

        # finding longest continuous trap wall
        maxLen = 0
        for trap in traps:
            temp = self.longest_trap_len(trap, map, 0, [])
            if temp > maxLen:
                maxLen = temp
        return maxLen

    def longest_trap_len(self, trap, map, length, seen):
        # print(length, trap)
        # left border 0, bottom border = 1, right border = 2, top border = 3
        # topleft = 4, topright = 5, bottomleft = 6, bottomright = 7
        left_len = 0
        right_len = 0
        down_len = 0
        up_len = 0
        lengths = []
        if trap in seen:
            return length
        # border cases (edge cases)
        if self.is_border(trap) != -1:
            border = self.is_border(trap)
            # top left
            if border == 4:
                if not self.check_trap_down(trap, map) and not self.check_trap_right(trap, map):
                    return length
                else:
                    if self.check_trap_down(trap, map):
                        seen.append((trap[0] + 1, trap[1]))
                        down_len = self.longest_trap_len((trap[0] + 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_right(trap, map):
                        seen.append((trap[0], trap[1] + 1))
                        right_len = self.longest_trap_len((trap[0], trap[1] + 1), map, length + 1, seen)
            # top right
            elif border == 5:
                if not self.check_trap_down(trap, map) and not self.check_trap_left(trap, map):
                    return length
                else:
                    if self.check_trap_down(trap, map):
                        seen.append((trap[0] + 1, trap[1]))
                        down_len = self.longest_trap_len((trap[0] + 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_left(trap, map):
                        seen.append((trap[0], trap[1] - 1))
                        left_len = self.longest_trap_len((trap[0], trap[1] - 1), map, length + 1, seen)
            # bottom left
            elif border == 6:
                if not self.check_trap_up(trap, map) and not self.check_trap_right(trap, map):
                    return length
                else:
                    if self.check_trap_up(trap, map):
                        seen.append((trap[0] - 1, trap[1]))
                        up_len = self.longest_trap_len((trap[0] - 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_right(trap, map):
                        seen.append((trap[0], trap[1] + 1))
                        right_len = self.longest_trap_len((trap[0], trap[1] + 1), map, length + 1, seen)

            # bottom right
            elif border == 7:
                if not self.check_trap_up(trap, map) and not self.check_trap_left(trap, map):
                    return length
                else:
                    if self.check_trap_up(trap, map):
                        seen.append((trap[0] - 1, trap[1]))
                        up_len = self.longest_trap_len((trap[0] - 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_left(trap, map):
                        seen.append((trap[0], trap[1] - 1))
                        left_len = self.longest_trap_len((trap[0], trap[1] - 1), map, length + 1, seen)

            # left
            elif border == 0:
                if not self.check_trap_right(trap, map) and not self.check_trap_up(trap, map) \
                        and not self.check_trap_down(trap, map):
                    return length
                else:
                    if self.check_trap_down(trap, map):
                        seen.append((trap[0] + 1, trap[1]))
                        down_len = self.longest_trap_len((trap[0] + 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_up(trap, map):
                        seen.append((trap[0] - 1, trap[1]))
                        up_len = self.longest_trap_len((trap[0] - 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_right(trap, map):
                        seen.append((trap[0], trap[1] + 1))
                        right_len = self.longest_trap_len((trap[0], trap[1] + 1), map, length + 1, seen)

            # bottom
            elif border == 1:
                if not self.check_trap_up(trap, map) and not self.check_trap_left(trap, map) \
                        and not self.check_trap_right(trap, map):
                    return length
                else:
                    if self.check_trap_up(trap, map):
                        seen.append((trap[0] - 1, trap[1]))
                        up_len = self.longest_trap_len((trap[0] - 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_left(trap, map):
                        seen.append((trap[0], trap[1] - 1))
                        left_len = self.longest_trap_len((trap[0], trap[1] - 1), map, length + 1, seen)
                    if self.check_trap_right(trap, map):
                        seen.append((trap[0], trap[1] + 1))
                        right_len = self.longest_trap_len((trap[0], trap[1] + 1), map, length + 1, seen)

            # right
            elif border == 2:
                if not self.check_trap_left(trap, map) and not self.check_trap_up(trap, map) \
                        and not self.check_trap_down(trap, map):
                    return length
                else:
                    if self.check_trap_down(trap, map):
                        seen.append((trap[0] + 1, trap[1]))
                        down_len = self.longest_trap_len((trap[0] + 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_up(trap, map):
                        seen.append((trap[0] - 1, trap[1]))
                        up_len = self.longest_trap_len((trap[0] - 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_left(trap, map):
                        seen.append((trap[0], trap[1] - 1))
                        left_len = self.longest_trap_len((trap[0], trap[1] - 1), map, length + 1, seen)

            # top
            elif border == 3:
                if not self.check_trap_down(trap, map) and not self.check_trap_left(trap, map) \
                        and not self.check_trap_right(trap, map):
                    return length
                else:
                    if self.check_trap_up(trap, map):
                        seen.append((trap[0] - 1, trap[1]))
                        up_len = self.longest_trap_len((trap[0] - 1, trap[1]), map, length + 1, seen)
                    if self.check_trap_left(trap, map):
                        seen.append((trap[0], trap[1] - 1))
                        left_len = self.longest_trap_len((trap[0], trap[1] - 1), map, length + 1, seen)
                    if self.check_trap_right(trap, map):
                        seen.append((trap[0], trap[1] + 1))
                        right_len = self.longest_trap_len((trap[0], trap[1] + 1), map, length + 1, seen)

            lengths.append(up_len)
            lengths.append(down_len)
            lengths.append(right_len)
            lengths.append(left_len)
            return max(lengths)
        # no more traps around (normal case)
        elif not self.check_trap_down(trap, map) and not self.check_trap_up(trap, map) \
                and not self.check_trap_left(trap, map) and not self.check_trap_right(trap, map):
            return length
        # more traps around
        else:
            if self.check_trap_down(trap, map):
                seen.append((trap[0] + 1, trap[1]))
                down_len = self.longest_trap_len((trap[0] + 1, trap[1]), map, length + 1, seen)
            if self.check_trap_up(trap, map):
                seen.append((trap[0] - 1, trap[1]))
                up_len = self.longest_trap_len((trap[0] - 1, trap[1]), map, length + 1, seen)
            if self.check_trap_left(trap, map):
                seen.append((trap[0], trap[1] - 1))
                left_len = self.longest_trap_len((trap[0], trap[1] - 1), map, length + 1, seen)
            if self.check_trap_right(trap, map):
                seen.append((trap[0], trap[1] + 1))
                right_len = self.longest_trap_len((trap[0], trap[1] + 1), map, length + 1, seen)

            lengths.append(up_len)
            lengths.append(down_len)
            lengths.append(right_len)
            lengths.append(left_len)
            return max(lengths)

    def check_trap_left(self, trap, map):
        return map[trap[0]][trap[1] - 1] == -1

    def check_trap_right(self, trap, map):
        return map[trap[0]][trap[1] + 1] == -1

    def check_trap_down(self, trap, map):
        return map[trap[0] + 1][trap[1]] == -1

    def check_trap_up(self, trap, map):
        return map[trap[0] - 1][trap[1]] == -1

    def get_good_traps(self, state, player_num):
        traps_around_enemy = state.get_neighbors(state.find(3 - player_num), only_available=True)

        if len(traps_around_enemy) <= min_traps_for_good_traps_to_activate:
            return traps_around_enemy
        else:
            return state.getAvailableCells()

    def maximizeTrap(self, state, depth, alpha, beta, trap):
        # print("entering maximize trap ")
        # print(state.print_grid())
        # print('player_move', state.find(self.player_num))
        
        opponent = state.find(3 - self.player_num)
        neighbors = state.get_neighbors(opponent, only_available=True)
        player_moves = len(state.get_neighbors(state.find(self.player_num), only_available=True))
        if depth == maxDepth or len(neighbors) == 0 or player_moves == 0 or self.outOfTime():
            return None, self.EVAL2(state, self.getPlayerNum(), trap)

        maxChild, maxUtility = None, -8

        potential_traps = self.get_good_traps(state, self.player_num)

        for child in potential_traps:
            #if manhattan_distance(state.find(3 - self.player_num), child) <= 2:
            if state.map[child] == self.player_num:
                continue
            state.map[child] = -1
            _, utility = self.minimizeTrap(state, depth + 1, alpha, beta, child)
            # _, utility = self.chance(state, depth + 1, alpha, beta, child)
            # print("after self chance utility = ", utility)
            # print('after minimize trap')
            # print(state.print_grid())
            if utility >= maxUtility:
                maxChild, maxUtility = child, utility
            #print("max utility now = ", maxUtility)
            alpha = max(alpha, maxUtility)
            if alpha >= beta:
                break
            return maxChild, maxUtility

    def minimizeTrap(self, state, depth, alpha, beta, trap):
        # print("entering minimize trap ")
        # print(state.print_grid())
        opponent = state.find(3 - self.player_num)
        neighbors = state.get_neighbors(opponent, only_available=True)
        #neighbors = self.get_good_traps(state, 3 - self.player_num)
        player_moves = len(state.get_neighbors(state.find(self.player_num), only_available=True))
        if depth == maxDepth or len(neighbors) == 0 or player_moves == 0 or self.outOfTime():
            return None, self.EVAL2(state, self.getPlayerNum(), trap)
        
        minChild, minUtility = None, -1
        # child = random.choice(neighbors)
        # print(state.print_grid())
        child = self.MaximizeMove(state.clone(), 0, -2, 2, 3 - (self.player_num))[0]
        old_pos = opponent
        temp = state.clone()
        if child is None:
            child = random.choice(state.get_neighbors(state.find(3-self.player_num), only_available=True))

        temp.map[old_pos] = 0
        temp.map[child] = 3 - self.player_num

        (_, utility) = self.maximizeTrap(temp.clone(), depth + 1, alpha, beta, trap)
        # print("after self maximize trap utility = ", utility)
        # print(temp.print_grid())
        if utility < minUtility:
            minChild, minUtility = child, utility
        beta = min(beta, utility)
        #print("min utility now = ", minUtility)
        if alpha >= beta:
            return minChild, minUtility

        return minChild, minUtility
        