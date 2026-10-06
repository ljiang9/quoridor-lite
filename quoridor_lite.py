"""Quoridor-lite: 9x9 迷宫棋，纯标准库实现。"""
import argparse
import copy
import random
import sys
from collections import deque

N = 9            # 棋盘 9x9
WALLS_EACH = 10  # 每人 10 面墙
GOAL = {0: 0, 1: N - 1}          # 0 号玩家目标行 0，1 号目标行 8
START = {0: (N - 1, 4), 1: (0, 4)}


def _neighbors(r, c, walls):
    """无视棋子，只看墙，返回 (r,c) 的可达邻格。"""
    out = []
    if r > 0 and ('h', r - 1, c) not in walls and ('h', r - 1, c - 1) not in walls:
        out.append((r - 1, c))
    if r < N - 1 and ('h', r, c) not in walls and ('h', r, c - 1) not in walls:
        out.append((r + 1, c))
    if c > 0 and ('v', r, c - 1) not in walls and ('v', r - 1, c - 1) not in walls:
        out.append((r, c - 1))
    if c < N - 1 and ('v', r, c) not in walls and ('v', r - 1, c) not in walls:
        out.append((r, c + 1))
    return out


def bfs_path_len(start, goal_row, walls):
    """到目标行的最短步数；不可达返回 None。"""
    if start[0] == goal_row:
        return 0
    seen = {start}
    q = deque([(start, 0)])
    while q:
        (r, c), d = q.popleft()
        for nb in _neighbors(r, c, walls):
            if nb in seen:
                continue
            if nb[0] == goal_row:
                return d + 1
            seen.add(nb)
            q.append((nb, d + 1))
    return None


class Quoridor:
    def __init__(self):
        self.pawns = {0: START[0], 1: START[1]}
        self.walls = set()          # ('h'/'v', r, c)，r,c ∈ 0..7
        self.walls_left = {0: WALLS_EACH, 1: WALLS_EACH}
        self.turn = 0
        self.winner = None
        self.half_moves = 0

    # ---- 走子 ----
    def pawn_moves(self, player):
        """合法走子目标格（含跳子）。"""
        r, c = self.pawns[player]
        foe = self.pawns[1 - player]
        moves = []
        for nb in _neighbors(r, c, self.walls):
            if nb != foe:
                moves.append(nb)
                continue
            # 相邻是对方棋子：尝试直线跳
            dr, dc = nb[0] - r, nb[1] - c
            land = (nb[0] + dr, nb[1] + dc)
            if (0 <= land[0] < N and 0 <= land[1] < N
                    and land in _neighbors(nb[0], nb[1], self.walls)):
                moves.append(land)
            else:
                # 直线被挡：落到对方棋子远端的两个斜格（官方规则）；
                # 斜对角之间没有墙的概念，只需在棋盘内
                for s in (-1, 1):
                    diag = (nb[0] + dr + (-dc) * s, nb[1] + dc + dr * s)
                    if 0 <= diag[0] < N and 0 <= diag[1] < N:
                        moves.append(diag)
        # 去重保序
        seen, uniq = set(), []
        for m in moves:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return uniq

    def apply_pawn(self, player, dest):
        if player != self.turn:
            raise ValueError("还没轮到你")
        if dest not in self.pawn_moves(player):
            raise ValueError(f"非法走子: {dest}")
        self.pawns[player] = dest
        self.half_moves += 1
        if dest[0] == GOAL[player]:
            self.winner = player
        else:
            self.turn = 1 - player
        return self.winner

    # ---- 放墙 ----
    @staticmethod
    def _wall_conflicts(wall, walls):
        o, r, c = wall
        for w in walls:
            wo, wr, wc = w
            if o == 'h':
                if wo == 'h' and wr == r and abs(wc - c) <= 1:
                    return True
                if wo == 'v' and wr == r and wc == c:
                    return True
            else:
                if wo == 'v' and wc == c and abs(wr - r) <= 1:
                    return True
                if wo == 'h' and wr == r and wc == c:
                    return True
        return False

    def wall_moves(self, player):
        if self.walls_left[player] <= 0:
            return []
        moves = []
        for o in ('h', 'v'):
            for r in range(N - 1):
                for c in range(N - 1):
                    w = (o, r, c)
                    if self._wall_conflicts(w, self.walls):
                        continue
                    trial = self.walls | {w}
                    if (bfs_path_len(self.pawns[0], GOAL[0], trial) is None
                            or bfs_path_len(self.pawns[1], GOAL[1], trial) is None):
                        continue  # 不能把路堵死
                    moves.append(w)
        return moves

    def apply_wall(self, player, wall):
        if player != self.turn:
            raise ValueError("还没轮到你")
        if self.walls_left[player] <= 0:
            raise ValueError("墙用完了")
        o, r, c = wall
        if o not in ('h', 'v') or not (0 <= r < N - 1 and 0 <= c < N - 1):
            raise ValueError(f"非法墙位置: {wall}")
        if self._wall_conflicts(wall, self.walls):
            raise ValueError(f"墙重叠/交叉: {wall}")
        trial = self.walls | {wall}
        if (bfs_path_len(self.pawns[0], GOAL[0], trial) is None
                or bfs_path_len(self.pawns[1], GOAL[1], trial) is None):
            raise ValueError("这面墙会堵死某条路")
        self.walls.add(wall)
        self.walls_left[player] -= 1
        self.half_moves += 1
        self.turn = 1 - player

    def clone(self):
        return copy.deepcopy(self)


def ai_choose(game, rng):
    """1 步贪心：最大化 (对手最短路 - 自己最短路)。"""
    me, foe = game.turn, 1 - game.turn
    base = game.clone()
    best, best_val = None, None
    cands = [('m', d) for d in game.pawn_moves(me)]
    if game.walls_left[me] > 0:
        cands += [('w', w) for w in rng.sample(game.wall_moves(me),
                                              min(24, len(game.wall_moves(me))))]
    rng.shuffle(cands)
    for kind, arg in cands:
        g = game.clone()
        if kind == 'm':
            g.apply_pawn(me, arg)
            if g.winner == me:
                return ('m', arg)  # 能赢直接走
        else:
            g.apply_wall(me, arg)
        mine = bfs_path_len(g.pawns[me], GOAL[me], g.walls)
        theirs = bfs_path_len(g.pawns[foe], GOAL[foe], g.walls)
        # 跳进墙围死胡同是合法走法（只是很蠢）：None 按极大路长处理
        mine = 999 if mine is None else mine
        theirs = 999 if theirs is None else theirs
        val = (theirs - mine) + rng.random() * 0.01
        if best_val is None or val > best_val:
            best, best_val = (kind, arg), val
    return best


def play_auto(seed=42, max_half=400):
    rng = random.Random(seed)
    g = Quoridor()
    while g.winner is None and g.half_moves < max_half:
        kind, arg = ai_choose(g, rng)
        if kind == 'm':
            g.apply_pawn(g.turn, arg)
        else:
            g.apply_wall(g.turn, arg)
    return g


def render(g):
    cell = [['·' for _ in range(N)] for _ in range(N)]
    cell[g.pawns[0][0]][g.pawns[0][1]] = '甲'
    cell[g.pawns[1][0]][g.pawns[1][1]] = '乙'
    lines = ['  ' + ' '.join(str(c) for c in range(N))]
    for r in range(N):
        lines.append(f"{r} " + ' '.join(cell[r]))
    lines.append(f"轮到: {'甲' if g.turn == 0 else '乙'}  "
                 f"墙: 甲{g.walls_left[0]} 乙{g.walls_left[1]}")
    if g.walls:
        ws = sorted(g.walls)
        lines.append("墙: " + ' '.join(f"{o}{r},{c}" for o, r, c in ws))
    return '\n'.join(lines)


def play_interactive():
    if not sys.stdin.isatty():
        print("交互模式需要终端；无头演示请用 --auto", file=sys.stderr)
        sys.exit(2)
    g = Quoridor()
    print("Quoridor-lite：甲从下往上(目标行0)，乙从上往下(目标行8)")
    print("命令: m 行 列 走子 | w h/v 行 列 放墙 | q 退出")
    while g.winner is None:
        print(render(g))
        try:
            parts = input("> ").split()
        except EOFError:
            break
        if not parts or parts[0] == 'q':
            break
        try:
            if parts[0] == 'm':
                g.apply_pawn(g.turn, (int(parts[1]), int(parts[2])))
            elif parts[0] == 'w':
                g.apply_wall(g.turn, (parts[1], int(parts[2]), int(parts[3])))
            else:
                print("未知命令")
        except (ValueError, IndexError) as e:
            print("走法非法:", e)
    print(render(g))
    if g.winner is not None:
        print(f"{'甲' if g.winner == 0 else '乙'} 胜！")


def main():
    ap = argparse.ArgumentParser(description="Quoridor-lite 迷宫棋")
    ap.add_argument('--auto', action='store_true', help='AI 对 AI 自动演示')
    ap.add_argument('--games', type=int, default=10, help='自动演示局数')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--verbose', action='store_true', help='打印每局棋盘')
    args = ap.parse_args()
    if args.auto:
        w0 = w1 = draws = 0
        for i in range(args.games):
            g = play_auto(seed=args.seed + i)
            if g.winner == 0:
                w0 += 1
            elif g.winner == 1:
                w1 += 1
            else:
                draws += 1
            if args.verbose:
                print(f"--- 第 {i+1} 局 ---")
                print(render(g))
        print(f"共 {args.games} 局：甲胜 {w0}，乙胜 {w1}，和棋 {draws}")
    else:
        play_interactive()


if __name__ == '__main__':
    main()
