import heapq
from collections import deque
import time

GOAL = ((1,2,3),(4,5,6),(7,8,0))

def manhattan(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_x = (val - 1) // 3
                goal_y = (val - 1) % 3
                distance += abs(goal_x - i) + abs(goal_y - j)
    return distance

def misplaced(state):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != 0 and state[i][j] != GOAL[i][j]:
                count += 1
    return count

def neighbors(state):
    # Find zero
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                x, y = i, j
    moves = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new = [list(row) for row in state]
            new[x][y], new[nx][ny] = new[nx][ny], new[x][y]
            moves.append(tuple(tuple(row) for row in new))
    return moves

def bfs(start):
    queue = deque([(start, [])])
    visited = set()
    nodes = 0
    while queue:
        state, path = queue.popleft()
        nodes += 1
        if state == GOAL:
            return path, nodes
        if state in visited:
            continue
        visited.add(state)
        for n in neighbors(state):
            if n not in visited:
                queue.append((n, path + [n]))
    return None, nodes

def dfs(start, limit=50):
    stack = [(start, [])]
    visited = set()
    nodes = 0
    while stack:
        state, path = stack.pop()
        nodes += 1
        if state == GOAL:
            return path, nodes
        if state in visited or len(path) > limit:
            continue
        visited.add(state)
        for n in neighbors(state):
            if n not in visited:
                stack.append((n, path + [n]))
    return None, nodes

def astar(start, heuristic):
    heap = []
    heapq.heappush(heap, (heuristic(start), 0, start, []))
    visited = set()
    nodes = 0
    while heap:
        f, g, state, path = heapq.heappop(heap)
        nodes += 1
        if state == GOAL:
            return path, nodes
        if state in visited:
            continue
        visited.add(state)
        for n in neighbors(state):
            if n not in visited:
                heapq.heappush(heap, (g+1+heuristic(n), g+1, n, path + [n]))
    return None, nodes

if __name__ == "__main__":
    start = ((1,2,3),(4,0,6),(7,5,8))
    print("BFS:")
    t0 = time.time()
    path, nodes = bfs(start)
    print(f"Solução em {len(path)} movimentos, {nodes} nós visitados, tempo: {1000*(time.time()-t0):.2f} ms")
    print("DFS:")
    t0 = time.time()
    path, nodes = dfs(start)
    print(f"Solução em {len(path)} movimentos, {nodes} nós visitados, tempo: {1000*(time.time()-t0):.2f} ms")
    print("A* (Misplaced):")
    t0 = time.time()
    path, nodes = astar(start, misplaced)
    print(f"Solução em {len(path)} movimentos, {nodes} nós visitados, tempo: {1000*(time.time()-t0):.2f} ms")
    print("A* (Manhattan):")
    t0 = time.time()
    path, nodes = astar(start, manhattan)
    print(f"Solução em {len(path)} movimentos, {nodes} nós visitados, tempo: {1000*(time.time()-t0):.2f} ms")
