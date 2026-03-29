import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PySide6.QtGui import QPainter, QColor
from PySide6.QtCore import Qt, QTimer
import heapq

GRID_SIZE = 12
CELL_SIZE = 40

# ======================
# HEURISTIC (EUCLIDEAN)
# ======================
def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ======================
# A* / DIJKSTRA
# ======================
def astar(grid, start, goal, use_astar=True):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    visited = []

    directions = [
        (1,0),(-1,0),(0,1),(0,-1),
        (1,1),(1,-1),(-1,1),(-1,-1)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)
        visited.append(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], visited

        for dx, dy in directions:
            neighbor = (current[0]+dx, current[1]+dy)

            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                if grid[neighbor] == 1:
                    continue

                cost = np.sqrt(2) if dx != 0 and dy != 0 else 1
                temp_g = g_score[current] + cost

                if neighbor not in g_score or temp_g < g_score[neighbor]:
                    g_score[neighbor] = temp_g

                    h = heuristic(neighbor, goal) if use_astar else 0
                    f = temp_g + h

                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current

    return None, visited

# ======================
# MAIN WIDGET
# ======================
class Maze(QWidget):
    def __init__(self):
        super().__init__()
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.start = None
        self.goal = None

        self.path = []
        self.visited = []

        self.mode = "wall"

        self.robot_pos = None
        self.path_index = 0

        self.use_astar = True

        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)

    def mousePressEvent(self, event):
        self.handle_mouse(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.handle_mouse(event)

    def handle_mouse(self, event):
        x = event.x() // CELL_SIZE
        y = event.y() // CELL_SIZE

        if x >= GRID_SIZE or y >= GRID_SIZE:
            return

        if self.mode == "wall":
            self.grid[y, x] = 1
        elif self.mode == "start":
            self.start = (y, x)
        elif self.mode == "goal":
            self.goal = (y, x)

        self.update()

    # ======================
    # RUN
    # ======================
    def run_astar(self):
        if self.start and self.goal:
            self.path, self.visited = astar(
                self.grid, self.start, self.goal, self.use_astar
            )

            if self.path:
                self.path_index = 0
                self.robot_pos = np.array(self.path[0], dtype=float)
                self.timer.start(40)

    # ======================
    # ANIMATION
    # ======================
    def animate(self):
        if not self.path:
            return

        if self.path_index >= len(self.path)-1:
            self.timer.stop()
            return

        current = np.array(self.path[self.path_index], dtype=float)
        next_node = np.array(self.path[self.path_index+1], dtype=float)

        direction = next_node - current
        self.robot_pos += 0.2 * direction

        if np.linalg.norm(self.robot_pos - next_node) < 0.1:
            self.path_index += 1

        self.update()

    # ======================
    # DRAW (LIGHT MODE)
    # ======================
    def paintEvent(self, event):
        painter = QPainter(self)

        # background putih
        painter.fillRect(self.rect(), QColor(245, 245, 245))

        # grid
        painter.setPen(QColor(200, 200, 200))
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = j * CELL_SIZE
                y = i * CELL_SIZE

                if self.grid[i, j] == 1:
                    painter.fillRect(x, y, CELL_SIZE, CELL_SIZE, QColor(50, 50, 50))
                else:
                    painter.drawRect(x, y, CELL_SIZE, CELL_SIZE)

        # visited
        for v in self.visited:
            painter.fillRect(
                v[1]*CELL_SIZE, v[0]*CELL_SIZE,
                CELL_SIZE, CELL_SIZE,
                QColor(180, 180, 180, 120)
            )

        # path
        if self.path:
            for p in self.path:
                painter.fillRect(
                    p[1]*CELL_SIZE, p[0]*CELL_SIZE,
                    CELL_SIZE, CELL_SIZE,
                    QColor(100, 180, 255, 180)
                )

        # start
        if self.start:
            painter.fillRect(
                self.start[1]*CELL_SIZE, self.start[0]*CELL_SIZE,
                CELL_SIZE, CELL_SIZE,
                QColor(0, 200, 120)
            )

        # goal
        if self.goal:
            painter.fillRect(
                self.goal[1]*CELL_SIZE, self.goal[0]*CELL_SIZE,
                CELL_SIZE, CELL_SIZE,
                QColor(220, 60, 60)
            )

        # robot
        if self.robot_pos is not None:
            painter.fillRect(
                int(self.robot_pos[1]*CELL_SIZE),
                int(self.robot_pos[0]*CELL_SIZE),
                CELL_SIZE, CELL_SIZE,
                QColor(50, 120, 255)
            )

# ======================
# APP
# ======================
app = QApplication(sys.argv)

# 🎨 LIGHT BUTTON STYLE
app.setStyleSheet("""
QPushButton {
    background-color: #e0e0e0;
    color: black;
    border-radius: 8px;
    padding: 6px;
}
QPushButton:hover {
    background-color: #d0d0d0;
}
""")

maze = Maze()
maze.setFixedSize(GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE)

btn_wall = QPushButton("Wall")
btn_start = QPushButton("Start")
btn_goal = QPushButton("Goal")
btn_run = QPushButton("Run")
btn_algo = QPushButton("A*")

btn_wall.clicked.connect(lambda: setattr(maze, 'mode', 'wall'))
btn_start.clicked.connect(lambda: setattr(maze, 'mode', 'start'))
btn_goal.clicked.connect(lambda: setattr(maze, 'mode', 'goal'))
btn_run.clicked.connect(maze.run_astar)

# toggle A* / Dijkstra
def toggle_algo():
    maze.use_astar = not maze.use_astar
    btn_algo.setText("A*" if maze.use_astar else "Dijkstra")

btn_algo.clicked.connect(toggle_algo)

layout = QVBoxLayout()
layout.addWidget(maze)

btn_layout = QHBoxLayout()
btn_layout.addWidget(btn_wall)
btn_layout.addWidget(btn_start)
btn_layout.addWidget(btn_goal)
btn_layout.addWidget(btn_run)
btn_layout.addWidget(btn_algo)

layout.addLayout(btn_layout)

container = QWidget()
container.setLayout(layout)
container.setWindowTitle("Robot Path Planning Simulator")
container.show()

sys.exit(app.exec())
