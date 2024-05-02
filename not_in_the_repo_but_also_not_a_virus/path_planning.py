import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as scipy
import itertools

from matplotlib.image import imread
from astar import AStar

# == PATH PLANNING ==
class PathPlanning(AStar):
    def __init__(self, map):
        self.map = map

    def neighbors(self, node):
        # Node is the pixel location
        y, x = node
        for (dx, dy) in itertools.product([-1, 0, 1], repeat=2):
            if dx == 0 and dy == 0:
                continue
            if not (0 <= x + dx < self.map.shape[1]):
                continue
            if not (0 <= y + dy < self.map.shape[0]):
                continue
            if not self.map[y + dy][x + dx]:
                continue
            yield (y + dy, x + dx)
    
    def distance_between(self, n1, n2):
        return ((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2) ** 0.5
    
    def heuristic_cost_estimate(self, current, goal):
        return self.distance_between(current, goal)
    
def winding(path):
    x = path[:, 1]
    y = path[:, 0]

    return sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) > 0


# == IMAGE PROCESSING ==
stata_basement = imread("stata_basement.png")
stata_basement_lane0 = imread("stata_basement_lane0.png")
stata_basement_lane1 = imread("stata_basement_lane1.png")

def process(img):
    img = np.average(img, axis=2)                   # Convert to grey-scale
    img = np.greater(img, 0.9)                      # White pixels represent holes in occupancy grid
    img = scipy.binary_erosion(img, iterations=5)   # Erosion adds a safe distance from walls, removes bumps

    return img

map0 = process(stata_basement_lane0)
map1 = process(stata_basement_lane1)

# == USER INTERACTION ==
astar0 = PathPlanning(map0)
astar1 = PathPlanning(map1)
start = None

fig, ax = plt.subplots()

def onclick(e):
    global start, path, viz

    x, y = round(e.xdata), round(e.ydata)

    # Right-click sets the starting position
    if e.button == 3:
        start = (y, x)
        return
    if start is None:
        print("Select a starting position with a right-click first!")
        return

    # Else set the goal position and immediately path find. Try both lanes and take the shortest one.
    # Note: these can be run concurrently on multiple threads
    found0 = astar0.astar(start, (y, x))
    found1 = astar1.astar(start, (y, x))
    if not found0 and found1:
        found = np.array(list(found1))
    elif not found1 and found0:
        found = np.array(list(found0))
    elif found1 and found0:
        found = np.array(list(found0 if len(found0) < len(found1) else found1))
    else:
        print("No path found!")
        return
    
    print(winding(found))
    
    # Overlay the path on the plot
    ax.lines.clear()
    ax.plot(found[:, 1], found[:, 0])
    fig.canvas.draw()
    fig.canvas.flush_events()

ax.imshow(stata_basement, cmap="gray")
ax.imshow(np.logical_not(map0).astype(float), cmap="Blues", alpha=0.3)
ax.imshow(np.logical_not(map1).astype(float), cmap="Reds_r", alpha=0.3)

fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()