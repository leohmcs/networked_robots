#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pyvisgraph as vg
import shapely.geometry

import colorama

# TODO: por que passo o caminho como argumento do objeto? devia ser argumento de goals()
class GoalsPlanner:
    def __init__(self, path, net_radius, obstacles, id_prefix='robot'):
        self.path = self.point_to_array(path)
        self.base_position = self.path[0]
        self.net_radius = net_radius
        
        if obstacles is None:
            polys = [[vg.Point(0.0,6.0), vg.Point(3.0,6.0), vg.Point(1.5,9.0)],
                    [vg.Point(2.0, 0.0), vg.Point(0.7071*2, 0.7071*2), vg.Point(0.0, 2.0), vg.Point(-0.7071*2, 0.7071*2), vg.Point(-2.0, 0.0), vg.Point(-0.7071*2, -0.7071*2), vg.Point(0.0, -2.0), vg.Point(0.7071*2, -0.7071*2)],	
                    [vg.Point(5.0,0.0), vg.Point(7.0,0.0), vg.Point(7.0,2.0), vg.Point(5.0,2.0)],
                    [vg.Point(6.0,4.0), vg.Point(9.0,4.0), vg.Point(9.0,7.0), vg.Point(7.5,8.0), vg.Point(5.0,4.5)],
                    [vg.Point(-3.0*1.5,3.0*1.5), vg.Point(-6.0*1.5,2.0*1.5), vg.Point(-6.0*1.5,6.0*1.5), vg.Point(-2.5*1.5,6.0*1.5), vg.Point(-2.0*1.5,1.5*1.5)],
                    [vg.Point(-8.0,-4.0), vg.Point(-5.0,-4.0), vg.Point(-6.5,0.5)],
                    [vg.Point(3.0+5.0, -5.0), vg.Point(0.7071*3+5.0, 0.7071*3-5.0), vg.Point(0.0+5.0, 3.0-5.0), vg.Point(-0.7071*3+5.0, 0.7071*3-5.0), vg.Point(-3.0+5.0, -5.0), vg.Point(-0.7071*3+5.0, -0.7071*3-5.0), vg.Point(0.0+5.0, -3.0-5.0), vg.Point(0.7071*3+5.0, -0.7071*3-5.0)],
                    [vg.Point(9.1, 0.0), vg.Point(9.0, -1.0), vg.Point(9.8, -1.0)]]
            obstacles = polys
            # self.log('Using example map since no map was provided.', 'warn')

        self.obstacles = obstacles

        self.id_prefix = id_prefix + '{}'

    def point_to_array(self, path):
        arr = np.array([path[0].x, path[0].y])
        for p in path[1:]:
            p = np.array([p.x, p.y])
            arr = np.vstack((arr, p))

        return arr

    def Rz(self, theta):
        rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        return rot

    def unit_vector(self, v):
        return v / np.linalg.norm(v)

    def ang_between(self, v, u):
        v = self.unit_vector(v)
        u = self.unit_vector(u)
        return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

    def vec_angle(self, v):
        i_hat = np.array([1, 0])
        ang = self.ang_between(v, i_hat)
        return np.sign(v[1]) * ang

    def translate(self, a, b):
        return a + b

    def goals_dict(self, goals):
        goals_dict = dict()
        for i, g in enumerate(goals):
            goals_dict[self.id_prefix.format(i)] = g

        return goals_dict

    def secant(self, points, center, radius, start=0):
        '''
        Returns the indexes of two first points in 'points' that define a line segment starting inside the circle centered 
        in 'center' of radius 'radius' and cros the circunference in any point.
        Input: points: list of points to check; center: center of circunference (x, y); radius: circunference radius; 
        start: index to start from.
        '''
        i1 = 0
        for p in points:
            if np.linalg.norm(p - center) >= radius:
                i2 = i1 + 1
                return i1, i2
            
            i1 += 1
        
        # no pair of points define a secant
        i1 = 0
        i2 = len(points)
        return i1, i2

    def goals(self):
        goals = list()
        reverse_path = self.path[::-1]   # reverse array to start from frontier position

        goals.append(reverse_path[0])   # first robot's goal is always the frontier

        ind = 1         # index of path point used in current calculations
        while np.linalg.norm(goals[-1] - self.base_position) > self.net_radius \
            or self.connection_lost(goals[-1], self.base_position):

            prev_robot_goal = goals[-1]
            d = np.linalg.norm(reverse_path[ind] - prev_robot_goal)

            if d >= self.net_radius:
                th = self.vec_angle(reverse_path[ind] - prev_robot_goal)
                goal = self.net_radius * np.array([np.cos(th), np.sin(th)]) + prev_robot_goal
                goals.append(goal)
            else:
                if ind == len(reverse_path) - 1:
                    break
                
                i1, i2 = self.secant(reverse_path[(ind + 1):], prev_robot_goal, self.net_radius)
                a = reverse_path[ind + i1] - prev_robot_goal
                b = reverse_path[ind + i2] - prev_robot_goal

                ab = b - a
                alpha = self.vec_angle(ab)
                if np.abs(alpha) > np.pi/2:
                    alpha = np.sign(alpha) * np.pi - alpha

                a_aux = np.dot(self.Rz(alpha), a)

                th = self.ang_between(a, ab)
                t_0 = (np.sqrt(self.net_radius**2 - a_aux[1]**2) - (np.sign(np.cos(th)) * np.abs(a_aux[0])))/np.linalg.norm(ab)
                goal = t_0 * ab + reverse_path[ind]

                # isn't working as expected yet. didn't try to fix because it seems it is not necessary for the tangled alg path
                # for j in reversed(range(i2)):
                #     connection_lost = self.connection_lost(prev_robot_goal, goal)
                #     if connection_lost:
                #         goal = self.reconnect(prev_robot_goal, a, b, t_0) + reverse_path[ind + j]
                #         ind += j

                connection_lost = self.connection_lost(prev_robot_goal, goal)
                if connection_lost:
                    goal = reverse_path[ind]

                goals.append(goal)
                ind += 1

        goals = self.goals_dict(goals)
        return goals

    def connection_lost(self, r1_pos, r2_pos, obstacles=None):
        '''
        Checks if the line connecting r1 and r2 intersects an obstacle
        Input: both robots positions; obs: obstacles to check
        Output: True if there is a collision, false otherwise.
        '''
        if obstacles is None:
            obstacles = self.obstacles 
        
        line = shapely.geometry.LineString([r1_pos, r2_pos])
        for obs in obstacles:
            obs = self.point_to_array(obs)
            obs = shapely.geometry.Polygon(obs)
            if obs.intersects(line):
                return True

        return False

    def reconnect(self, r1, a, b, t_0, step=0.01):
        '''
        Finds a new goal avoiding the detected collision
        '''

        ab = b - a
        r2 = t_0 * ab + a
        collision = True
        while collision:
            t_0 -= step
            if t_0 < 0:
                t_0 = 0
                break

            r2 = t_0 * ab
            collision = self.connection_lost(r1, r2)

        new_goal = t_0 * ab
        return new_goal

    def check_solution(self, goals_dict):
        goals = np.array(list(goals_dict.values()))

        d = np.linalg.norm(self.base_position - goals[-1])
        if d > self.net_radius:
            print(colorama.Fore.RED + 'This frontier is too far: {} m from last backbone robot to the base.'.format(d) + colorama.Style.RESET_ALL)
            return False


        for i in range(len(goals) - 1):
            p1 = goals[i]
            p2 = goals[i + 1]
            d = np.linalg.norm(p2 - p1)
            print(d)
            if d > self.net_radius:
                print(colorama.Fore.RED + 'Distance between {} and {} is greater than network connection radius: {} > {}'.format(p1, p2, d, self.net_radius) + colorama.Style.RESET_ALL)
                return False

        print(colorama.Fore.GREEN + 'Solution is valid.' + colorama.Style.RESET_ALL)
        return True

    def plot_solution(self, goals_dict, plot_obstacles=True, plot_distances=False):
        if plot_obstacles:
            for i in range(0, len(self.obstacles)):
                for j in range(0,len(self.obstacles[i])-1):
                    plt.plot([self.obstacles[i][j].x, self.obstacles[i][j+1].x], [self.obstacles[i][j].y, self.obstacles[i][j+1].y], 'b')
                plt.plot([self.obstacles[i][0].x, self.obstacles[i][len(self.obstacles[i])-1].x], [self.obstacles[i][0].y, self.obstacles[i][len(self.obstacles[i])-1].y], 'b')
    
        path_t = np.transpose(self.path)
        plt.plot(path_t[0], path_t[1])

        plt.plot(self.base_position[0], self.base_position[1], 'g*', markersize=10, label="Base")

        for id in goals_dict:
            goal = goals_dict[id]
            plt.plot(goal[0], goal[1], 'ro')
            plt.text(goal[0], goal[1] + 0.4, id, fontsize=14)

        if plot_distances:
            goals = list(goals_dict.values())
            goals.append(self.base_position)
            for i in range(len(goals) - 1):
                color = 'g'
                if np.linalg.norm(goals[i] - goals[i + 1]) > self.net_radius:
                    color = 'r'
                
                plt.plot([goals[i][0], goals[i + 1][0]], [goals[i][1], goals[i + 1][1]], color=color)

        plt.legend()
        plt.grid(True)
        plt.axis('scaled')
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.show()

    def log(self, msg, level):
        log_colors = {'error':colorama.Fore.RED, 'warn':colorama.Fore.YELLOW, 'success':colorama.Fore.GREEN}
        color = log_colors[level]
        print(color + msg + colorama.Fore.RESET)

def test_example():
    example_path = [vg.Point(11.42, 6.62), vg.Point(7.50, 8.00), vg.Point(1.50, 9.00), vg.Point(-9.00, 9.00), vg.Point(-11.90, 8.19)]
    example_path = [vg.Point(-3.94, -0.01), vg.Point(-1.41, -1.41), vg.Point(0.00, -2.00), vg.Point(1.41, -1.41), vg.Point(2.87, -0.14)]
    # example_path = [vg.Point(-3.94, -0.01), vg.Point(-1.41, -1.41), vg.Point(0.00, -2.00)]
    net_radius = 5

    solver = GoalsPlanner(example_path, net_radius, None)
    goals_dict = solver.goals()
    print(goals_dict)
    solver.plot_solution(goals_dict, plot_distances=True)
    solver.check_solution(goals_dict)


if __name__ == '__main__':    
    test_example()