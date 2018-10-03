import math
import numpy as np
from helper import *
import copy
from policies import *

def no_prune(env):
	pt_lst = []
	for ind in range(len(env.objs)*256):
		obj_ind = ind // 256
		theta1 = ((ind % 256) // 16) * 3.14 * 2 / 16
		theta2 = (ind % 16) * 3.14 * 2 / 16
		if theta1 == theta2:
			continue
		pt1 = (math.cos(theta1) * env.objs[obj_ind].bounding_circle_radius, math.sin(theta1) * env.objs[obj_ind].bounding_circle_radius)
		pt2 = (math.cos(theta2) * env.objs[obj_ind].bounding_circle_radius, math.sin(theta2) * env.objs[obj_ind].bounding_circle_radius)
		pts = parametrize_by_bounding_circle(np.array(pt1) + np.array(env.objs[obj_ind].original_pos), np.array(pt2) - np.array(pt1), env.objs[obj_ind].original_pos, env.objs[obj_ind].bounding_circle_radius+0.1)
		if pts is not None:
			pt_lst.append(pts)
	return pt_lst

def com_only(env):
	pt_lst = []
	for ind in range(len(env.objs)*16):
		obj_ind = ind // 16
		theta = (ind % 16) * 3.14 * 2 / 16
		pt1 = np.array(env.objs[obj_ind].original_pos)
		pt2 = (math.cos(theta) * env.objs[obj_ind].bounding_circle_radius, math.sin(theta) * env.objs[obj_ind].bounding_circle_radius)
		pts = parametrize_by_bounding_circle(pt1, np.array(pt2), env.objs[obj_ind].original_pos, env.objs[obj_ind].bounding_circle_radius+0.1)
		if pts is not None:
			pt_lst.append(pts)
	return pt_lst

def max_gain_only(env):
	pt_lst = []
	obj_ind = find_best_remove_object(env)
	for ind in range(16):
		theta = (ind % 16) * 3.14 * 2 / 16
		pt1 = np.array(env.objs[obj_ind].original_pos)
		pt2 = (math.cos(theta) * env.objs[obj_ind].bounding_circle_radius, math.sin(theta) * env.objs[obj_ind].bounding_circle_radius)
		pts = parametrize_by_bounding_circle(pt1, np.array(pt2), env.objs[obj_ind].original_pos, env.objs[obj_ind].bounding_circle_radius+0.1)
		if pts is not None:
			pt_lst.append(pts)
	return pt_lst

def two_only(env):
	pt_lst = []
	obj_ind = find_best_remove_object(env)
	for i in range(len(env.objs)):
		if i != obj_ind:
			vector = normalize(env.objs[i].original_pos - env.objs[obj_ind].original_pos)
			pts = parametrize_by_bounding_circle(env.objs[obj_ind].original_pos, vector, env.objs[obj_ind].original_pos, env.objs[obj_ind].bounding_circle_radius+0.1)
			if pts is not None:
				pt_lst.append(pts)
	return pt_lst

# def one_only(env):
# 	pt_lst = []
# 	obj_ind = find_best_remove_object(env)
# 	for ind in range(16):
