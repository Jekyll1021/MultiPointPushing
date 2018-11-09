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
		pts = parametrize_by_bounding_circle(np.array(pt1) + np.array([env.objs[obj_ind].body.position[0], env.objs[obj_ind].body.position[1]]), np.array(pt2) - np.array(pt1), np.array([env.objs[obj_ind].body.position[0], env.objs[obj_ind].body.position[1]]), env.objs[obj_ind].bounding_circle_radius+0.1)
		if pts is not None:
			pt_lst.append(pts)
	return pt_lst

def com_only(env):
	pt_lst = []
	for ind in range(len(env.objs)*16):
		obj_ind = ind // 16
		theta = (ind % 16) * 3.14 * 2 / 16
		pt1 = np.array([env.objs[obj_ind].body.position[0], env.objs[obj_ind].body.position[1]])
		pt2 = (math.cos(theta) * env.objs[obj_ind].bounding_circle_radius, math.sin(theta) * env.objs[obj_ind].bounding_circle_radius)
		pts = parametrize_by_bounding_circle(pt1, np.array(pt2), np.array([env.objs[obj_ind].body.position[0], env.objs[obj_ind].body.position[1]]), env.objs[obj_ind].bounding_circle_radius+0.1)
		if pts is not None:
			pt_lst.append(pts)
	return pt_lst

def max_gain_only(env):
	pt_lst = []
	obj_ind = find_best_remove_object(env)
	for ind in range(16):
		theta = (ind % 16) * 3.14 * 2 / 16
		pt1 = np.array([env.objs[obj_ind].body.position[0], env.objs[obj_ind].body.position[1]])
		pt2 = (math.cos(theta) * env.objs[obj_ind].bounding_circle_radius, math.sin(theta) * env.objs[obj_ind].bounding_circle_radius)
		pts = parametrize_by_bounding_circle(pt1, np.array(pt2), np.array([env.objs[obj_ind].body.position[0], env.objs[obj_ind].body.position[1]]), env.objs[obj_ind].bounding_circle_radius+0.1)
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

def pairwise_only(env):
	pt_lst = []
	# obj_ind = find_best_remove_object(env)
	for i in range(len(env.objs)):
		for j in range(len(env.objs)):
			if i != j:
				vector = normalize(np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]) - np.array([env.objs[j].body.position[0], env.objs[j].body.position[1]]))
				pts = parametrize_by_bounding_circle(env.objs[j].original_pos, vector, env.objs[j].original_pos, env.objs[j].bounding_circle_radius+0.1)
				if pts is not None:
					pt_lst.append(pts)
	return pt_lst

def cluster_only(env):
	pt_lst = []
	cluster_lst = find_dist_cluster(env)
	cluster_to_push = cluster_lst[0]
	for i in range(1, len(cluster_lst)):
		if len(cluster_to_push) < len(cluster_lst[i]):
			cluster_to_push = cluster_lst[i]
	for i in cluster_to_push:
		for ind in range(16):
			theta = (ind % 16) * 3.14 * 2 / 16
			pt1 = np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]])
			pt2 = (math.cos(theta) * env.objs[i].bounding_circle_radius, math.sin(theta) * env.objs[i].bounding_circle_radius)
			pts = parametrize_by_bounding_circle(pt1, np.array(pt2), np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]), env.objs[i].bounding_circle_radius+0.1)
			if pts is not None:
				pt_lst.append(pts)
	return pt_lst

def cluster_pairwise_only(env):
	pt_lst = []
	cluster_lst = find_dist_cluster(env)
	cluster_to_push = cluster_lst[0]
	for i in range(1, len(cluster_lst)):
		if len(cluster_to_push) < len(cluster_lst[i]):
			cluster_to_push = cluster_lst[i]
	for i in range(len(env.objs)):
		for j in cluster_to_push:
			if i != j:
				vector = normalize(np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]) - np.array([env.objs[j].body.position[0], env.objs[j].body.position[1]]))
				pts = parametrize_by_bounding_circle(env.objs[j].original_pos, vector, env.objs[j].original_pos, env.objs[j].bounding_circle_radius+0.1)
				if pts is not None:
					pt_lst.append(pts)
	return pt_lst

def minpair_only(env):
	pt_lst = []

	pair = find_closest_pair(env, [])

	for i in pair:
		for ind in range(16):
			theta = (ind % 16) * 3.14 * 2 / 16
			pt1 = np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]])
			pt2 = (math.cos(theta) * env.objs[i].bounding_circle_radius, math.sin(theta) * env.objs[i].bounding_circle_radius)
			pts = parametrize_by_bounding_circle(pt1, np.array(pt2), np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]), env.objs[i].bounding_circle_radius+0.1)
			if pts is not None:
				pt_lst.append(pts)
	return pt_lst


def minpair_pairwise_only(env):
	pt_lst = []

	pair = find_closest_pair(env, [])

	for i in range(len(env.objs)):
		for j in pair:
			if i != j:
				vector = normalize(np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]) - np.array([env.objs[j].body.position[0], env.objs[j].body.position[1]]))
				pts = parametrize_by_bounding_circle(env.objs[j].original_pos, vector, env.objs[j].original_pos, env.objs[j].bounding_circle_radius+0.1)
				if pts is not None:
					pt_lst.append(pts)
	return pt_lst


def center_minpair_only(env):
	pt_lst = []
	candidate_pair = find_closest_pair(env, [])
	center_obj = -1
	dist = 1e2

	vertices = []
	for i in candidate_pair:
		if euclidean_dist(env.centroid, np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]])) < dist:
			dist = euclidean_dist(env.centroid, np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]))
			center_obj = i
	for ind in range(16):
		theta = (ind % 16) * 3.14 * 2 / 16
		pt1 = np.array([env.objs[center_obj].body.position[0], env.objs[center_obj].body.position[1]])
		pt2 = (math.cos(theta) * env.objs[center_obj].bounding_circle_radius, math.sin(theta) * env.objs[center_obj].bounding_circle_radius)
		pts = parametrize_by_bounding_circle(pt1, np.array(pt2), np.array([env.objs[center_obj].body.position[0], env.objs[center_obj].body.position[1]]), env.objs[center_obj].bounding_circle_radius+0.1)
		if pts is not None:
			pt_lst.append(pts)
	return pt_lst

def center_minpair_two_only(env):
	pt_lst = []
	candidate_pair = find_closest_pair(env, [])
	obj_ind = -1
	dist = 1e2
	for i in candidate_pair:
		if euclidean_dist(env.centroid, np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]])) < dist:
			dist = euclidean_dist(env.centroid, np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]))
			obj_ind = i
	for j in range(len(env.objs)):
		if j != obj_ind:
			vector = normalize(env.objs[j].original_pos - env.objs[obj_ind].original_pos)
			pts = parametrize_by_bounding_circle(env.objs[obj_ind].original_pos, vector, env.objs[obj_ind].original_pos, env.objs[obj_ind].bounding_circle_radius+0.1)
			if pts is not None:
				pt_lst.append(pts)
	return pt_lst



#######################
# construct fcc graph #
#######################

def construct_graph(env, threshold=2.3):
	graph = {}
	for i in range(len(env.objs)):
		if i not in graph.keys():
			graph[i] = []
		for j in range(len(env.objs)):
			if i != j and euclidean_dist(np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]), \
							np.array([env.objs[j].body.position[0], env.objs[j].body.position[1]])) < threshold:
				if j not in graph[i]:
					graph[i].append(j)
	return graph

def find_fcc_cluster(env):
	graph = construct_graph(env)
	s = []
	d = set()
	cluster_lst = []

	for i in range(len(env.objs)):
		cluster = set()
		if i not in d:
			s.append(i)
		while s != []:
			v = s.pop()
			if v not in d:
				d.add(v)
				cluster.add(v)
				for v2 in graph[v]:
					s.append(v2)
		if len(cluster) > 0:
			cluster_lst.append(list(cluster))
	return cluster_lst

def fcc_cluster_only(env):
	pt_lst = []
	cluster_lst = find_fcc_cluster(env)
	cluster_to_push = cluster_lst[0]
	for i in range(1, len(cluster_lst)):
		if len(cluster_to_push) < len(cluster_lst[i]):
			cluster_to_push = cluster_lst[i]
	# print(len(cluster_to_push))
	for i in cluster_to_push:
		for ind in range(16):
			theta = (ind % 16) * 3.14 * 2 / 16
			pt1 = np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]])
			pt2 = (math.cos(theta) * env.objs[i].bounding_circle_radius, math.sin(theta) * env.objs[i].bounding_circle_radius)
			pts = parametrize_by_bounding_circle(pt1, np.array(pt2), np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]), env.objs[i].bounding_circle_radius+0.1)
			if pts is not None:
				pt_lst.append(pts)
	return pt_lst

def fcc_cluster_center_only(env):
	pt_lst = []
	cluster_lst = find_fcc_cluster(env)
	cluster_to_push = cluster_lst[0]
	for i in range(1, len(cluster_lst)):
		if len(cluster_to_push) < len(cluster_lst[i]):
			cluster_to_push = cluster_lst[i]
	# print(len(cluster_to_push))
	if len(cluster_to_push) <= 2:
		push_obj = cluster_to_push[0]
	else:
		centroid_lst = []
		for i in cluster_to_push:
			centroid_lst.append([env.objs[i].body.position[0], env.objs[i].body.position[1]])
		centroid = compute_centroid(centroid_lst)
		push_obj = cluster_to_push[0]
		min_dist = euclidean_dist([env.objs[cluster_to_push[0]].body.position[0], env.objs[cluster_to_push[0]].body.position[1]], centroid)
		for i in cluster_to_push:
			if euclidean_dist([env.objs[i].body.position[0], env.objs[i].body.position[1]], centroid) < min_dist:
				min_dist = euclidean_dist([env.objs[i].body.position[0], env.objs[i].body.position[1]], centroid)
				push_obj = i

	for ind in range(16):
		theta = (ind % 16) * 3.14 * 2 / 16
		pt1 = np.array([env.objs[push_obj].body.position[0], env.objs[push_obj].body.position[1]])
		pt2 = (math.cos(theta) * env.objs[push_obj].bounding_circle_radius, math.sin(theta) * env.objs[push_obj].bounding_circle_radius)
		pts = parametrize_by_bounding_circle(pt1, np.array(pt2), np.array([env.objs[push_obj].body.position[0], env.objs[push_obj].body.position[1]]), env.objs[push_obj].bounding_circle_radius+0.1)
		if pts is not None:
			pt_lst.append(pts)
	return pt_lst

#######################
# construct scc graph #
#######################

## under progress

def construct_graph(env, threshold=2.3):
	graph = {}
	for i in range(len(env.objs)):
		if i not in graph.keys():
			graph[i] = []
		for j in range(i+1, len(env.objs)):
			if i != j and euclidean_dist(np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]), \
							np.array([env.objs[j].body.position[0], env.objs[j].body.position[1]])) < threshold:
				if j not in graph[i]:
					graph[i].append(j)
	return graph


