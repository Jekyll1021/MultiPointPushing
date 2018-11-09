import math
import numpy as np
from helper import *
import copy

####################
# Helper functions #
####################

def find_best_remove_object(env, removal_lst=[]):
	# select object to push
	push_obj = -1
	max_dist_sum = 0
	for obj in range(len(env.objs)):
		if not obj in removal_lst:
			dist_sum = 0
			for i in range(len(env.objs) - 1):
				for j in range(i + 1, len(env.objs)):
					if i != obj and j != obj:
						dist_sum += math.log(euclidean_dist(np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]), env.objs[j].original_pos))
			if dist_sum > max_dist_sum:
				push_obj = obj
				max_dist_sum = dist_sum

	return push_obj

def find_dist_center_obj_cluster(env, max_dist=2.3):
	center_obj = -1
	dist = 1e2
	for i in range(len(env.objs)):
		if euclidean_dist(env.centroid, env.objs[i].original_pos) < dist:
			dist = euclidean_dist(env.centroid, env.objs[i].original_pos)
			center_obj = i

	cluster_lst = []
	cluster = [center_obj]
	leftoff = []
	for i in range(len(env.objs)):
		if (i != center_obj) and euclidean_dist(env.objs[center_obj].original_pos, env.objs[i].original_pos) < max_dist:
			cluster.append(i)
		elif (i != center_obj):
			leftoff.append(i)
	cluster_lst.append(cluster)
	
	cluster = []
	
	while leftoff != []:
		center_obj = leftoff[0]
		
		cluster = [center_obj]

		new_leftoff = []
		for obj in leftoff:
			if (obj != center_obj) and euclidean_dist(env.objs[center_obj].original_pos, env.objs[obj].original_pos) < max_dist:
				cluster.append(obj)
			elif (obj != center_obj):
				new_leftoff.append(obj)
		cluster_lst.append(cluster)
		cluster = []
		leftoff = new_leftoff

	return cluster_lst

def find_dist_cluster(env, max_dist=2.3):
	candidate_pair = find_closest_pair(env, [])
	center_obj = -1
	dist = 1e2
	for i in candidate_pair:
		if euclidean_dist(env.centroid, np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]])) < dist:
			dist = euclidean_dist(env.centroid, np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]))
			center_obj = i

	cluster_lst = []
	cluster = [center_obj]
	leftoff = []
	for i in range(len(env.objs)):
		if (i != center_obj) and euclidean_dist(np.array([env.objs[center_obj].body.position[0], env.objs[center_obj].body.position[1]]), np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]])) < max_dist:
			cluster.append(i)
		elif (i != center_obj):
			leftoff.append(i)
	cluster_lst.append(cluster)
	
	cluster = []
	
	while leftoff != []:
		center_obj = leftoff[0]
		
		cluster = [center_obj]

		new_leftoff = []
		for obj in leftoff:
			if (obj != center_obj) and euclidean_dist(np.array([env.objs[center_obj].body.position[0], env.objs[center_obj].body.position[1]]), np.array([env.objs[obj].body.position[0], env.objs[obj].body.position[1]])) < max_dist:
				cluster.append(obj)
			elif (obj != center_obj):
				new_leftoff.append(obj)
		cluster_lst.append(cluster)
		cluster = []
		leftoff = new_leftoff

	return cluster_lst

def find_closest_pair(env, removal_lst):
	# list can be empty
	min_dist = 100
	pair = []
	for i in range(len(env.objs) - 1):
		for j in range(i + 1, len(env.objs)):
			if (not i in removal_lst) and (not j in removal_lst):
				if euclidean_dist(np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]), np.array([env.objs[j].body.position[0], env.objs[j].body.position[1]])) < min_dist:
					min_dist = euclidean_dist(np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]), np.array([env.objs[j].body.position[0], env.objs[j].body.position[1]]))
					pair = [i, j]
	return pair

def find_closest_ranking_to_object(env, obj):
	dic = {}
	for i in range(len(env.objs)):
		if i != obj:
			dic[i] = euclidean_dist(np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]), env.objs[obj].original_pos)
	return [k for k in sorted(dic, key=dic.get)]

def find_clusters(env, cluster_num, first_obj=-1):
	if cluster_num == 1:
		return [list(range(len(env.objs)))]

	if first_obj == -1:
		first_obj = 0
		for obj in range(len(env.objs)):
			if euclidean_dist(env.objs[first_obj].original_pos, env.centroid) < euclidean_dist(env.objs[obj].original_pos, env.centroid):
				first_obj = obj

	cluster_center = [first_obj]
	cluster_lst = []

	for i in range(cluster_num - 1):
		max_sum_dist = 0
		center_item = None
		for obj in range(len(env.objs)):
			if obj not in cluster_center:
				sum_dist = sum([euclidean_dist(env.objs[obj].original_pos, env.objs[c].original_pos) for c in cluster_center])
				if sum_dist > max_sum_dist:
					center_item = obj
					max_sum_dist = sum_dist
		if center_item is not None:
			cluster_center.append(center_item)

	for c in cluster_center:
		cluster_lst.append([c])

	for obj in range(len(env.objs)):
		if obj not in cluster_center:
			center = None
			min_dist = 1e2
			group = -1
			for i in range(len(cluster_center)):
				if euclidean_dist(env.objs[cluster_center[i]].original_pos, env.objs[obj].original_pos) < min_dist:
					group = i
					min_dist = euclidean_dist(env.objs[cluster_center[i]].original_pos, env.objs[obj].original_pos)
			cluster_lst[group].append(obj)
			vertices_lst = []

	return cluster_lst

############
# Policies #
############

def proposed0(env):
	push_obj = find_best_remove_object(env)
	dist_lst = find_closest_ranking_to_object(env, push_obj)
	seg = np.array(env.objs[dist_lst[0]].original_pos) - np.array(env.objs[dist_lst[1]].original_pos)
	vector1 = (1, -(seg[0] / (seg[1]+1e-8)))
	vector2 = (-1, (seg[0] / (seg[1]+1e-8)))
	vector1 = normalize(vector1)
	vector2 = normalize(vector2)
	max_away = normalize(findMaxAwayVector([env.objs[push_obj].original_pos - env.objs[dist_lst[0]].original_pos, env.objs[push_obj].original_pos - env.objs[dist_lst[1]].original_pos]))

	# print(vector1, vector2, max_away, euclidean_dist(vector1, max_away), euclidean_dist(vector2, max_away))
	if euclidean_dist(vector1, max_away) < euclidean_dist(vector2, max_away):
		# print(1)
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector1, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
	else:
		# print(2)
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector2, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
	
	return pts

def proposed1(env):
	push_obj = find_best_remove_object(env)
	vertices = np.array(env.objs[push_obj].vertices) + np.array(env.objs[push_obj].original_pos)
	min_contact_range = 1e2
	push_vector = None
	for j in range(16):
		vector = (math.cos(2*j*3.14 / 16), math.sin(2*j*3.14 / 16))
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		contact_range, range_pt_lst = find_max_contact_range(vertices, pts[0], pts[1])
		if contact_range != 0 and contact_range < min_contact_range:
			min_contact_range = contact_range
			push_vector = vector
	if not push_vector is None:
		dist_lst = find_closest_ranking_to_object(env, push_obj)
		vector1 = normalize(push_vector)
		vector2 = normalize((-push_vector[0], -push_vector[1]))
		max_away = normalize(findMaxAwayVector([env.objs[push_obj].original_pos - env.objs[dist_lst[i]].original_pos for i in range(len(dist_lst))]))
		if euclidean_dist(vector1, max_away) < euclidean_dist(vector2, max_away):
			pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector1, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		else:
			pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector2, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		return pts
	return None

def proposed2(env):
	push_obj = find_best_remove_object(env)
	max_dist_sum = 0
	push_pts = None
	for j in range(16):
		vector = (math.cos(2*j*3.14 / 16), math.sin(2*j*3.14 / 16))
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		min_dist_l = 1e2
		min_dist_r = 1e2
		for k in range(len(env.objs)):
			if k != push_obj and scalarProject(pts[0], pts[1], env.objs[k].original_pos) > 0:
				side_com = side_of_point_on_line(pts[0], pts[1], env.objs[k].original_pos)
				if side_com < 0 and pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius < min_dist_l:
					min_dist_l = pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius
				if side_com > 0 and pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius < min_dist_r:
					min_dist_r = pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius
		if min_dist_l + min_dist_r > max_dist_sum:
			max_dist_sum = min_dist_l + min_dist_r
			push_pts = pts
	return push_pts

def proposed3(env):
	push_obj = find_best_remove_object(env)
	min_dist_sum = 1e2
	push_pts = None
	for j in range(16):
		vector = (math.cos(2*j*3.14 / 16), math.sin(2*j*3.14 / 16))
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		max_dist_l = 0
		max_dist_r = 0
		for k in range(len(env.objs)):
			if k != push_obj and scalarProject(pts[0], pts[1], env.objs[k].original_pos) > 0:
				side_com = side_of_point_on_line(pts[0], pts[1], env.objs[k].original_pos)
				if side_com < 0 and pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius < 1 and (pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius - env.objs[push_obj].bounding_circle_radius) ** 2 * (4 - scalarProject(pts[0], pts[1], env.objs[k].original_pos)) / 4 > max_dist_l: #* (1 - scalarProject(pts[0], pts[1], env.objs[k].original_pos))
					max_dist_l = (pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius - 1) ** 2 * (4 - scalarProject(pts[0], pts[1], env.objs[k].original_pos)) / 4
				if side_com > 0 and pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius < 1 and (pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius - env.objs[push_obj].bounding_circle_radius) ** 2 * (4 - scalarProject(pts[0], pts[1], env.objs[k].original_pos)) / 4 > max_dist_r:
					max_dist_r = (pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius - 1) ** 2 * (4 - scalarProject(pts[0], pts[1], env.objs[k].original_pos)) / 4
		if max_dist_l + max_dist_r <= min_dist_sum:
			min_dist_sum = max_dist_l + max_dist_r
			push_pts = pts
	max_away = normalize(findMaxAwayVector([env.objs[push_obj].original_pos - np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]) for i in range(len(env.objs)) if i != push_obj]))
	vector = normalize(np.array(pts[1]) - np.array(pts[0]))
	# print(min_dist_sum, euclidean_dist(vector, max_away))
	return push_pts
	
def proposed4(env):
	push_obj = find_best_remove_object(env)
	min_dist_sum = 1e2
	push_pts = None
	for j in range(16):
		vector = (math.cos(2*j*3.14 / 16), math.sin(2*j*3.14 / 16))
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		max_dist_l = 0
		max_dist_r = 0
		for k in range(len(env.objs)):
			if k != push_obj and scalarProject(pts[0], pts[1], env.objs[k].original_pos) > 0:
				side_com = side_of_point_on_line(pts[0], pts[1], env.objs[k].original_pos)
				if side_com < 0 and env.objs[k].bounding_circle_radius - pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) > 0 and env.objs[k].bounding_circle_radius - pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) > max_dist_l:
					max_dist_l = pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius
				if side_com > 0 and env.objs[k].bounding_circle_radius - pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) > 0 and env.objs[k].bounding_circle_radius - pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) > max_dist_r:
					max_dist_r = pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius
		if max_dist_l + max_dist_r < min_dist_sum:
			min_dist_sum = max_dist_l + max_dist_r
			push_pts = pts
	return push_pts

def proposed5(env):
	push_obj = find_best_remove_object(env)
	min_dist_sum = 1e2
	push_pts = None
	for j in range(16):
		vector = (math.cos(2*j*3.14 / 16), math.sin(2*j*3.14 / 16))
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		min_dist_l = 1e2
		min_dist_r = 1e2
		for k in range(len(env.objs)):
			if k != push_obj and scalarProject(pts[0], pts[1], env.objs[k].original_pos) > 0:
				side_com = side_of_point_on_line(pts[0], pts[1], env.objs[k].original_pos)
				if side_com < 0 and pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius < min_dist_l:
					min_dist_l = pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius
				if side_com > 0 and pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius < min_dist_r:
					min_dist_r = pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius
		if min_dist_l + min_dist_r < min_dist_sum:
			min_dist_sum = min_dist_l + min_dist_r
			push_pts = pts
	return push_pts

def proposed6(env):
	"""separate two cluster"""
	cluster_lst = find_clusters(env, 2)

	cluster_ind = np.argmax([len(lst) for lst in cluster_lst])
	cluster = cluster_lst[cluster_ind]
	vertices_lst = []
	for obj in cluster:
		vertices_lst.extend((np.array(env.objs[obj].vertices)+np.array(env.objs[obj].original_pos)).tolist())
	cluster_center = compute_centroid(create_convex_hull(np.array(vertices_lst)))

	other_cluster = cluster_lst[1 - cluster_ind]
	vertices_lst = []
	for obj in other_cluster:
		vertices_lst.extend((np.array(env.objs[obj].vertices)+np.array(env.objs[obj].original_pos)).tolist())
	other_cluster_center = compute_centroid(create_convex_hull(np.array(vertices_lst)))

	min_dist = 1e2
	push_pts = None
	ref_v = normalize(np.array(cluster_center) - np.array(other_cluster_center))
	for obj in cluster:
		v = normalize(np.array(cluster_center) - np.array(env.objs[obj].original_pos))
		if euclidean_dist(ref_v, v) < min_dist:
			min_dist = euclidean_dist(ref_v, v)
			push_pts = parametrize_by_bounding_circle(env.objs[obj].original_pos, v, env.objs[obj].original_pos, env.objs[obj].bounding_circle_radius+0.1)
	return push_pts

def proposed7(env):
	"""find two objects and smash them together"""
	push_obj = find_best_remove_object(env)
	target_obj = find_best_remove_object(env, [push_obj])
	v = np.array(env.objs[target_obj].original_pos - env.objs[push_obj].original_pos)
	pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, v, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
	return pts

def proposed8(env):
	"""group objects into 3 clusters and do proposed2"""
	push_obj = find_best_remove_object(env)
	cluster_lst = find_clusters(env, 3, first_obj=push_obj)
	cluster_center_list = []
	cluster_bounding_radius_list = []
	for cluster in cluster_lst:
		vertices_lst = []
		for obj in cluster:
			vertices_lst.extend((np.array(env.objs[obj].vertices)+np.array(env.objs[obj].original_pos)).tolist())
		cluster_center = compute_centroid(create_convex_hull(np.array(vertices_lst)))
		cluster_center_list.append(cluster_center)
		cluster_bounding_radius = max([euclidean_dist(vertices_lst[i], cluster_center) for i in range(len(vertices_lst))])
		cluster_bounding_radius_list.append(cluster_bounding_radius)

	max_dist_sum = 0
	push_pts = None
	for j in range(16):
		vector = (math.cos(2*j*3.14 / 16), math.sin(2*j*3.14 / 16))
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		min_dist_l = 1e2
		min_dist_r = 1e2
		for k in range(len(cluster_lst)):
			if (not push_obj in cluster_lst[k]) and scalarProject(pts[0], pts[1], cluster_center_list[k]) > 0:
				side_com = side_of_point_on_line(pts[0], pts[1], cluster_center_list[k])
				if side_com < 0 and pointToLineDistance(pts[0], pts[1], cluster_center_list[k]) - cluster_bounding_radius_list[k] < min_dist_l:
					min_dist_l = pointToLineDistance(pts[0], pts[1], cluster_center_list[k]) - cluster_bounding_radius_list[k]
				if side_com > 0 and pointToLineDistance(pts[0], pts[1], cluster_center_list[k]) - cluster_bounding_radius_list[k] < min_dist_r:
					min_dist_r = pointToLineDistance(pts[0], pts[1], cluster_center_list[k]) - cluster_bounding_radius_list[k]
		if min_dist_l + min_dist_r > max_dist_sum:
			max_dist_sum = min_dist_l + min_dist_r
			push_pts = pts
	return push_pts

def proposed9(env):
	cluster_lst = find_dist_center_obj_cluster(env)
	push_obj = cluster_lst[0][0]
	push_pts = None

	if len(cluster_lst[0]) == max([len(cluster) for cluster in cluster_lst]) and len(cluster_lst) != 1 and (not (len(cluster_lst) == 2 and len(cluster_lst[0]) == 2)):
		max_away = normalize(findMaxAwayVector([env.objs[push_obj].original_pos - np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]) for i in range(len(env.objs)) if i not in cluster_lst[0]]))
		dist = 1e2
		
		for obj in cluster_lst[0]:
			if obj != push_obj:
				vector = normalize(env.objs[obj].original_pos - env.objs[push_obj].original_pos)
				if euclidean_dist(vector, max_away) < dist:
					dist = euclidean_dist(vector, max_away)
					push_pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
	else:
		push_pts = proposed8(env)

	return push_pts

def proposed9_sequential(env):
	cluster_lst = find_dist_cluster(env)
	push_obj = cluster_lst[0][0]
	push_pts = None
	# print(len(cluster_lst))

	if len(cluster_lst[0]) == max([len(cluster) for cluster in cluster_lst]) and ((len(cluster_lst[0]) > 2)):
		if len(cluster_lst) > 1:
			max_away = normalize(findMaxAwayVector([np.array([env.objs[push_obj].body.position[0], env.objs[push_obj].body.position[1]]) - np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]) for i in range(len(env.objs)) if i not in cluster_lst[0]]))
		else: 
			max_away = normalize(findMaxAwayVector([np.array([env.objs[push_obj].body.position[0], env.objs[push_obj].body.position[1]]) - np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]) for i in range(len(env.objs))]))
		dist = 1e2
		
		for obj in cluster_lst[0]:
			if obj != push_obj:
				vector = normalize(np.array([env.objs[obj].body.position[0], env.objs[obj].body.position[1]]) - np.array([env.objs[push_obj].body.position[0], env.objs[push_obj].body.position[1]]))
				if euclidean_dist(vector, max_away) < dist:
					dist = euclidean_dist(vector, max_away)
					push_pts = parametrize_by_bounding_circle(np.array([env.objs[push_obj].body.position[0], env.objs[push_obj].body.position[1]]), vector, np.array([env.objs[push_obj].body.position[0], env.objs[push_obj].body.position[1]]), env.objs[push_obj].bounding_circle_radius+0.1)
	else:
		cluster_to_push = cluster_lst[0]
		for i in range(1, len(cluster_lst)):
			if len(cluster_to_push) < len(cluster_lst[i]):
				cluster_to_push = cluster_lst[i]
		if len(cluster_to_push) == 1:
			push_pts = proposed8(env)
		else:
			vertices_lst = []
			for o in cluster_to_push:
				vertices_lst.extend((env.objs[o].vertices+np.array([env.objs[o].body.position[0], env.objs[o].body.position[1]])).tolist())
			cluster_center = compute_centroid(create_convex_hull(np.array(vertices_lst)))
			dist = 1e2
			push_pts = None
			for o in cluster_to_push:
				vector = normalize(np.array([env.objs[o].body.position[0], env.objs[o].body.position[1]]) - np.array(cluster_center))
				away = normalize(findMaxAwayVector([np.array([env.objs[o].body.position[0], env.objs[o].body.position[1]]) - np.array([env.objs[i].body.position[0], env.objs[i].body.position[1]]) for i in range(len(env.objs)) if i not in cluster_to_push]))
				if euclidean_dist(vector, away) < dist:
					push_pts = parametrize_by_bounding_circle(np.array([env.objs[o].body.position[0], env.objs[o].body.position[1]]), vector,np.array([env.objs[o].body.position[0], env.objs[o].body.position[1]]), env.objs[o].bounding_circle_radius+0.1)
					dist = euclidean_dist(vector, away)

	return push_pts

def boundaryShear(env):
	max_free_space = 0
	obj = None
	p = None
	v = None
	candidate_pair = find_closest_pair(env, [])
	linking_line = np.array(env.objs[candidate_pair[0]].original_pos) - np.array(env.objs[candidate_pair[1]].original_pos)
	vector_lst = [(1, -(linking_line[0] / (linking_line[1]+1e-8))), (-1, (linking_line[0] / (linking_line[1]+1e-8)))]
	for candidate in candidate_pair:
		for vt in vector_lst:
			free_space = find_free_space(np.array(env.objs[candidate].original_pos), vt, env.objs)
			if free_space > max_free_space:
				max_free_space = free_space
				p = np.array(env.objs[candidate].original_pos)
				v = vt
				obj = candidate
	if (not p is None) and (not v is None):
		pts = parametrize_by_bounding_circle(p, v, env.objs[obj].original_pos, env.objs[obj].bounding_circle_radius+0.1)
		return pts
	return None

def clusterDiffusion(env):
	cluster_num = (len(env.objs)-1) // 3 + 1
	cluster_lst = find_clusters(env, cluster_num)
	cluster_center_lst = []
	for cluster in cluster_lst:
		vertices_lst = []
		for obj in cluster:
			vertices_lst.extend((np.array(env.objs[obj].vertices)+np.array(env.objs[obj].original_pos)).tolist())
		cluster_center = compute_centroid(create_convex_hull(np.array(vertices_lst)))
		cluster_center_lst.append(cluster_center)

	push_pts = None

	min_dist = 1e2

	# max_free_space = 0

	for obj in range(len(env.objs)):
		for i in range(len(cluster_lst)): 
			if obj in cluster_lst[i]:
				vector = normalize(env.objs[obj].original_pos - np.array(cluster_center_lst[i]))
				max_away = normalize(findMaxAwayVector([env.objs[obj].original_pos - env.objs[o].original_pos for o in range(len(env.objs)) if o != obj]))
				if euclidean_dist(vector, max_away) < min_dist:
					min_dist = euclidean_dist(vector, max_away)
					push_pts = parametrize_by_bounding_circle(env.objs[obj].original_pos, vector, env.objs[obj].original_pos, env.objs[obj].bounding_circle_radius+0.1)
				# free_space = find_free_space(np.array(env.objs[obj].original_pos), vector, env.objs)
				# if free_space > max_free_space:
				# 	max_free_space = free_space
				# 	push_pts = parametrize_by_bounding_circle(env.objs[obj].original_pos, vector, env.objs[obj].original_pos, env.objs[obj].bounding_circle_radius+0.1)
	# print(push_pts)
			
	return push_pts

def maximumClearanceRatio(env):
	p_lst = [np.array(obj.original_pos) for obj in env.objs]
	v_lst = [(math.cos(i * 3.14* 2 / 16), math.sin(i * 3.14* 2 / 16)) for i in range(16)]
	max_free_space = 0
	p = None
	v = None
	candidate = None
	for obj_ind in range(len(env.objs)):
		for vt in v_lst:
			free_space = find_free_space(np.array(env.objs[obj_ind].original_pos), vt, env.objs) / find_free_space(np.array(env.objs[obj_ind].original_pos), (-vt[0], -vt[1]), env.objs)
			if free_space > max_free_space:
				max_free_space = free_space
				p = np.array(env.objs[obj_ind].original_pos)
				v = vt
				candidate = obj_ind
	if (not p is None) and (not v is None):
		pts = parametrize_by_bounding_circle(p, v, env.objs[candidate].original_pos, env.objs[candidate].bounding_circle_radius+0.1)
	return pts




