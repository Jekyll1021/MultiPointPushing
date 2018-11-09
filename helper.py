import math
import numpy as np
import matplotlib.pyplot as plt


# Function to know if we have a CCW turn
def CCW(p1, p2, p3):
	if (p3[1]-p1[1])*(p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
		return True
	return False

# Main function:
def create_convex_hull(S):
	"""takes in an [np array] of points!
	and return a convex hull
	"""
	n = len(S)
	P = [None] * n
	l = np.where(S[:,0] == np.min(S[:,0]))
	pointOnHull = S[l[0][0]]
	i = 0
	while True:
		P[i] = pointOnHull
		endpoint = S[0]
		for j in range(1,n):
			if (endpoint[0] == pointOnHull[0] and endpoint[1] == pointOnHull[1]) or not CCW(S[j],P[i],endpoint):
				endpoint = S[j]
		i = i + 1
		pointOnHull = endpoint
		if endpoint[0] == P[0][0] and endpoint[1] == P[0][1]:
			break
	for i in range(n):
		if P[-1] is None:
			del P[-1]
	return np.array(P)

def euclidean_dist(pos1, pos2):
	return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def compute_centroid(vertices):
	"""
	helper function:

	input:
	vertices: a list of vertices of a polygon 
	under the assumption that all vertices are ordered either clockwise/counterclockwise

	output: 
	centroid: position of (x, y) tuple of the polygon relative to the local origin of polygon. 
	"""
	c_x = 0
	c_y = 0
	area = 0
	n = len(vertices)
	for i in range(n):
		curr = vertices[(i - n) % n]
		next = vertices[(i + 1 - n) % n]
		diff = (curr[0] * next[1] - curr[1] * next[0])
		c_x += (curr[0] + next[0]) * diff
		c_y += (curr[1] + next[1]) * diff
		area += diff
	area = area / 2
	c_x = c_x / (6 * area)
	c_y = c_y / (6 * area)
	return c_x, c_y

def compute_area(vertices):
	"""
	helper function:

	input:
	vertices: a list of vertices of a polygon 
	under the assumption that all vertices are ordered either clockwise/counterclockwise

	output: 
	centroid: position of (x, y) tuple of the polygon relative to the local origin of polygon. 
	"""
	c_x = 0
	c_y = 0
	area = 0
	n = len(vertices)
	for i in range(n):
		curr = vertices[(i - n) % n]
		next = vertices[(i + 1 - n) % n]
		diff = (curr[0] * next[1] - curr[1] * next[0])
		c_x += (curr[0] + next[0]) * diff
		c_y += (curr[1] + next[1]) * diff
		area += diff
	area = area / 2
	return abs(area)

def normalize(vector):
	"""
	helper function: 

	input:
	vector: (x, y) force vector

	output:
	vector: (x, y) force vector with normalized magnitude 1
	"""
	mag = math.sqrt(vector[0] ** 2 + vector[1] ** 2)+1e-6
	return vector[0] / mag, vector[1] / mag

def side_of_point_on_line(start_pt, end_pt, query_pt):
	det = (end_pt[0] - start_pt[0]) * (query_pt[1] - start_pt[1]) - (end_pt[1] - start_pt[1]) * (query_pt[0] - start_pt[0])
	if det > 0:
		return 1
	elif det < 0:
		return -1
	else:
		return 0

def pointToLineDistance(e1, e2, p1):
	numerator = np.abs((e2[1] - e1[1])*p1[0] - (e2[0] - e1[0])*p1[1] + e2[0]*e1[1] - e1[0]*e2[1])
	normalization = np.sqrt((e2[1] - e1[1])**2 + (e2[0] - e1[0])**2)
	return numerator/normalization

def scalarProject(start_pt, end_pt, point):
	a = np.array(point) - np.array(start_pt)
	unit_b = normalize(np.array(end_pt) - np.array(start_pt))
	return a[0]*unit_b[0]+a[1]*unit_b[1]

def projectedPtToStartDistance(e1, e2, p1):
	d1 = pointToLineDistance(e1, e2, p1)
	d2 = euclidean_dist(e1, p1)
	if abs(d1) > abs(d2):
		return None
	return math.sqrt(d2 ** 2 - d1 ** 2)

def two_line_intersect(e1, e2, e3, e4):
	denom = (e1[0]-e2[0])*(e3[1]-e4[1]) - (e1[1]-e2[1])*(e3[0]-e4[0])
	f1 = (e1[0]*e2[1] - e1[1]*e2[0])
	f2 = (e3[0]*e4[1] - e3[1]*e4[0])
	if denom == 0:
		return None
	pt = ((f1*(e3[0] - e4[0]) - f2 * (e1[0] - e2[0])) / (denom+1e-6), (f1*(e3[1] - e4[1]) - f2 * (e1[1] - e2[1]))/(denom+1e-6))
	kap = np.dot(np.array(pt) - np.array(e3), np.array(e4) - np.array(e3))
	kab = np.dot(np.array(e4) - np.array(e3), np.array(e4) - np.array(e3))
	if kap > kab or kap < 0:
		return None
	else:
		return pt
	# return pt

def find_max_contact_range(vertices, e1, e2):
	p = np.array(e1) - np.array(e2)
	vector = (1, -(p[0] / (p[1] + 1e-6)))
	# print(vector)
	max_contact_range = 0

	start_pt = None
	max_dist = 0
	for i in range(len(vertices)):
		dist = pointToLineDistance(e1, e2, vertices[i])
		if dist > max_dist:
			max_dist = dist
			start_pt = vertices[i]
	# print(start_pt)

	if not start_pt is None:
		end_pt = np.array(start_pt) + np.array(vector)
		start_pt = np.array(start_pt) - np.array(vector)

		intersect_list = set()
		for j in range(len(vertices)):
			intersect = two_line_intersect(start_pt, end_pt, vertices[j], vertices[(j + 1) % len(vertices)])
			# print(vertices[j], vertices[(j + 1) % len(vertices)])
			# print(intersect)
			if not intersect is None:
				add = True
				for pt in intersect_list:
					if euclidean_dist(pt, intersect) < 0.01: 
						add = False
				if add:
					intersect_list.add(intersect)
		# print(intersect_list)
		if len(intersect_list) == 2:
			# print(intersect_list)
			intersect_list = list(intersect_list)
			contact_range = euclidean_dist(intersect_list[0], intersect_list[1])
			if contact_range > max_contact_range:
				max_contact_range = contact_range

	# for i in range(len(vertices)):
	# 	perp_end_pt = np.array(vertices[i]) + np.array(vector)
	# 	intersect_list = []
	# 	for j in range(len(vertices)):
	# 		intersect = two_line_intersect(vertices[i], perp_end_pt, vertices[j], vertices[(j + 1) % len(vertices)])
	# 		if not intersect is None:
	# 			intersect_list.append(intersect)
	# 	print(intersect_list)
	# 	if len(intersect_list) == 2:
	# 		# print(intersect_list)
	# 		contact_range = euclidean_dist(intersect_list[0], intersect_list[1])
	# 		if contact_range > max_contact_range:
	# 			max_contact_range = contact_range
	return max_contact_range, list(intersect_list)

def find_collision_dist_convex_hull(start_pt, vector, centroid, vertices):
	abs_vertices = np.array(vertices) + np.array(centroid)
	end_pt = np.array(start_pt) + np.array(vector)
	dist = 1e2

	for i in range(len(vertices)):
		intersect = two_line_intersect(start_pt, end_pt, abs_vertices[i], abs_vertices[(i + 1) % len(abs_vertices)])
		if not intersect is None:
			if (np.array(intersect) - np.array(start_pt))[0] / (vector[0] + 1e-6) > 0 or (np.array(intersect) - np.array(start_pt))[1] / (vector[1] + 1e-6) > 0:
				dist = min(dist, euclidean_dist(intersect, start_pt))
	return dist

def find_free_space(start_pt, vector, object_lst):
	return min([find_collision_dist_convex_hull(start_pt, vector, obj.original_pos, obj.vertices) for obj in object_lst])

def parametrize_by_bounding_circle(start_pt, vector, centroid, bounding_circle_radius):
	"""parametrize as p1 to p2"""
	point = (start_pt[0] - centroid[0], start_pt[1] - centroid[1])
	a = (vector[0]**2 + vector[1]**2) + 1e-6
	b = (2 * point[0] * vector[0] + 2 * point[1] * vector[1])
	c = (point[0] ** 2 + point[1] ** 2 - bounding_circle_radius ** 2)
	if (b**2 - 4 * a * c) < 0:
		print("unable to parametrize by bounding circle: line of force does not touch bounding circle")
		return None
	else:
		t1 = (-b + math.sqrt(b**2 - 4 * a * c))/(2*a)
		t2 = (-b - math.sqrt(b**2 - 4 * a * c))/(2*a)
		p1 = (point[0] + t2 * vector[0], point[1] + t2 * vector[1])
		p2 = (point[0] + t1 * vector[0], point[1] + t1 * vector[1])
		return [np.array(normalize([p1[0], p1[1]])) * bounding_circle_radius + np.array(centroid), np.array(normalize([p2[0], p2[1]])) * bounding_circle_radius + np.array(centroid)]

def generatePolygon(min_rad=math.sqrt(2)*2/3, max_rad=math.sqrt(2), num_ver=6) :
	angles = sorted([np.random.uniform(0, 2*math.pi) for i in range(num_ver)])
	rad = [np.random.uniform(min_rad, max_rad) for i in range(num_ver)]
	
	return [[math.cos(angles[i]) * rad[i], math.sin(angles[i]) * rad[i]] for i in range(num_ver)]

def unitVector2Degree(vector):
	vector = normalize(vector)
	return math.atan2(vector[1], vector[0])*180/math.pi

def findMaxAwayVector(vector_lst):
	sum_degree = sum([unitVector2Degree(vector) for vector in vector_lst])
	radius = (sum_degree/len(vector_lst))/180*math.pi
	return normalize((math.cos(radius), math.sin(radius)))

def rotatePt(point, vector):
	radius = unitVector2Degree(vector)/180*math.pi
	x = point[0]*math.cos(radius)-point[1]*math.sin(radius)
	y = point[0]*math.sin(radius)+point[1]*math.cos(radius)
	return (x, y)

def findLoads(vertices, start_pt, end_pt):
	left_points = []
	right_points = []
	for i in range(len(vertices)):
		curr = vertices[i]
		next = vertices[(i+1) % len(vertices)]
		side_c = side_of_point_on_line(start_pt, end_pt, curr)
		side_n = side_of_point_on_line(start_pt, end_pt, next)
		# print(curr, next, side_c, side_n)
		if side_c <= 0:
			left_points.append(curr)
		if side_c >= 0:
			right_points.append(curr)
		if side_c != side_n:
			print(curr, next)
			intersect = two_line_intersect(start_pt, end_pt, curr, next)
			print(intersect)
			if not intersect is None:
				left_points.append(intersect)
				right_points.append(intersect)
	# print(left_points)
	left = create_convex_hull(np.array(left_points))
	right = create_convex_hull(np.array(right_points))
	v = compute_centroid(vertices)
	return euclidean_dist(compute_centroid(left), v), euclidean_dist(compute_centroid(right), v)


