import logging
import math
import time
import copy

# import pygame
# from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import numpy as np
import pickle

import Box2D  
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody)

import matplotlib.pyplot as plt
import os
from helper import *
import json
import random
from policies import *
from prune import *
from behavioral_clone import *

PPM = 60.0  # pixels per meter
TIME_STEP = 0.1
SCREEN_WIDTH, SCREEN_HEIGHT = 720, 720

GROUPS = [(1, 1, 15), (0.75, 1, 12), (0.5, 1, 9), (0.25, 1, 6), (0.25, 1, 15)]


num_classes = 8
base = plt.cm.get_cmap('Accent')
Color_list = base(np.linspace(0, 1, num_classes))

Colors = [(int(255*c[0]), int(255*c[1]), int(255*c[2]), int(255*c[3])) for c in Color_list]

class Action:
	def __init__(self, vector, point):
		"""
		action that consists of:
		vector: (x, y) force vector
		point: (x, y) point of contact

		all relative to the local origin of polygon.
		"""
		self.vector = normalize(vector)
		self.point = point

	def __eq__(self, other):
		"""
		check if vector == vector, point == point
		"""
		return self.vector == other.vector and self.point == self.point

class Polygon:
	def __init__(self, body, fixtures, vertices, color=(255, 255, 255, 255)):
		"""body: polygon shape (dynamicBody)
		fixture: fixture
		vertices: list of relative coordinates
		"""
		self.body = body
		self.fixtures = fixtures
		self.vertices = vertices
		self.color = color
		self.original_pos = np.array(self.body.position)
		self.bounding_circle_radius = math.sqrt(max((self.vertices)[:,0]**2 + (self.vertices)[:,1]**2))
		self.disk_coverage = compute_area(self.vertices)/(self.bounding_circle_radius**2*math.pi)

	def test_overlap(self, other_vertices, other_centroid):
		"""test if the current polygon overlaps with a polygon of another polygon 
		with other_centroid as centroid and other_vertices as vertices(all in numpy array)"""
		abs_vertices = (other_vertices + other_centroid).tolist()
		for i in range(len(abs_vertices)):
			for fix in self.fixtures:
				if fix.TestPoint(abs_vertices[i]) \
				or fix.TestPoint(((np.array(abs_vertices[i])+other_centroid)/2).tolist())\
				or fix.TestPoint(((np.array(abs_vertices[i])+np.array(abs_vertices[i-1]))/2).tolist()):
					return True
		for fix in self.fixtures:
			if fix.TestPoint(other_centroid):
				return True
		return False

class SingulationEnv:
	def __init__(self):

		self.world = world(gravity=(0, 0), doSleep=True)

		self.objs = []

		self.rod = None
		self.rod2 = None

		self.bounding_convex_hull = np.array([])
		self.centroid = (0, 0)
		self.bounding_circle_radius = 0

	def create_random_env(self, num_objs=3, group=0):
		assert num_objs >= 1

		if len(self.objs) > 0:
			for obj in self.objs:
				for fix in obj.fixtures:
					obj.body.DestroyFixture(fix)
				self.world.DestroyBody(obj.body)
		self.objs = []

		for i in range(num_objs): 
			# create shape
			# vertices = create_convex_hull(np.array([(np.random.uniform(-1,1),np.random.uniform(-1,1)) for i in range(9)]))

			vertices = generatePolygon(GROUPS[group][0], GROUPS[group][1], GROUPS[group][2])
			raw_com = np.array(compute_centroid(vertices))
			vertices = (vertices - raw_com)
			bounding_r = math.sqrt(max((vertices)[:,0]**2 + (vertices)[:,1]**2))
			vertices = vertices / bounding_r

			if len(self.objs) <= 0:
				original_pos = np.array([np.random.uniform(4,8),np.random.uniform(4,8)])
				body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
				fixture = body.CreatePolygonFixture(density=1, vertices=vertices.tolist(), friction=0.5)
				self.objs.append(Polygon(body, [fixture], vertices, Colors[i]))
			else:
				max_iter = 1000
				while True:
					max_iter -= 1
					original_pos = np.array([np.random.uniform(-1.8,1.8),np.random.uniform(-1.8,1.8)]) + np.array(self.objs[-1].body.position)
					# original_pos = np.array([np.random.uniform(1,11),np.random.uniform(1,11)])
					no_overlap = True
					original_pos = np.clip(original_pos, 1, 11)

					body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
					fixture = body.CreatePolygonFixture(density=1, vertices=vertices.tolist(), friction=0.5)
					# curr_polygon = Polygon(body, fixture, vertices, Colors[i])
					curr_polygon = Polygon(body, [fixture], vertices, Colors[i % len(Colors)])
					for obj in self.objs:
						if obj.test_overlap(vertices, original_pos) or curr_polygon.test_overlap(obj.vertices, np.array(obj.body.position)):
							no_overlap = False
					if no_overlap:
						self.objs.append(curr_polygon)
						# prev_pos = original_pos
						break
					else:
						body.DestroyFixture(fixture)
						self.world.DestroyBody(body)
					if max_iter <= 0:
						raise Exception("max iter reaches")

		self.bounding_convex_hull = create_convex_hull(np.concatenate([obj.vertices+obj.original_pos for obj in self.objs]))
		self.centroid = compute_centroid(self.bounding_convex_hull.tolist())
		self.bounding_circle_radius = math.sqrt(max((self.bounding_convex_hull - np.array(self.centroid))[:,0]**2 + (self.bounding_convex_hull - np.array(self.centroid))[:,1]**2))

	def create_random_concave_env(self, num_objs=3):
		assert num_objs >= 1

		if len(self.objs) > 0:
			for obj in self.objs:
				for fix in obj.fixtures:
					obj.body.DestroyFixture(fix)
				self.world.DestroyBody(obj.body)
		self.objs = []

		for i in range(num_objs): 
			# create shape
			vertices = generatePolygon()
			raw_com = np.array(compute_centroid(vertices))
			vertices = (vertices - raw_com)

			if len(self.objs) <= 0:
				original_pos = np.array([np.random.uniform(3,6),np.random.uniform(3,6)])
				body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
				fixtures = []
				for j in range(len(vertices)):
					fixture = body.CreatePolygonFixture(density=1, vertices=vertices[np.array([j, (j+1) % len(vertices), (j+2) % len(vertices)])].tolist(), friction=0.5)
					fixtures.append(fixture)
				self.objs.append(Polygon(body, fixtures, vertices, Colors[i]))
			else:
				max_iter = 1000
				while True:
					max_iter -= 1
					original_pos = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1)]) + np.array(self.objs[-1].body.position)
					no_overlap = True

					body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
					fixtures = []
					for j in range(len(vertices) - 2):
						fixture = body.CreatePolygonFixture(density=1, vertices=vertices[j:3+j].tolist(), friction=0.5)
						fixtures.append(fixture)
					curr_polygon = Polygon(body, fixtures, vertices, Colors[i % len(Colors)])
					for obj in self.objs:
						if obj.test_overlap(vertices, original_pos) or curr_polygon.test_overlap(obj.vertices, np.array(obj.body.position)):
							no_overlap = False
					if no_overlap:
						self.objs.append(curr_polygon)
						break
					else:
						for fixture in curr_polygon.fixtures:
							body.DestroyFixture(fixture)
						self.world.DestroyBody(body)
					if max_iter <= 0:
						raise Exception("max iter reaches")

		self.bounding_convex_hull = create_convex_hull(np.concatenate([obj.vertices+obj.original_pos for obj in self.objs]))
		self.centroid = compute_centroid(self.bounding_convex_hull.tolist())
		self.bounding_circle_radius = math.sqrt(max((self.bounding_convex_hull - np.array(self.centroid))[:,0]**2 + (self.bounding_convex_hull - np.array(self.centroid))[:,1]**2))

	def load_env_convex(self, vertices_lst):
		"""take absolute vertices"""

		if len(self.objs) > 0:
			for obj in self.objs:
				for fix in obj.fixtures:
					obj.body.DestroyFixture(fix)
				self.world.DestroyBody(obj.body)
		self.objs = []

		for i in range(len(vertices_lst)):
			vertices = create_convex_hull(np.array(vertices_lst[i]))
			original_pos = np.array(compute_centroid(vertices))
			vertices = (vertices - original_pos)
			
			body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
			fixture = body.CreatePolygonFixture(density=1, vertices=vertices.tolist(), friction=0.5)
			self.objs.append(Polygon(body, [fixture], vertices, Colors[i]))

		self.bounding_convex_hull = create_convex_hull(np.concatenate([obj.vertices+obj.original_pos for obj in self.objs]))
		self.centroid = compute_centroid(self.bounding_convex_hull.tolist())
		self.bounding_circle_radius = math.sqrt(max((self.bounding_convex_hull - np.array(self.centroid))[:,0]**2 + (self.bounding_convex_hull - np.array(self.centroid))[:,1]**2))

	def load_env_concave(self, vertices_lst):
		"""take absolute vertices"""
		if len(self.objs) > 0:
			for obj in self.objs:
				for fix in obj.fixtures:
					obj.body.DestroyFixture(fix)
				self.world.DestroyBody(obj.body)
		self.objs = []

		for i in range(len(vertices_lst)):
			vertices = create_convex_hull(np.array(vertices_lst[i]))
			original_pos = np.array(compute_centroid(vertices))
			vertices = (vertices - original_pos)

			body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
			fixtures = []
			for j in range(len(vertices)):
				fixture = body.CreatePolygonFixture(density=1, vertices=vertices[np.array([j, (j+1) % len(vertices), (j+2) % len(vertices)])].tolist(), friction=0.5)
				fixtures.append(fixture)
			self.objs.append(Polygon(body, fixtures, vertices, Colors[i]))

		self.bounding_convex_hull = create_convex_hull(np.concatenate([obj.vertices+obj.original_pos for obj in self.objs]))
		self.centroid = compute_centroid(self.bounding_convex_hull.tolist())
		self.bounding_circle_radius = math.sqrt(max((self.bounding_convex_hull - np.array(self.centroid))[:,0]**2 + (self.bounding_convex_hull - np.array(self.centroid))[:,1]**2))

	
	def step(self, start_pt, end_pt, path, display=False):

		# display
		# if display:
		# 	# self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
		# 	# pygame.display.iconify()

		# 	self.screen.fill((255, 255, 255, 255))

		# 	def my_draw_polygon(polygon, body, fixture, color):
		# 		vertices = [(body.transform * v) * PPM for v in polygon.vertices]
		# 		vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
				
		# 		# pygame.draw.polygon(self.screen, color, vertices, 0)
		# 		# pygame.draw.polygon(self.screen, (0,0,0,0), vertices, 3)

		# 	polygonShape.draw = my_draw_polygon
		#



		start_pt = np.array(start_pt)
		end_pt = np.array(end_pt)

		self.rod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
		self.rodfix = self.rod.CreatePolygonFixture(vertices=[(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)])

		vector = normalize(end_pt - np.array(self.rod.position))
		self.rod.linearVelocity[0] = vector[0]
		self.rod.linearVelocity[1] = vector[1]
		self.rod.angularVelocity = 0.0

		timestamp = 0

		damping_factor = 1 - ((1-0.5) / 3)

		# display
		# if display:
		# 	for obj in self.objs:
		# 		for fix in obj.fixtures:
		# 			fix.shape.draw(obj.body, fix, obj.color)

		# 	self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))

			# pygame.image.save(self.screen, path+"start.png")
		#

		first_contact = -1

		while (timestamp < 100):

			if first_contact == -1:
				for i in range(len(self.objs)):
					if (self.objs[i].body.linearVelocity[0] ** 2 + self.objs[i].body.linearVelocity[1] ** 2 > 0.001):
						first_contact = i

			for obj in self.objs:
				obj.body.linearVelocity[0] = obj.body.linearVelocity[0] * damping_factor
				obj.body.linearVelocity[1] = obj.body.linearVelocity[1] * damping_factor
				obj.body.angularVelocity = obj.body.angularVelocity * damping_factor

			if (math.sqrt(np.sum((start_pt - np.array(self.rod.position))**2)) < 4):

				vector = normalize((end_pt+1e-8) - (start_pt+1e-8))
				self.rod.linearVelocity[0] = vector[0]
				self.rod.linearVelocity[1] = vector[1]

			else:
				self.rod.linearVelocity[0] = 0
				self.rod.linearVelocity[1] = 0

			self.world.Step(TIME_STEP, 10, 10)
			timestamp += 1

			# display
			# if display:
			# 	self.screen.fill((255, 255, 255, 255))

			# 	for obj in self.objs:
			# 		for fix in obj.fixtures:
			# 			fix.shape.draw(obj.body, fix, obj.color)

			# 	self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))

				# pygame.image.save(self.screen, path+str(timestamp)+".png")
			#

			
		# # display
		# if display:
		# 	# pygame.display.quit()
		# 	# pygame.quit()
		#

		return first_contact

	def step_area(self, start_pt, end_pt, gripper_width, path, display=False):

		# display
		# if display:
		# 	# self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
		# 	# pygame.display.iconify()

		# 	self.screen.fill((255, 255, 255, 255))

		# 	def my_draw_polygon(polygon, body, fixture, color):
		# 		vertices = [(body.transform * v) * PPM for v in polygon.vertices]
		# 		vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
				
		# 		# pygame.draw.polygon(self.screen, color, vertices, 0)
		# 		# pygame.draw.polygon(self.screen, (0,0,0,0), vertices, 3)

		# 	polygonShape.draw = my_draw_polygon
		#



		start_pt = np.array(start_pt)
		end_pt = np.array(end_pt)

		vertices_lst=[(0.1, gripper_width/2), (-0.1, gripper_width/2), (-0.1, -gripper_width/2), (0.1, -gripper_width/2)]

		vector = normalize(end_pt - start_pt)
		self.rod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
		self.rodfix = self.rod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])

		self.rod.linearVelocity[0] = vector[0]
		self.rod.linearVelocity[1] = vector[1]
		self.rod.angularVelocity = 0.0

		timestamp = 0

		damping_factor = 1 - ((1-0.5) / 3)

		# # display
		# if display:
		# 	for obj in self.objs:
		# 		for fix in obj.fixtures:
		# 			fix.shape.draw(obj.body, fix, obj.color)

		# 	self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))

			# pygame.image.save(self.screen, path+"start.png")
		#

		first_contact = -1

		while (timestamp < 100):

			if first_contact == -1:
				for i in range(len(self.objs)):
					if (self.objs[i].body.linearVelocity[0] ** 2 + self.objs[i].body.linearVelocity[1] ** 2 > 0.001):
						first_contact = i

			for obj in self.objs:
				obj.body.linearVelocity[0] = obj.body.linearVelocity[0] * damping_factor
				obj.body.linearVelocity[1] = obj.body.linearVelocity[1] * damping_factor
				obj.body.angularVelocity = obj.body.angularVelocity * damping_factor

			if (math.sqrt(np.sum((start_pt - np.array(self.rod.position))**2)) < 4):

				vector = normalize((end_pt+1e-8) - (start_pt+1e-8))
				self.rod.linearVelocity[0] = vector[0]
				self.rod.linearVelocity[1] = vector[1]

			else:
				self.rod.linearVelocity[0] = 0
				self.rod.linearVelocity[1] = 0

			self.world.Step(TIME_STEP, 10, 10)
			timestamp += 1

			# display
			# if display:
			# 	self.screen.fill((255, 255, 255, 255))

			# 	for obj in self.objs:
			# 		for fix in obj.fixtures:
			# 			fix.shape.draw(obj.body, fix, obj.color)

			# 	self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))

				# pygame.image.save(self.screen, path+str(timestamp)+".png")
			#

			
		# # display
		# if display:
		# 	# pygame.display.quit()
		# 	# pygame.quit()
		# #

		return first_contact

	def step_two_points(self, start_pt, end_pt, gripper_length, path, display=False):

		# display
		# if display:
		# 	# self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
		# 	# pygame.display.iconify()

		# 	self.screen.fill((255, 255, 255, 255))

		# 	def my_draw_polygon(polygon, body, fixture, color):
		# 		vertices = [(body.transform * v) * PPM for v in polygon.vertices]
		# 		vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
				
		# 		# pygame.draw.polygon(self.screen, color, vertices, 0)
		# 		# pygame.draw.polygon(self.screen, (0,0,0,0), vertices, 3)

		# 	polygonShape.draw = my_draw_polygon
		#



		start_pt = np.array(start_pt)
		end_pt = np.array(end_pt)

		vector = normalize(end_pt - start_pt)
		gripper1_vector = normalize((1, -(vector[0] / (vector[1]+1e-8))))
		gripper2_vector = normalize((-1, (vector[0] / (vector[1]+1e-8))))
		# print(np.array(gripper1_vector), gripper_length)
		gripper1_pt = start_pt + np.array(gripper1_vector) * gripper_length / 2
		gripper2_pt = start_pt + np.array(gripper2_vector) * gripper_length / 2

		self.rod = self.world.CreateKinematicBody(position=(gripper1_pt[0], gripper1_pt[1]), allowSleep=False)
		self.rodfix = self.rod.CreatePolygonFixture(vertices=[(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)])

		self.rod2 = self.world.CreateKinematicBody(position=(gripper2_pt[0], gripper2_pt[1]), allowSleep=False)
		self.rodfix2 = self.rod2.CreatePolygonFixture(vertices=[(0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1)])

		self.rod.linearVelocity[0] = vector[0]
		self.rod.linearVelocity[1] = vector[1]
		self.rod.angularVelocity = 0.0

		self.rod2.linearVelocity[0] = vector[0]
		self.rod2.linearVelocity[1] = vector[1]
		self.rod2.angularVelocity = 0.0

		timestamp = 0

		damping_factor = 1 - ((1-0.5) / 3)

		# display
		# if display:
		# 	for obj in self.objs:
		# 		for fix in obj.fixtures:
		# 			fix.shape.draw(obj.body, fix, obj.color)

		# 	self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))
		# 	self.rodfix2.shape.draw(self.rod2, self.rodfix2, (0, 0, 0, 255))

			# pygame.image.save(self.screen, path+"start.png")
		#

		first_contact = -1

		while (timestamp < 100):

			if first_contact == -1:
				for i in range(len(self.objs)):
					if (self.objs[i].body.linearVelocity[0] ** 2 + self.objs[i].body.linearVelocity[1] ** 2 > 0.001):
						first_contact = i

			for obj in self.objs:
				obj.body.linearVelocity[0] = obj.body.linearVelocity[0] * damping_factor
				obj.body.linearVelocity[1] = obj.body.linearVelocity[1] * damping_factor
				obj.body.angularVelocity = obj.body.angularVelocity * damping_factor

			if (math.sqrt(np.sum((start_pt - np.array(self.rod.position))**2)) < 4):

				vector = normalize((end_pt+1e-8) - (start_pt+1e-8))
				self.rod.linearVelocity[0] = vector[0]
				self.rod.linearVelocity[1] = vector[1]
				self.rod2.linearVelocity[0] = vector[0]
				self.rod2.linearVelocity[1] = vector[1]

			else:
				self.rod.linearVelocity[0] = 0
				self.rod.linearVelocity[1] = 0
				self.rod2.linearVelocity[0] = 0
				self.rod2.linearVelocity[1] = 0

			self.world.Step(TIME_STEP, 10, 10)
			timestamp += 1

			# display
		# 	if display:
		# 		self.screen.fill((255, 255, 255, 255))

		# 		for obj in self.objs:
		# 			for fix in obj.fixtures:
		# 				fix.shape.draw(obj.body, fix, obj.color)

		# 		self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))
		# 		self.rodfix2.shape.draw(self.rod2, self.rodfix2, (0, 0, 0, 255))

		# 		# pygame.image.save(self.screen, path+str(timestamp)+".png")
		# 	#

			
		# # display
		# if display:
		# 	# pygame.display.quit()
		# 	# pygame.quit()
		# #

		return first_contact

	def avg_centroid(self):
		total_separation = 0
		for i in range(len(self.objs) - 1):
			for j in range(i+1, len(self.objs)):
				if i != j:
					total_separation += math.log(euclidean_dist(self.objs[i].body.position, self.objs[j].body.position))
		return total_separation * 2/(len(self.objs) * (len(self.objs) - 1))

	def avg_geometry(self):
		total_separation = 0.0
		for i in range(len(self.objs) - 1):
			for j in range(i+1, len(self.objs)):
				if i != j:
					shape1 = self.objs[i].fixtures[0].shape
					shape2 = self.objs[j].fixtures[0].shape
					transform1 = Box2D.b2Transform()
					pos1 = self.objs[i].body.position
					angle1 = self.objs[i].body.angle
					transform1.Set(pos1, angle1)
					transform2 = Box2D.b2Transform()
					pos2 = self.objs[j].body.position
					angle2 = self.objs[j].body.angle
					transform2.Set(pos2, angle2)
					pointA, pointB, distance, iterations = Box2D.b2Distance(shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
					# print(pointA, pointB, distance, iterations)
					total_separation += distance
		return total_separation * 2/(len(self.objs) * (len(self.objs) - 1))

	def min_geometry(self):
		min_dist = 1e2
		for i in range(len(self.objs) - 1):
			for j in range(i+1, len(self.objs)):
				if i != j:
					shape1 = self.objs[i].fixtures[0].shape
					shape2 = self.objs[j].fixtures[0].shape
					transform1 = Box2D.b2Transform()
					pos1 = self.objs[i].body.position
					angle1 = self.objs[i].body.angle
					transform1.Set(pos1, angle1)
					transform2 = Box2D.b2Transform()
					pos2 = self.objs[j].body.position
					angle2 = self.objs[j].body.angle
					transform2.Set(pos2, angle2)
					pointA, pointB, distance, iterations = Box2D.b2Distance(shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
					if distance < min_dist:
						min_dist = distance
		return min_dist

	def min_centroid(self):
		min_dist = 1e2
		for i in range(len(self.objs) - 1):
			for j in range(i+1, len(self.objs)):
				if i != j:
					if min_dist > (euclidean_dist(self.objs[i].body.position, self.objs[j].body.position)):
						min_dist = euclidean_dist(self.objs[i].body.position, self.objs[j].body.position)
		return min_dist

	def count_threshold(self, threshold=0.3):
		count = 0
		for i in range(len(self.objs)):
			min_dist = 1e2
			for j in range(len(self.objs)):
				if i != j:
					shape1 = self.objs[i].fixtures[0].shape
					shape2 = self.objs[j].fixtures[0].shape
					transform1 = Box2D.b2Transform()
					pos1 = self.objs[i].body.position
					angle1 = self.objs[i].body.angle
					transform1.Set(pos1, angle1)
					transform2 = Box2D.b2Transform()
					pos2 = self.objs[j].body.position
					angle2 = self.objs[j].body.angle
					transform2.Set(pos2, angle2)
					pointA, pointB, distance, iterations = Box2D.b2Distance(shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
					if distance < min_dist:
						min_dist = distance
			if min_dist >= threshold:
				count += 1
		return count

	def count_soft_threshold(self, threshold_max=0.3, threshold_min=0.1):
		count = 0.0
		for i in range(len(self.objs)):
			min_dist = 1e2
			for j in range(len(self.objs)):
				if i != j:
					shape1 = self.objs[i].fixtures[0].shape
					shape2 = self.objs[j].fixtures[0].shape
					transform1 = Box2D.b2Transform()
					pos1 = self.objs[i].body.position
					angle1 = self.objs[i].body.angle
					transform1.Set(pos1, angle1)
					transform2 = Box2D.b2Transform()
					pos2 = self.objs[j].body.position
					angle2 = self.objs[j].body.angle
					transform2.Set(pos2, angle2)
					pointA, pointB, distance, iterations = Box2D.b2Distance(shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
					if distance < min_dist:
						min_dist = distance
			if min_dist >= threshold_max:
				count += 1
			elif min_dist > threshold_min and min_dist < threshold_max:
				count += (min_dist-0.1) * 5				
		return count

	def collect_data_summary(self, start_pt, end_pt, img_path, sum_path=None):
		summary = {}
		abs_start_pt = np.array(start_pt)
		abs_end_pt = np.array(end_pt)
		summary["start pt"] = abs_start_pt.tolist()
		summary["end pt"] = abs_end_pt.tolist()
		
		for i in range(len(self.objs)):
			summary[str(i)+" dist to pushing line"] = pointToLineDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" original pos"] = np.array(self.objs[i].body.position).tolist()
			# print(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" project dist"] = projectedPtToStartDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" vertices"] = np.array(self.objs[i].vertices).tolist()
			summary[str(i)+" disk coverage"] = self.objs[i].disk_coverage

		summary["avg centroid before push"] = self.avg_centroid()
		summary["avg geometry before push"] = self.avg_geometry()
		summary["min centroid before push"] = self.min_centroid()
		summary["min geometry before push"] = self.min_geometry()
		summary["count threshold before push"] = self.count_threshold()
		summary["count soft threshold before push"] = self.count_soft_threshold()


		first_contact = self.step(start_pt, end_pt, img_path)

		# if first_contact == -1:
		# 	return

		for i in range(len(self.objs)):
			summary[str(i)+" change of pos"] = euclidean_dist(self.objs[i].body.position, self.objs[i].original_pos)
		summary["avg centroid after push"] = self.avg_centroid()
		summary["avg geometry after push"] = self.avg_geometry()
		summary["min centroid after push"] = self.min_centroid()
		summary["min geometry after push"] = self.min_geometry()
		summary["count threshold after push"] = self.count_threshold()
		summary["count soft threshold after push"] = self.count_soft_threshold()
		summary["first contact object"] = first_contact
		
		if sum_path is not None:
			with open(sum_path+'summary.json', 'w') as f:
				json.dump(summary, f)
		
		return summary

	def prune_best_summary(self, prune_method, sum_path, metric="avg centroid"):
		pt_lst = prune_method(self)
		best_pt = None
		# if metric == "avg centroid" or metric == "avg geometry":
		best_sep = -1e2
		for pts in pt_lst:
			self.reset()
			summary = self.collect_data_summary(pts[0], pts[1], "/")
			if summary[metric +" after push"] - summary[metric + " before push"] >= best_sep:
				best_pt = pts
				self.reset()
				best_sep = summary[metric +" after push"] - summary[metric + " before push"]
		
			# best_sep = 1e2
			# for pts in pt_lst:
			# 	self.reset()
			# 	summary = self.collect_data_summary(pts[0], pts[1], "/")
			# 	if summary[metric +" after push"] - summary[metric + " before push"] < best_sep:
			# 		best_pt = pts
			# 		self.reset()
			# 		best_sep = summary[metric +" after push"] - summary[metric + " before push"]

		if best_pt is not None:
			self.reset()
			return self.collect_data_summary(best_pt[0], best_pt[1], "/", sum_path=sum_path)


	def prune_best_summary_all(self, prune_method, folder_path, ind_num, metrics=["avg centroid"]):
		pt_lst = prune_method(self)
		best_pt = []
		# if metric == "avg centroid" or metric == "avg geometry":
		best_sep = []
		for i in range(len(metrics)):
			best_sep.append(-1e2)
			best_pt.append(None) 
		for pts in pt_lst:
			self.reset()
			# print(pts)
			summary = self.collect_data_summary(pts[0], pts[1], "/")
			for i in range(len(metrics)):
				
				if summary[metrics[i] +" after push"] - summary[metrics[i] + " before push"] >= best_sep[i]:
					best_pt[i] = pts
					self.reset()
					best_sep[i] = summary[metrics[i] +" after push"] - summary[metrics[i] + " before push"]
		
			# best_sep = 1e2
			# for pts in pt_lst:
			# 	self.reset()
			# 	summary = self.collect_data_summary(pts[0], pts[1], "/")
			# 	if summary[metric +" after push"] - summary[metric + " before push"] < best_sep:
			# 		best_pt = pts
			# 		self.reset()
			# 		best_sep = summary[metric +" after push"] - summary[metric + " before push"]

		best_sum = []

		for i in range(len(metrics)):
			if best_pt is not None:
				self.reset()
				best_sum.append(self.collect_data_summary(best_pt[i][0], best_pt[i][1], "/", sum_path=folder_path+"/"+metrics[i]+"/"+ind_num))
		# print(best_sep)
		return best_sum

	def collect_data_area_summary(self, start_pt, end_pt, gripper_width, img_path, sum_path):
		summary = {}
		abs_start_pt = np.array(start_pt)
		abs_end_pt = np.array(end_pt)
		summary["start pt"] = abs_start_pt.tolist()
		summary["end pt"] = abs_end_pt.tolist()
		summary["gripper_width"] = gripper_width
		
		for i in range(len(self.objs)):
			summary[str(i)+" dist to pushing line"] = pointToLineDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" original pos"] = np.array(self.objs[i].body.position).tolist()
			summary[str(i)+" project dist"] = projectedPtToStartDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" vertices"] = np.array(self.objs[i].vertices).tolist()

		summary["mean separation before push"] = self.mean_object_separation()

		first_contact = self.step_area(start_pt, end_pt, gripper_width, img_path)

		# if first_contact == -1:
		# 	return

		for i in range(len(self.objs)):
			summary[str(i)+" change of pos"] = euclidean_dist(self.objs[i].body.position, self.objs[i].original_pos)
		summary["mean separation after push"] = self.mean_object_separation()
		summary["first contact object"] = first_contact
		
		with open(sum_path+'summary.json', 'w') as f:
			json.dump(summary, f)
		
		return summary

	def collect_data_two_points_summary(self, start_pt, end_pt, gripper_length, img_path, sum_path):
		summary = {}
		abs_start_pt = np.array(start_pt)
		abs_end_pt = np.array(end_pt)
		summary["start pt"] = abs_start_pt.tolist()
		summary["end pt"] = abs_end_pt.tolist()
		summary["gripper_length"] = gripper_length
		
		for i in range(len(self.objs)):
			summary[str(i)+" dist to pushing line"] = pointToLineDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" original pos"] = np.array(self.objs[i].body.position).tolist()
			summary[str(i)+" project dist"] = projectedPtToStartDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" vertices"] = np.array(self.objs[i].vertices).tolist()

		summary["mean separation before push"] = self.mean_object_separation()

		first_contact = self.step_two_points(start_pt, end_pt, gripper_length, img_path)

		# if first_contact == -1:
		# 	return

		for i in range(len(self.objs)):
			summary[str(i)+" change of pos"] = euclidean_dist(self.objs[i].body.position, self.objs[i].original_pos)
		summary["mean separation after push"] = self.mean_object_separation()
		summary["first contact object"] = first_contact
		
		with open(sum_path+'summary.json', 'w') as f:
			json.dump(summary, f)
		
		return summary

	def reset(self):
		if self.rod:
			self.rod.DestroyFixture(self.rodfix)
			self.world.DestroyBody(self.rod)
			self.rod = None

		if self.rod2:
			self.rod2.DestroyFixture(self.rodfix2)
			self.world.DestroyBody(self.rod2)
			self.rod2 = None

		for obj in self.objs:
			obj.body.position[0] = obj.original_pos[0]
			obj.body.position[1] = obj.original_pos[1]
			obj.body.angle = 0.0
			obj.body.linearVelocity[0] = 0.0
			obj.body.linearVelocity[1] = 0.0
			obj.body.angularVelocity = 0.0

	# def visualize(self, path):
	# 	# self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
	# 	# pygame.display.iconify()

	# 	self.screen.fill((0, 0, 0, 0))

	# 	def my_draw_polygon(polygon, body, fixture, color):
	# 		vertices = [(body.transform * v) * PPM for v in polygon.vertices]
	# 		vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			
	# 		# pygame.draw.polygon(self.screen, color, vertices, 0)

	# 	polygonShape.draw = my_draw_polygon

	# 	for obj in self.objs:
	# 		obj.fixture.shape.draw(obj.body, obj.fixture, obj.color)

	# 	self.rodfix.shape.draw(self.rod, self.rodfix, (100, 100, 100, 255))

	# 	# pygame.image.save(self.screen, path+"debug.png")

		# pygame.display.quit()
		# pygame.quit()

	def load_env(self, dic):
		assert (len(dic) - 15) % 6 == 0
		num_obj = (len(dic) - 15) // 6
		for i in range(num_obj):
			original_pos = np.array(dic[str(i)+" original pos"])
			vertices = np.array(dic[str(i)+" vertices"])
			body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
			fixture = body.CreatePolygonFixture(density=1, vertices=vertices.tolist(), friction=0.5)
			self.objs.append(Polygon(body, [fixture], vertices, Colors[i%len(Colors)]))

	def save_curr_position(self):
		position = {}
		for i in range(len(self.objs)):
			position[i] = [self.objs[i].body.position[0], self.objs[i].body.position[1], self.objs[i].body.angle]
		return position

	def load_position(self, position):
		if self.rod:
			self.rod.DestroyFixture(self.rodfix)
			self.world.DestroyBody(self.rod)
			self.rod = None

		if self.rod2:
			self.rod2.DestroyFixture(self.rodfix2)
			self.world.DestroyBody(self.rod2)
			self.rod2 = None

		for i in range(len(self.objs)):
			self.objs[i].body.position[0] = position[i][0]
			self.objs[i].body.position[1] = position[i][1]
			self.objs[i].body.angle = position[i][2]
			self.objs[i].body.linearVelocity[0] = 0.0
			self.objs[i].body.linearVelocity[1] = 0.0
			self.objs[i].body.angularVelocity = 0.0

	def sequential_planning_helper(self, prune_method, step, position, history, metric="count threshold"):
		if step == 0:
			return history[-1][metric+" after push"]-history[0][metric+" before push"], history
		else:
			pt_lst = prune_method(self)
			best_sum = None
			best_metric = 1e2
			for pts in pt_lst:
				self.load_position(position)
				curr_hist = self.collect_data_summary(pts[0], pts[1], None)
				hist_sum = copy.deepcopy(history)
				hist_sum.append(curr_hist)
				curr_pos = self.save_curr_position()
				value, total = self.sequential_planning_helper(prune_method, step-1, curr_pos, hist_sum, metric)
				if value is not None and value > best_metric:
					best_sum = total
					best_metric = total[-1][metric+" after push"]-total[0][metric+" before push"]
			if best_sum is not None:
				print(best_sum[-1][metric +" after push"], best_sum[-1][metric + " before push"])
				return best_metric, best_sum
			else:
				return None, None

	def sequential_tree_prune_planning(self, prune_method, max_step=10, metric="count threshold", sum_path=None):
		curr_pos = self.save_curr_position()
		best_metric, best_sum = self.sequential_planning_helper(prune_method, max_step, curr_pos, [], metric=metric)

		# for i in range(max_step):
		# 	curr_pos = self.save_curr_position()
		# 	best_pts = self.prune_best(prune_method, metric, curr_pos)
		# 	self.load_position(curr_pos)
		# 	curr_sum = self.collect_data_summary(best_pts[0], best_pts[1], None)
		# 	data_sum.append(curr_sum)
		# 	print(curr_sum[metric +" after push"], curr_sum[metric + " before push"])
		# return data_sum[-1][metric +" after push"], data_sum[-1][metric + " before push"]
		if sum_path is not None and best_sum is not None:
			print(best_metric)
			with open(sum_path+'.json', 'w') as f:
				json.dump(best_sum, f)
		return best_sum

	def sequential_prune_planning(self, prune_method, max_step=5, metric="count threshold", sum_path=None):
		data_sum = []
		for i in range(max_step):
			curr_pos = self.save_curr_position()
			best_pts = self.prune_best(prune_method, metric, curr_pos)
			self.load_position(curr_pos)
			curr_sum = self.collect_data_summary(best_pts[0], best_pts[1], None)
			data_sum.append(curr_sum)
			print(curr_sum[metric +" after push"], curr_sum[metric + " before push"])
		# return data_sum[-1][metric +" after push"], data_sum[-1][metric + " before push"]
		if sum_path is not None:
			with open(sum_path+'.json', 'w') as f:
				json.dump(data_sum, f)
		return data_sum

	def sequential_policy_planning(self, policy, max_step=5, metric="count threshold", sum_path=None):
		data_sum = []
		for i in range(max_step):
			curr_pos = self.save_curr_position()
			best_pts = policy(self)
			self.load_position(curr_pos)
			curr_sum = self.collect_data_summary(best_pts[0], best_pts[1], None)
			data_sum.append(curr_sum)
			print(curr_sum[metric +" after push"], curr_sum[metric + " before push"])
		# return data_sum[-1][metric +" after push"], data_sum[-1][metric + " before push"]
		if sum_path is not None:
			with open(sum_path+'.json', 'w') as f:
				json.dump(data_sum, f)
		return data_sum

	def prune_best(self, prune_method, metric="count soft threshold", position=None, sum_path=None):
		pt_lst = prune_method(self)
		best_pt = None
		# if metric == "avg centroid" or metric == "avg geometry":
		best_sep = -1e2
		for pts in pt_lst:
			if position is None:
				self.reset()
			else:
				self.load_position(position)
			summary = self.collect_data_summary(pts[0], pts[1], None)
			if summary[metric +" after push"] - summary[metric + " before push"] >= best_sep:
				best_pt = pts
				if position is None:
					self.reset()
				else:
					self.load_position(position)
				best_sep = summary[metric +" after push"] - summary[metric + " before push"]
		
		# if best_pt is not None:
		# 	self.reset()
			# return self.collect_data_summary(best_pt[0], best_pt[1], "/", sum_path=sum_path)
		return best_pt

	def count_threshold_data_generation(self, max_step=4):
		input_data = []
		item_label = []
		action_label = []
		# data_sum = []
		for i in range(max_step):
			curr_pos = self.save_curr_position()
			best_pts = self.prune_best(no_prune, "count soft threshold", curr_pos)
			if best_pts is None:
				break
			best_summary = self.collect_data_summary(best_pts[0], best_pts[1], None)
			best_dist = best_summary["count soft threshold after push"] - best_summary["count soft threshold before push"]
			if best_dist <= 0:
				break
			print(best_dist, best_summary["count threshold after push"] - best_summary["count threshold before push"])

			# if metric == "avg centroid" or metric == "avg geometry":
			
			self.load_position(curr_pos)
			
			summary = self.collect_data_summary(best_pts[0], best_pts[1], None)
			
			if summary["count soft threshold after push"] - summary["count soft threshold before push"] >= best_dist:
				# save: centroids, vertices, pts
				# self.load_position(curr_pos)
				state = []
				action = []
				for o in self.objs:
					state.append(o.body.position[0])
					state.append(o.body.position[1])
					state.append(o.body.angle)
					for v in o.vertices:
						state.append(v[0])
						state.append(v[1])
				input_data.append(state)
				item_label.append(summary["first contact object"])

				if summary["first contact object"] == -1: 
					continue

				item_pos = curr_pos[summary["first contact object"]]
				start_pt = best_pts[0] - np.array([item_pos[0], item_pos[1]])

				vec = normalize(np.array(best_pts[1]) - np.array(best_pts[0]))
				action.append(best_pts[0][0])
				action.append(best_pts[0][1])
				action.append(vec[0])
				action.append(vec[1])
				action_label.append(action)
				# print(state)
				# print(output)
		
			
			# self.load_position(curr_pos)
			# best_summary = self.collect_data_summary(best_pts[0], best_pts[1], None)
			# data_sum.append(curr_sum)
			# print(curr_sum["count threshold after push"], curr_sum["count threshold before push"])
		# return data_sum[-1][metric +" after push"], data_sum[-1][metric + " before push"]
		
		return input_data, item_label, action_label

if __name__ == "__main__":
	path = "/nfs/diskstation/zdong/singulation_bc/"
	# path = ""
	# state_file = []
	# label_file = []
	for i in range(0, 10):
		print(i)
		state_file = []
		label_file = []
		item_label = []
		while len(state_file) < 1024:
			print(len(state_file))
			test = SingulationEnv()
			while True:
				try:
					test.create_random_env(7, 4)
					break
				except:
					print("retry")

			state, items, label = test.count_threshold_data_generation()
			state_file.extend(state)
			item_label.extend(items)
			label_file.extend(label)
		# state_file_bad.extend(state_bad)
		# label_file_bad.extend(label_bad)
		# print(len(state_bad))
			np.save(path+str(i).zfill(3)+"in.npy", np.array(state_file))
			np.save(path+str(i).zfill(3)+"item.npy", np.array(item_label))
			np.save(path+str(i).zfill(3)+"out.npy", np.array(label_file))

	# bc = []
	# cp2d = []
	# es = []

	# model = BC()
	# model.load_model()
	# for i in range(0, 100):
	# 	print(i)
	# 	test = SingulationEnv()
	# 	while True:
	# 		try:
	# 			test.create_random_env(7, 4)
	# 			break
	# 		except:
	# 			print("retry")
	# 	test.reset()
	# 	state = []
	# 	for o in test.objs:
	# 		state.append(o.body.position[0])
	# 		state.append(o.body.position[1])
	# 		state.append(o.body.angle)
	# 		for v in o.vertices:
	# 			state.append(v[0])
	# 			state.append(v[1])
	# 	# input_data.append(state)
	# 	raw_output = model.predict([state])[0][0]
	# 	if np.any(raw_output == np.nan):
	# 		continue

	# 	pt1 = np.array([raw_output[0], raw_output[1]])
	# 	pt2 = pt1 + np.array(normalize([raw_output[2], raw_output[3]]))
	# 	print(pt1)
	# 	print(pt2)
	# 	summary = test.collect_data_summary(pt1.tolist(), pt2.tolist(), None)
	# 	bc.append(summary["count threshold after push"] - summary["count threshold before push"])

	# 	test.reset()
	# 	best_pts = test.prune_best(no_prune, "count threshold", curr_pos)
	# 	summary = test.collect_data_summary(best_pts[0], best_pts[1], None)
	# 	es.append(summary["count threshold after push"] - summary["count threshold before push"])

	# 	test.reset()
	# 	pts = proposed9_sequential(test)
	# 	summary = test.collect_data_summary(pts[0], pts[1], None)
	# 	cp2d.append(summary["count threshold after push"] - summary["count threshold before push"])

	# print(np.mean(bc), np.mean(es), np.mean(cp2d))



	# metrics = ["min geometry", "count threshold", "avg geometry", "avg centroid", "min centroid"]

	# for num_obj in range(6, 16):
	# 	if not os.path.exists("prune/"+str(num_obj)):
	# 		os.makedirs("prune/"+str(num_obj))
	# 	# for g in range(len(GROUPS)):
	# 	for g in range(len(GROUPS)):
	# 		# print(num_obj, g)

	# 		if not os.path.exists("prune/"+str(num_obj)+"/"+str(g)):
	# 			os.makedirs("prune/"+str(num_obj)+"/"+str(g))
	# 		for m in metrics:
	# 			if not os.path.exists("prune/"+str(num_obj)+"/"+str(g)+"/"+m):
	# 				os.makedirs("prune/"+str(num_obj)+"/"+str(g)+"/"+m)
	# 		for i in range(100):
	# 			print(num_obj, g, i)
	# 			test = SingulationEnv()
	# 			# with open("prune/"+str(num_obj)+"/"+str(g)+"/avg centroid/"+str(i).zfill(2)+"no_prunesummary.json") as json_data:
	# 			# 	dic = json.load(json_data)
	# 			# test.load_env(dic)
	# 			while True:
	# 				try:
	# 					test.create_random_env(num_obj, g)
	# 					break
	# 				except:
	# 					print("retry")

	# 			test.reset()
	# 			dic = test.prune_best_summary_all(no_prune, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"no_prune", metrics=metrics)[1]

	# 			test.reset()
	# 			dic_all = test.prune_best_summary_all(com_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"com_only", metrics=metrics)
	# 			print("com_only: " + str((dic_all[1]["count threshold after push"] - dic_all[3]["count threshold before push"]) / (dic["count threshold after push"] - dic["count threshold before push"])))

	# 			test.reset()
	# 			dic_all = test.prune_best_summary_all(pairwise_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"pairwise_only", metrics=metrics)
	# 			print("pairwise_only: " + str((dic_all[1]["count threshold after push"] - dic_all[3]["count threshold before push"]) / (dic["count threshold after push"] - dic["count threshold before push"])))
				
	# 			test.reset()
	# 			dic_all = test.prune_best_summary_all(cluster_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"cluster_only", metrics=metrics)
	# 			print("cluster_only: " + str((dic_all[1]["count threshold after push"] - dic_all[3]["count threshold before push"]) / (dic["count threshold after push"] - dic["count threshold before push"])))

	# 			test.reset()
	# 			dic_all = test.prune_best_summary_all(fcc_cluster_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"fcc_cluster_only", metrics=metrics)
	# 			print("fcc_cluster_only: " + str((dic_all[1]["count threshold after push"] - dic_all[3]["count threshold before push"]) / (dic["count threshold after push"] - dic["count threshold before push"])))

				# test.reset()
				# dic_all = test.prune_best_summary_all(cluster_pairwise_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"cluster_pairwise_only", metrics=metrics)
				# print("cluster_pairwise_only: " + str((dic_all[3]["avg centroid after push"] - dic_all[3]["avg centroid before push"]) / (dic["avg centroid after push"] - dic["avg centroid before push"])))

				# test.reset()
				# dic_all = test.prune_best_summary_all(minpair_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"minpair_only", metrics=metrics)
				# print("minpair_only: " + str((dic_all[1]["count threshold after push"] - dic_all[3]["count threshold before push"]) / (dic["count threshold after push"] - dic["count threshold before push"])))

				# test.reset()
				# dic_all = test.prune_best_summary_all(minpair_pairwise_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"minpair_pairwise_only", metrics=metrics)
				# print("minpair_pairwise_only: " + str((dic_all[3]["avg centroid after push"] - dic_all[3]["avg centroid before push"]) / (dic["avg centroid after push"] - dic["avg centroid before push"])))

				# test.reset()
				# dic_all = test.prune_best_summary_all(center_minpair_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"center_minpair_only", metrics=metrics)
				# print("center_minpair_only: " + str((dic_all[3]["avg centroid after push"] - dic_all[3]["avg centroid before push"]) / (dic["avg centroid after push"] - dic["avg centroid before push"])))

				# test.reset()
				# dic_all = test.prune_best_summary_all(center_minpair_two_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"center_minpair_two_only", metrics=metrics)
				# print("center_minpair_two_only: " + str((dic_all[3]["avg centroid after push"] - dic_all[3]["avg centroid before push"]) / (dic["avg centroid after push"] - dic["avg centroid before push"])))

	# for num_obj in range(7, 8):
	# 	# if not os.path.exists("sequence/"+str(num_obj)):
	# 	# 	os.makedirs("sequence/"+str(num_obj))
	# 	for g in range(4):
	# 		if not os.path.exists("sequence/"+str(num_obj)+"/"+str(g)+"/tree_search_center_minpair/"):
	# 			os.makedirs("sequence/"+str(num_obj)+"/"+str(g)+"/tree_search_center_minpair/")
	# 		for i in range(100):
	# 			print(num_obj, g, i)
	# 			test = SingulationEnv()
	# 			with open("sequence/"+str(num_obj)+"/"+str(g)+"/no_prune/"+str(i)+".json") as json_data:
	# 				dic = json.load(json_data)
	# 			test.load_env(dic[0])
	# 			test.reset()
	# 			test.sequential_tree_prune_planning(center_minpair_only, max_step=4, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/tree_search_center_minpair/"+str(i))

	# method = ["com_only", "no_prune", "cluster_only", "fcc_cluster_only", "Cluster2DSeq", "center_minpair_only", "minpair_only", "center_minpair_multistage", "center_minpair_finalstage", "center_minpair_midstage"]
	# for num_obj in range(9, 10):
	# 	if not os.path.exists("sequence/"+str(num_obj)):
	# 		os.makedirs("sequence/"+str(num_obj))
	# 	for g in range(4):
	# 		if not os.path.exists("sequence/"+str(num_obj)+"/"+str(g)):
	# 			os.makedirs("sequence/"+str(num_obj)+"/"+str(g))
	# 		for m in method:
	# 			if not os.path.exists("sequence/"+str(num_obj)+"/"+str(g)+"/"+m):
	# 				os.makedirs("sequence/"+str(num_obj)+"/"+str(g)+"/"+m)
	# 		for i in range(100):
	# 			print(num_obj, g, i)
	# 			test = SingulationEnv()
	# 			with open("sequence/"+str(num_obj)+"/"+str(g)+"/no_prune/"+str(i)+".json") as json_data:
	# 				dic = json.load(json_data)
	# 			test.load_env(dic[0])
	# 			# while True:
	# 			# 	try:
	# 			# 		test.create_random_env(num_obj, g)
	# 			# 		break
	# 			# 	except:
	# 			# 		print("retry")

	# 			# test.reset()
	# 			# test.sequential_prune_planning(no_prune, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/no_prune/"+str(i))
	# 			# test.reset()
	# 			# test.sequential_prune_planning(com_only, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/com_only/"+str(i))
	# 			# test.reset()
	# 			# test.sequential_prune_planning(cluster_only, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/cluster_only/"+str(i))
	# 			# test.reset()
	# 			# test.sequential_prune_planning(fcc_cluster_only, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/fcc_cluster_only/"+str(i))
	# 			# test.reset()
	# 			# test.sequential_prune_planning(minpair_only, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/minpair_only/"+str(i))
	# 			# test.reset()
	# 			# # test.sequential_prune_planning(fcc_cluster_center_only, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/fcc_cluster_center_only/"+str(i))
	# 			# # test.reset()
	# 			# test.sequential_prune_planning(center_minpair_only, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/center_minpair_only/"+str(i))
	# 			# test.reset()
	# 			# test.sequential_policy_planning(proposed9_sequential, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/Cluster2DSeq/"+str(i))
	# 			# test.reset()

	# 			test.reset()
	# 			test.sequential_prune_planning(center_minpair_multistage, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/center_minpair_multistage/"+str(i))
	# 			test.reset()
	# 			test.sequential_prune_planning(center_minpair_finalstage, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/center_minpair_finalstage/"+str(i))
	# 			test.reset()
	# 			test.sequential_prune_planning(center_minpair_midstage, max_step=num_obj, sum_path="sequence/"+str(num_obj)+"/"+str(g)+"/center_minpair_midstage/"+str(i))
	# 			test.reset()

	