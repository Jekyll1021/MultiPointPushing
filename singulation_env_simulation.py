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

PPM = 60.0  # pixels per meter
TIME_STEP = 0.1
SCREEN_WIDTH, SCREEN_HEIGHT = 720, 720

GROUPS = [(1, 1, 15), (0.75, 1, 12), (0.5, 1, 9), (0.25, 1, 6)]


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

	def test_overlap(self, other_polygon):
		"""test if the current polygon overlaps with a polygon of another polygon 
		with other_centroid as centroid and other_vertices as vertices(all in numpy array)"""
		if self.dist(other_polygon) > 0:
			return False
		return True

	def dist(self, other_polygon):
		"""do not work on a polygon and a rod"""
		shape1 = self.fixtures[0].shape
		shape2 = other_polygon.fixtures[0].shape
		transform1 = Box2D.b2Transform()
		pos1 = self.body.position
		angle1 = self.body.angle
		transform1.Set(pos1, angle1)
		transform2 = Box2D.b2Transform()
		pos2 = other_polygon.body.position
		angle2 = other_polygon.body.angle
		transform2.Set(pos2, angle2)
		pointA, pointB, distance, iterations = Box2D.b2Distance(shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
		return distance

	def dist_rod(self, rod_fix, rod_body):
		shape1 = self.fixtures[0].shape
		shape2 = rod_fix.shape
		transform1 = Box2D.b2Transform()
		pos1 = self.body.position
		angle1 = self.body.angle
		transform1.Set(pos1, angle1)
		transform2 = Box2D.b2Transform()
		pos2 = rod_body.position
		angle2 = rod_body.angle
		transform2.Set(pos2, angle2)
		pointA, pointB, distance, iterations = Box2D.b2Distance(shapeA=shape1, shapeB=shape2, transformA=transform1, transformB=transform2)
		# print(distance)
		return distance

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
					original_pos = np.array([np.random.uniform(-1.8, 1.8),np.random.uniform(-1.8, 1.8)]) + np.array(self.objs[-1].body.position)
					# original_pos = np.array([np.random.uniform(1,11),np.random.uniform(1,11)])
					no_overlap = True
					original_pos = np.clip(original_pos, 1, 11)

					body = self.world.CreateDynamicBody(position=original_pos.tolist(), allowSleep=False)
					fixture = body.CreatePolygonFixture(density=1, vertices=vertices.tolist(), friction=0.5)
					# curr_polygon = Polygon(body, fixture, vertices, Colors[i])
					curr_polygon = Polygon(body, [fixture], vertices, Colors[i % len(Colors)])
					for obj in self.objs:
						if obj.test_overlap(curr_polygon):
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

	
	def step(self, start_pt, end_pt, path, display=False, check_reachable=True):

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

		vector = np.array(normalize(end_pt - np.array(self.rod.position)))

		# reachability check
		if check_reachable:
			vertices_lst=[(0.0, 0.1), (0.0, -0.1), (-0.3, -0.1), (-0.3, 0.1)]

			testrod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
			testrodfix = testrod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])

			while (np.count_nonzero(np.array([o.dist_rod(testrodfix, testrod) for o in self.objs]) <= 0) > 0):
				# print(start_pt, [o.dist_rod(self.rodfix, self.rod) for o in self.objs])
				start_pt -= 0.1 * vector
				testrod.DestroyFixture(testrodfix)
				self.world.DestroyBody(testrod)

				testrod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
				testrodfix = testrod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])

			testrod.DestroyFixture(testrodfix)
			self.world.DestroyBody(testrod)

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
				break

			self.world.Step(TIME_STEP, 8, 3)
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

	def step_area(self, start_pt, end_pt, gripper_width, path, display=False, check_reachable=True):

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

		vector = np.array(normalize(end_pt - start_pt))
		self.rod = self.world.CreateKinematicBody(position=(start_pt[0], start_pt[1]), allowSleep=False)
		self.rodfix = self.rod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])

		# reachability check
		if check_reachable:
			while (np.count_nonzero(np.array([o.dist_rod(self.rodfix, self.rod) for o in self.objs]) <= 0) > 0):
				# print(start_pt, [o.dist_rod(self.rodfix, self.rod) for o in self.objs])
				start_pt -= 0.1 * vector
				self.rod.DestroyFixture(self.rodfix)
				self.world.DestroyBody(self.rod)
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
				break

			self.world.Step(TIME_STEP, 8, 3)
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

	def step_two_points(self, start_pt, end_pt, gripper_length, path, display=False, check_reachable=True):

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

		vector = np.array(normalize(end_pt - start_pt))
		gripper1_vector = normalize((1, -(vector[0] / (vector[1]+1e-8))))
		gripper2_vector = normalize((-1, (vector[0] / (vector[1]+1e-8))))
		# print(np.array(gripper1_vector), gripper_length)
		gripper1_pt = start_pt + np.array(gripper1_vector) * gripper_length / 2
		gripper2_pt = start_pt + np.array(gripper2_vector) * gripper_length / 2

		# reachability check
		if check_reachable:
			vertices_lst=[(0.0, 0.1), (0.0, -0.1), (-0.3, -0.1), (-0.3, 0.1)]

			testrod = self.world.CreateKinematicBody(position=(gripper1_pt[0], gripper1_pt[1]), allowSleep=False)
			testrodfix = testrod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])

			testrod2 = self.world.CreateKinematicBody(position=(gripper2_pt[0], gripper2_pt[1]), allowSleep=False)
			testrodfix2 = testrod2.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])

			while (np.count_nonzero(np.array([o.dist_rod(testrodfix, testrod) for o in self.objs]) <= 0) > 0):
				# print(start_pt, [o.dist_rod(self.rodfix, self.rod) for o in self.objs])
				gripper1_pt -= 0.1 * vector
				gripper2_pt -= 0.1 * vector

				testrod.DestroyFixture(testrodfix)
				self.world.DestroyBody(testrod)

				testrod2.DestroyFixture(testrodfix2)
				self.world.DestroyBody(testrod2)

				testrod = self.world.CreateKinematicBody(position=(gripper1_pt[0], gripper1_pt[1]), allowSleep=False)
				testrodfix = testrod.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])

				testrod2 = self.world.CreateKinematicBody(position=(gripper2_pt[0], gripper2_pt[1]), allowSleep=False)
				testrodfix2 = testrod2.CreatePolygonFixture(vertices=[rotatePt(pt, vector) for pt in vertices_lst])

			testrod.DestroyFixture(testrodfix)
			self.world.DestroyBody(testrod)

			testrod2.DestroyFixture(testrodfix2)
			self.world.DestroyBody(testrod2)

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
				break

			self.world.Step(TIME_STEP, 8, 3)
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
			isolated = True
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
					if distance < threshold:
						isolated = False
			if isolated:
				count += 1
		return count

	def count_soft_threshold(self):
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
			# print(min_dist, (sigmoid(min_dist*10) - 0.5) * 2)
			count += (sigmoid(min_dist*10) - 0.5) * 2
		return count

	def collect_data_summary(self, start_pt, end_pt, img_path=None, sum_path=None):
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
		# summary["min centroid before push"] = self.min_centroid()
		# summary["min geometry before push"] = self.min_geometry()
		summary["count threshold before push"] = self.count_threshold()


		first_contact = self.step(start_pt, end_pt, img_path)

		# if first_contact == -1:
		# 	return

		for i in range(len(self.objs)):
			summary[str(i)+" change of pos"] = euclidean_dist(self.objs[i].body.position, self.objs[i].original_pos)
		summary["avg centroid after push"] = self.avg_centroid()
		summary["avg geometry after push"] = self.avg_geometry()
		# summary["min centroid after push"] = self.min_centroid()
		# summary["min geometry after push"] = self.min_geometry()
		summary["count threshold after push"] = self.count_threshold()
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

	def collect_data_area_summary(self, start_pt, end_pt, gripper_width, img_path=None, sum_path=None):
		summary = {}
		abs_start_pt = np.array(start_pt)
		abs_end_pt = np.array(end_pt)
		summary["start pt"] = abs_start_pt.tolist()
		summary["end pt"] = abs_end_pt.tolist()
		summary["gripper_width"] = gripper_width
		
		for i in range(len(self.objs)):
			summary[str(i)+" dist to pushing line"] = pointToLineDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" original pos"] = np.array(self.objs[i].body.position).tolist()
			# print(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" project dist"] = projectedPtToStartDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" vertices"] = np.array(self.objs[i].vertices).tolist()
			summary[str(i)+" disk coverage"] = self.objs[i].disk_coverage

		summary["avg centroid before push"] = self.avg_centroid()
		summary["avg geometry before push"] = self.avg_geometry()
		# summary["min centroid before push"] = self.min_centroid()
		# summary["min geometry before push"] = self.min_geometry()
		summary["count threshold before push"] = self.count_threshold()


		first_contact = self.step_area(start_pt, end_pt, gripper_width, img_path)

		# if first_contact == -1:
		# 	return

		for i in range(len(self.objs)):
			summary[str(i)+" change of pos"] = euclidean_dist(self.objs[i].body.position, self.objs[i].original_pos)
		summary["avg centroid after push"] = self.avg_centroid()
		summary["avg geometry after push"] = self.avg_geometry()
		# summary["min centroid after push"] = self.min_centroid()
		# summary["min geometry after push"] = self.min_geometry()
		summary["count threshold after push"] = self.count_threshold()
		summary["first contact object"] = first_contact
		
		if sum_path is not None:
			with open(sum_path+'summary.json', 'w') as f:
				json.dump(summary, f)
		
		return summary

	def collect_data_two_points_summary(self, start_pt, end_pt, gripper_length, img_path=None, sum_path=None):
		summary = {}
		abs_start_pt = np.array(start_pt)
		abs_end_pt = np.array(end_pt)
		summary["start pt"] = abs_start_pt.tolist()
		summary["end pt"] = abs_end_pt.tolist()
		summary["gripper_length"] = gripper_length
		
		for i in range(len(self.objs)):
			summary[str(i)+" dist to pushing line"] = pointToLineDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" original pos"] = np.array(self.objs[i].body.position).tolist()
			# print(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" project dist"] = projectedPtToStartDistance(abs_start_pt, abs_end_pt, self.objs[i].body.position)
			summary[str(i)+" vertices"] = np.array(self.objs[i].vertices).tolist()
			summary[str(i)+" disk coverage"] = self.objs[i].disk_coverage

		summary["avg centroid before push"] = self.avg_centroid()
		summary["avg geometry before push"] = self.avg_geometry()
		# summary["min centroid before push"] = self.min_centroid()
		# summary["min geometry before push"] = self.min_geometry()
		summary["count threshold before push"] = self.count_threshold()


		first_contact = self.step(start_pt, end_pt, gripper_length, img_path)

		# if first_contact == -1:
		# 	return

		for i in range(len(self.objs)):
			summary[str(i)+" change of pos"] = euclidean_dist(self.objs[i].body.position, self.objs[i].original_pos)
		summary["avg centroid after push"] = self.avg_centroid()
		summary["avg geometry after push"] = self.avg_geometry()
		# summary["min centroid after push"] = self.min_centroid()
		# summary["min geometry after push"] = self.min_geometry()
		summary["count threshold after push"] = self.count_threshold()
		summary["first contact object"] = first_contact
		
		if sum_path is not None:
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
		assert (len(dic) - 13) % 6 == 0
		num_obj = (len(dic) - 13) // 6
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

	# def sequential_planning_helper(self, prune_method, step, position, history, metric="count threshold"):
	# 	if step == 0:
	# 		return history[-1][metric+" after push"]-history[0][metric+" before push"], history
	# 	else:
	# 		pt_lst = prune_method(self)
	# 		best_sum = None
	# 		best_metric = 1e2
	# 		for pts in pt_lst:
	# 			self.load_position(position)
	# 			curr_hist = self.collect_data_summary(pts[0], pts[1], None)
	# 			hist_sum = copy.deepcopy(history)
	# 			hist_sum.append(curr_hist)
	# 			curr_pos = self.save_curr_position()
	# 			value, total = self.sequential_planning_helper(prune_method, step-1, curr_pos, hist_sum, metric)
	# 			if value is not None and value > best_metric:
	# 				best_sum = total
	# 				best_metric = total[-1][metric+" after push"]-total[0][metric+" before push"]
	# 		if best_sum is not None:
	# 			return best_metric, best_sum
	# 		else:
	# 			return None, None

	# def sequential_prune_planning(self, prune_method, max_step=10, metric="count threshold", sum_path=None):
	# 	curr_pos = self.save_curr_position()
	# 	best_metric, best_sum = self.sequential_planning_helper(prune_method, max_step, curr_pos, [], metric=metric)

	# 	# for i in range(max_step):
	# 	# 	curr_pos = self.save_curr_position()
	# 	# 	best_pts = self.prune_best(prune_method, metric, curr_pos)
	# 	# 	self.load_position(curr_pos)
	# 	# 	curr_sum = self.collect_data_summary(best_pts[0], best_pts[1], None)
	# 	# 	data_sum.append(curr_sum)
	# 	# 	print(curr_sum[metric +" after push"], curr_sum[metric + " before push"])
	# 	# return data_sum[-1][metric +" after push"], data_sum[-1][metric + " before push"]
	# 	if sum_path is not None and best_sum is not None:
	# 		print(best_metric)
	# 		with open(sum_path+'.json', 'w') as f:
	# 			json.dump(best_sum, f)
	# 	return best_sum

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

	def prune_best(self, prune_method, metric="count threshold", position=None, sum_path=None):
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
		
		if best_pt is not None:
			self.reset()
			return self.collect_data_summary(best_pt[0], best_pt[1], "/", sum_path=sum_path)
		return best_pt

if __name__ == "__main__":

	# path = "/nfs/diskstation/zdong/cluster_push_final/"
	path = "testing/"
	for num_obj in range(2, 21):
		if not os.path.exists(path+str(num_obj)):
			os.makedirs(path+str(num_obj))

		for g in range(0, 4):
			print(num_obj, g)
			i=0
			while i < 25:
				if not os.path.exists(path+str(num_obj)+"/"+str(g)+"/avg_centroid"):
					os.makedirs(path+str(num_obj)+"/"+str(g)+"/avg_centroid")
				if not os.path.exists(path+str(num_obj)+"/"+str(g)+"/count_threshold"):
					os.makedirs(path+str(num_obj)+"/"+str(g)+"/count_threshold")

				if not os.path.exists(path+str(num_obj)+"/"+str(g)+"/quasi_random"):
					os.makedirs(path+str(num_obj)+"/"+str(g)+"/quasi_random")
				
				print(i)
				test = SingulationEnv()
				while True:
					try:
						test.create_random_env(num_obj, g)
						break
					except:
						print("retry")

				test.reset()
				summary = test.prune_best(no_prune, metric="avg centroid", sum_path=path+str(num_obj)+"/"+str(g)+"/avg_centroid/"+str(i))
				print("take one")
				test.reset()
				summary = test.prune_best(no_prune, metric="count threshold", sum_path=path+str(num_obj)+"/"+str(g)+"/count_threshold/"+str(i))
				print("take two")

				# quasi random
				test.reset()
				push_pts = quasi_random(test)
				if push_pts is None:
					continue
				test.collect_data_summary(push_pts[0], push_pts[1], sum_path=path+str(num_obj)+"/"+str(g)+"/quasi_random/"+str(i))

				i = i+1
