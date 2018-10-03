import logging
import math
import time

import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
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
					original_pos = np.array([np.random.uniform(-0.5*num_objs,0.5*num_objs),np.random.uniform(-0.5*num_objs,0.5*num_objs)]) + np.array(self.objs[-1].body.position)
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
		if display:
			self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
			pygame.display.iconify()

			self.screen.fill((255, 255, 255, 255))

			def my_draw_polygon(polygon, body, fixture, color):
				vertices = [(body.transform * v) * PPM for v in polygon.vertices]
				vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
				
				pygame.draw.polygon(self.screen, color, vertices, 0)
				pygame.draw.polygon(self.screen, (0,0,0,0), vertices, 3)

			polygonShape.draw = my_draw_polygon
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
		if display:
			for obj in self.objs:
				for fix in obj.fixtures:
					fix.shape.draw(obj.body, fix, obj.color)

			self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))

			pygame.image.save(self.screen, path+"start.png")
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
			if display:
				self.screen.fill((255, 255, 255, 255))

				for obj in self.objs:
					for fix in obj.fixtures:
						fix.shape.draw(obj.body, fix, obj.color)

				self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))

				pygame.image.save(self.screen, path+str(timestamp)+".png")
			#

			
		# display
		if display:
			pygame.display.quit()
			pygame.quit()
		#

		return first_contact

	def step_area(self, start_pt, end_pt, gripper_width, path, display=False):

		# display
		if display:
			self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
			pygame.display.iconify()

			self.screen.fill((255, 255, 255, 255))

			def my_draw_polygon(polygon, body, fixture, color):
				vertices = [(body.transform * v) * PPM for v in polygon.vertices]
				vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
				
				pygame.draw.polygon(self.screen, color, vertices, 0)
				pygame.draw.polygon(self.screen, (0,0,0,0), vertices, 3)

			polygonShape.draw = my_draw_polygon
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

		# display
		if display:
			for obj in self.objs:
				for fix in obj.fixtures:
					fix.shape.draw(obj.body, fix, obj.color)

			self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))

			pygame.image.save(self.screen, path+"start.png")
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
			if display:
				self.screen.fill((255, 255, 255, 255))

				for obj in self.objs:
					for fix in obj.fixtures:
						fix.shape.draw(obj.body, fix, obj.color)

				self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))

				pygame.image.save(self.screen, path+str(timestamp)+".png")
			#

			
		# display
		if display:
			pygame.display.quit()
			pygame.quit()
		#

		return first_contact

	def step_two_points(self, start_pt, end_pt, gripper_length, path, display=False):

		# display
		if display:
			self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
			pygame.display.iconify()

			self.screen.fill((255, 255, 255, 255))

			def my_draw_polygon(polygon, body, fixture, color):
				vertices = [(body.transform * v) * PPM for v in polygon.vertices]
				vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
				
				pygame.draw.polygon(self.screen, color, vertices, 0)
				pygame.draw.polygon(self.screen, (0,0,0,0), vertices, 3)

			polygonShape.draw = my_draw_polygon
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
		if display:
			for obj in self.objs:
				for fix in obj.fixtures:
					fix.shape.draw(obj.body, fix, obj.color)

			self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))
			self.rodfix2.shape.draw(self.rod2, self.rodfix2, (0, 0, 0, 255))

			pygame.image.save(self.screen, path+"start.png")
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
			if display:
				self.screen.fill((255, 255, 255, 255))

				for obj in self.objs:
					for fix in obj.fixtures:
						fix.shape.draw(obj.body, fix, obj.color)

				self.rodfix.shape.draw(self.rod, self.rodfix, (0, 0, 0, 255))
				self.rodfix2.shape.draw(self.rod2, self.rodfix2, (0, 0, 0, 255))

				pygame.image.save(self.screen, path+str(timestamp)+".png")
			#

			
		# display
		if display:
			pygame.display.quit()
			pygame.quit()
		#

		return first_contact

	def avg_centroid(self):
		total_separation = 0
		for i in range(len(self.objs) - 1):
			for j in range(i+1, len(self.objs)):
				if i != j:
					total_separation += math.log(euclidean_dist(self.objs[i].body.position, self.objs[j].body.position))
		return total_separation * 2/(len(self.objs) * (len(self.objs) - 1))

	def avg_geometry(self):
		total_separation = 0
		for i in range(len(self.objs) - 1):
			for j in range(i+1, len(self.objs)):
				if i != j:
					min_dist = 1e2
					for v1 in self.objs[i].vertices+self.objs[i].body.position:
						for v2 in self.objs[j].vertices+self.objs[j].body.position:
							if min_dist > math.log(euclidean_dist(v1, v2)):
								min_dist = math.log(euclidean_dist(v1, v2))
					total_separation += min_dist
		return total_separation * 2/(len(self.objs) * (len(self.objs) - 1))

	def min_geometry(self):
		min_dist = 1e2
		for i in range(len(self.objs) - 1):
			for j in range(i+1, len(self.objs)):
				if i != j:
					min_dist = 1e2
					for v1 in self.objs[i].vertices+self.objs[i].body.position:
						for v2 in self.objs[j].vertices+self.objs[j].body.position:
							if min_dist > (euclidean_dist(v1, v2)):
								min_dist = (euclidean_dist(v1, v2))
		return min_dist

	def min_centroid(self):
		min_dist = 1e2
		for i in range(len(self.objs) - 1):
			for j in range(i+1, len(self.objs)):
				if i != j:
					if min_dist > (euclidean_dist(self.objs[i].body.position, self.objs[j].body.position)):
						min_dist = euclidean_dist(self.objs[i].body.position, self.objs[j].body.position)
		return min_dist

	def count_threshold(self, threshold=0.4):
		count = 0
		for i in range(len(self.objs) - 1):
			for j in range(i+1, len(self.objs)):
				if i != j:
					min_dist = 1e2
					for v1 in self.objs[i].vertices+self.objs[i].body.position:
						for v2 in self.objs[j].vertices+self.objs[j].body.position:
							if min_dist > (euclidean_dist(v1, v2)):
								min_dist = (euclidean_dist(v1, v2))
					if min_dist > threshold:
						count += 1
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

	def visualize(self, path):
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
		pygame.display.iconify()

		self.screen.fill((0, 0, 0, 0))

		def my_draw_polygon(polygon, body, fixture, color):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			
			pygame.draw.polygon(self.screen, color, vertices, 0)

		polygonShape.draw = my_draw_polygon

		for obj in self.objs:
			obj.fixture.shape.draw(obj.body, obj.fixture, obj.color)

		self.rodfix.shape.draw(self.rod, self.rodfix, (100, 100, 100, 255))

		pygame.image.save(self.screen, path+"debug.png")

		pygame.display.quit()
		pygame.quit()

if __name__ == "__main__":

	metrics = ["min geometry", "count threshold", "avg geometry", "avg centroid", "min centroid"]
	for num_obj in range(5, 17):
		if not os.path.exists("prune/"+str(num_obj)):
			os.makedirs("prune/"+str(num_obj))
		for g in range(len(GROUPS)):
		# for g in range(1, len(GROUPS)):
			print(num_obj, g)
			
			if not os.path.exists("prune/"+str(num_obj)+"/"+str(g)):
				os.makedirs("prune/"+str(num_obj)+"/"+str(g))
			for m in metrics:
				if not os.path.exists("prune/"+str(num_obj)+"/"+str(g)+"/"+m):
					os.makedirs("prune/"+str(num_obj)+"/"+str(g)+"/"+m)
			i = 0
			while i < 100:
				test = SingulationEnv()
				while True:
					try:
						test.create_random_env(num_obj, g)
						break
					except:
						print("retry")
				success = True
				print(i)
				
				test.reset()
				dic_all = test.prune_best_summary_all(no_prune, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"no_prune", metrics=metrics)
				optima_all = np.array([dic_all[j][metrics[j]+" after push"] - dic_all[j][metrics[j]+" before push"] + 1e-2 for j in range(len(metrics))])
				# print(optima_all)
				print("no_prune:" + str(np.array([dic_all[j][metrics[j]+" after push"] - dic_all[j][metrics[j]+" before push"] for j in range(len(metrics))])/optima_all))
				
				test.reset()
				dic_all = test.prune_best_summary_all(com_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"com_only", metrics=metrics)
				print("com_only:" + str(np.array([dic_all[j][metrics[j]+" after push"] - dic_all[j][metrics[j]+" before push"] for j in range(len(metrics))])/optima_all))
				
				test.reset()
				dic_all = test.prune_best_summary_all(max_gain_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"max_gain_only", metrics=metrics)
				print("max_gain_only:" + str(np.array([dic_all[j][metrics[j]+" after push"] - dic_all[j][metrics[j]+" before push"] for j in range(len(metrics))])/optima_all))
				
				test.reset()
				dic_all = test.prune_best_summary_all(two_only, "prune/"+str(num_obj)+"/"+str(g), str(i).zfill(2)+"two_only", metrics=metrics)
				print("two_only:" + str(np.array([dic_all[j][metrics[j]+" after push"] - dic_all[j][metrics[j]+" before push"] for j in range(len(metrics))])/optima_all))
				
				for j in range(len(metrics)):
					# test.reset()
					# dic = test.prune_best_summary(no_prune, "prune/"+str(num_obj)+"/"+str(g)+"/"+m+"/"+str(i).zfill(2)+"no_prune", metric=m)
					# optima = dic[m+" after push"] - dic[m+" before push"]+1e-2
					# print(dic[m+" after push"], dic[m+" before push"])
					# print(m+" no_prune:" + str((dic[m+" after push"] - dic[m+" before push"])/optima))
					# test.reset()
					# dic = test.prune_best_summary(com_only, "prune/"+str(num_obj)+"/"+str(g)+"/"+m+"/"+str(i).zfill(2)+"com_only", metric=m)
					# print(m+" com_only:" + str((dic[m+" after push"] - dic[m+" before push"])/optima))
					# test.reset()
					# dic = test.prune_best_summary(max_gain_only, "prune/"+str(num_obj)+"/"+str(g)+"/"+m+"/"+str(i).zfill(2)+"max_gain_only", metric=m)
					# print(m+" max_gain_only:" + str((dic[m+" after push"] - dic[m+" before push"])/optima))
					# test.reset()
					# dic = test.prune_best_summary(two_only, "prune/"+str(num_obj)+"/"+str(g)+"/"+m+"/"+str(i).zfill(2)+"two_only", metric=m)
					# # print(dic)
					# print(m+" two_only:" + str((dic[m+" after push"] - dic[m+" before push"])/optima))
					test.reset()
					pts = proposed9(test)
					if pts is not None:
						dic = test.collect_data_summary(pts[0], pts[1], "/", "prune/"+str(num_obj)+"/"+str(g)+"/"+metrics[j]+"/"+str(i).zfill(2)+"clusterPush")
						print(metrics[j]+" cluster2D:" + str((dic[metrics[j]+" after push"] - dic[metrics[j]+" before push"])/optima_all[j]))
						test.reset()
					else: 
						success = False

					test.reset()
					pts = proposed9_refined(test)
					if pts is not None:
						dic = test.collect_data_summary(pts[0], pts[1], "/", "prune/"+str(num_obj)+"/"+str(g)+"/"+metrics[j]+"/"+str(i).zfill(2)+"clusterPush_v1")
						print(metrics[j]+" cluster2D_v1:" + str((dic[metrics[j]+" after push"] - dic[metrics[j]+" before push"])/optima_all[j]))
						test.reset()
					else: 
						success = False

					test.reset()
					pts = proposed9_refined2(test)
					if pts is not None:
						dic = test.collect_data_summary(pts[0], pts[1], "/", "prune/"+str(num_obj)+"/"+str(g)+"/"+metrics[j]+"/"+str(i).zfill(2)+"clusterPush_v2")
						print(metrics[j]+" cluster2D_v2:" + str((dic[metrics[j]+" after push"] - dic[metrics[j]+" before push"])/optima_all[j]))
						test.reset()
					else: 
						success = False

				if success:
					i += 1






	# test = SingulationEnv()
	# test.create_random_env(6, 2)

	# test.reset()
	# pts = proposed0(test)
	# test.step(pts[0], pts[1], "visual/proposed0/", display=True)
	# test.reset()
	# test.step_area(pts[0], pts[1], 0.4, "visual/proposed0_area/", display=True)
	# test.reset()
	# test.step_two_points(pts[0], pts[1], 0.9, "visual/proposed0_two_pts/", display=True)

	# test.reset()
	# pts = proposed1(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/proposed1/", display=True)

	# test.reset()
	# pts = proposed2(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/proposed2/", display=True)

	# test.reset()
	# pts = proposed3(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/proposed3/", display=True)

	# test.reset()
	# pts = proposed4(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/proposed4/", display=True)

	# test.reset()
	# pts = proposed5(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/proposed5/", display=True)

	# test.reset()
	# pts = proposed6(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/proposed6/", display=True)

	# # test.reset()
	# # pts = proposed7(test)
	# # if pts is not None:
	# # 	test.step(pts[0], pts[1], "visual/proposed7/", display=True)

	# test.reset()
	# pts = proposed8(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/proposed8/", display=True)

	# test.reset()
	# pts = proposed9(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/proposed9/", display=True)

	# test.reset()
	# pts = boundaryShear(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/boundaryShear/", display=True)

	# test.reset()
	# pts = clusterDiffusion(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/clusterDiffusion/", display=True)

	# test.reset()
	# pts = maximumClearanceRatio(test)
	# if pts is not None:
	# 	test.step(pts[0], pts[1], "visual/maximumClearanceRatio/", display=True)

	
	# for num_obj in range(6, 16):
	# 	if not os.path.exists("new_proposed/"+str(num_obj)):
	# 		os.makedirs("new_proposed/"+str(num_obj))
	# 	# group_log = {"bruteForce":[], "proposed0":[], "proposed1":[], "proposed6":[], "proposed8":[], "proposed9":[], \
	# 	# 		"clusterDiffusion":[], "boundaryShear":[], "quasiRandom":[], "maximumClearanceRatio":[]}

	# 	for g in range(len(GROUPS)):
	# 		print(num_obj, g)
			
	# 		if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)):
	# 			os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g))
	# 		for i in range(100):
	# 			# print(i)
	# 			test = SingulationEnv()
	# 			while True:
	# 				try:
	# 					test.create_random_env(num_obj, g)
	# 					break
	# 				except:
	# 					print("retry")


	# 			for ind in range(num_obj*256):

				
	# 				obj_ind = ind // 256
	# 				theta1 = ((ind % 256) // 16) * 3.14 * 2 / 16
	# 				theta2 = (ind % 16) * 3.14 * 2 / 16
	# 				if theta1 == theta2:
	# 					continue
	# 				pt1 = (math.cos(theta1) * test.objs[obj_ind].bounding_circle_radius, math.sin(theta1) * test.objs[obj_ind].bounding_circle_radius)
	# 				pt2 = (math.cos(theta2) * test.objs[obj_ind].bounding_circle_radius, math.sin(theta2) * test.objs[obj_ind].bounding_circle_radius)
	# 				# print(pt1, pt2)
	# 				pts = parametrize_by_bounding_circle(np.array(pt1) + np.array(test.objs[obj_ind].original_pos), np.array(pt2) - np.array(pt1), test.objs[obj_ind].original_pos, test.objs[obj_ind].bounding_circle_radius+0.1)
	# 				if pts is None:
	# 					print(obj_ind, np.array(pt1) + np.array(test.objs[obj_ind].original_pos), np.array(pt2) - np.array(pt1), test.objs[obj_ind].original_pos, test.objs[obj_ind].bounding_circle_radius+0.1)
	# 					continue
	# 				test.reset()

	# 				# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution/"):
	# 				# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution/")
					
	# 				# summary = test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution/"+str(i)+"_"+str(ind))
	# 				# group_log["quasiRandom"].append(time.perf_counter()-timelog_qr)
	# 				# test.reset()
					
	# 				# summary = test.collect_data_two_points_summary(pts[0], pts[1], test.objs[obj_ind].bounding_circle_radius*2/3, "/", "compare_gap/distribution_multi_pts/"+str(i)+"_"+str(ind))
	# 				# test.reset()
	# 				# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "/distribution_area/"+str(i)+"_"+str(ind))
	# 				# test.reset()

					

	# 				if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution_stable/"):
	# 					os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution_stable/")
	# 				# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution_stable_two/"):
	# 				# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution_stable_two/")
	# 				# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution_stable_area/"):
	# 				# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution_stable_area/")

	# 				if abs(abs(theta1 - theta2) - 3.14) < 0.1:
	# 					summary = test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution_stable/"+str(i)+"_"+str(ind))
	# 					test.reset()
	# 					# summary = test.collect_data_two_points_summary(pts[0], pts[1], test.objs[obj_ind].bounding_circle_radius*0.6, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution_stable_two/"+str(i)+"_"+str(ind))
	# 					# test.reset()
	# 					# summary = test.collect_data_area_summary(pts[0], pts[1], 0.4, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/distribution_stable_area/"+str(i)+"_"+str(ind))
	# 					# test.reset()


				
	# 			# test.reset()
	# 			# timelog_p = time.perf_counter()
	# 			# pts = proposed0(test)
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed0/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed0/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed0_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed0_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed0_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed0_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed0/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed0_area/"+str(i))
	# 			# 	# for l in np.arange(0.1, 2, 0.1):
	# 			# 	# 	test.reset()
	# 			# 	# 	summary = test.collect_data_two_points_summary(pts[0], pts[1], l * test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed0_two/"+str(i)+"_"+str(int(l*10)))
	# 			# group_log["proposed0"].append(time.perf_counter()-timelog_p)
				
	# 			# test.reset()

	# 			# timelog_p = time.perf_counter()
	# 			# pts = proposed1(test)
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed1/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed1/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed1_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed1_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed1_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed1_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed1/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed1_area/"+str(i))
	# 			# 	# for l in np.arange(0.1, 2, 0.1):
	# 			# 	# 	test.reset()
	# 			# 	# 	summary = test.collect_data_two_points_summary(pts[0], pts[1], l * test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed1_two/"+str(i)+"_"+str(int(l*10)))
	# 			# group_log["proposed1"].append(time.perf_counter()-timelog_p)
				
	# 			# test.reset()
	# 			# pts = proposed2(test)
	# 			# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed2/"):
	# 			# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed2/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed2_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed2_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed2_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed2_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed2/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed2_area/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_two_points_summary(pts[0], pts[1], test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed2_two/"+str(i))

	# 			# test.reset()
	# 			# pts = proposed3(test)
	# 			# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed3/"):
	# 			# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed3/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed3_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed3_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed3_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed3_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed3/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed3_area/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_two_points_summary(pts[0], pts[1], test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed3_two/"+str(i))

	# 			# test.reset()
	# 			# timelog_p = time.perf_counter()
	# 			# pts = proposed4(test)
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed4/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed4/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed4_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed4_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed4_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed4_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed4/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed4_area/"+str(i))
	# 			# 	# for l in np.arange(0.1, 2, 0.1):
	# 			# 	# 	test.reset()
	# 			# 	# 	summary = test.collect_data_two_points_summary(pts[0], pts[1], l * test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed4_two/"+str(i)+"_"+str(int(l*10)))
	# 			# group_log["proposed4"].append(time.perf_counter()-timelog_p)
				
	# 			# test.reset()
	# 			# pts = proposed5(test)
	# 			# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed5/"):
	# 			# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed5/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed5_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed5_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed5_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed5_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed5/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed5_area/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_two_points_summary(pts[0], pts[1], test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed5_two/"+str(i))

	# 			# test.reset()
	# 			# timelog_p = time.perf_counter()
	# 			# pts = proposed6(test)
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed6/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed6/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed6_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed6_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed6_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed6_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed6/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed6_area/"+str(i))
	# 			# 	# for l in np.arange(0.1, 2, 0.1):
	# 			# 	# 	test.reset()
	# 			# 	# 	summary = test.collect_data_two_points_summary(pts[0], pts[1], l * test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed6_two/"+str(i)+"_"+str(int(l*10)))
	# 			# group_log["proposed6"].append(time.perf_counter()-timelog_p)
				
	# 			# # test.reset()
	# 			# # pts = proposed7(test)
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed7/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed7/")
	# 			# # # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed7_area/"):
	# 			# # # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed7_area/")
	# 			# # # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed7_two/"):
	# 			# # # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed7_two/")
	# 			# # if pts is not None:
	# 			# # 	test.reset()
	# 			# # 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed7/"+str(i))
	# 			# # 	# test.reset()
	# 			# # 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed7_area/"+str(i))
	# 			# # 	# test.reset()
	# 			# # 	# summary = test.collect_data_two_points_summary(pts[0], pts[1], test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed7_two/"+str(i))

	# 			# test.reset()
	# 			# timelog_p = time.perf_counter()
	# 			# pts = proposed8(test)
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed8/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed8/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed8_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed8_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed8_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed8_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed8/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed8_area/"+str(i))
	# 			# 	# for l in np.arange(0.1, 2, 0.1):
	# 			# 	# 	test.reset()
	# 			# 	# 	summary = test.collect_data_two_points_summary(pts[0], pts[1], l * test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed8_two/"+str(i)+"_"+str(int(l*10)))
	# 			# group_log["proposed8"].append(time.perf_counter()-timelog_p)
				
				# test.reset()
				# # timelog_p = time.perf_counter()
				# pts = proposed9(test)
				# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed9/"):
				# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed9/")
				# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed9_area/"):
				# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed9_area/")
				# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed9_two/"):
				# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed9_two/")
				# if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)):
				# 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g))
				# if pts is not None:
				# 	test.reset()
				# 	summary = test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed9/"+str(i))
				# 	test.reset()
				# 	summary_area = test.collect_data_area_summary(pts[0], pts[1], 0.4, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed9_area/"+str(i))
				# 	# for l in np.arange(0.1, 2, 0.1):
				# 	test.reset()
				# 	summary_two = test.collect_data_two_points_summary(pts[0], pts[1], 0.6 * test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/proposed9_two/"+str(i)+"_"+str(int(0.6*10)))
					# if (summary_area['mean separation after push'] - summary_area['mean separation before push']) - (summary['mean separation after push'] - summary['mean separation before push']) > 2:
					# 	if not os.path.exists("best_worst/"+str(num_obj)+"/"+str(g)+"/best_area/"+str(i)):
					# 		os.makedirs("best_worst/"+str(num_obj)+"/"+str(g)+"/best_area/"+str(i))
					# 	if not os.path.exists("best_worst/"+str(num_obj)+"/"+str(g)+"/best_area/"+str(i)+"area/"):
					# 		os.makedirs("best_worst/"+str(num_obj)+"/"+str(g)+"/best_area/"+str(i)+"area/")
					# 	test.reset()
					# 	test.step(pts[0], pts[1], "best_worst/"+str(num_obj)+"/"+str(g)+"/best_area/"+str(i)+"/", display=True)
					# 	test.reset()
					# 	test.step_area(pts[0], pts[1], 0.4, "best_worst/"+str(num_obj)+"/"+str(g)+"/best_area/"+str(i)+"area/", display=True)


					# if (summary_area['mean separation after push'] - summary_area['mean separation before push']) - (summary['mean separation after push'] - summary['mean separation before push']) < -1:
					# 	if not os.path.exists("best_worst/"+str(num_obj)+"/"+str(g)+"/worst_area/"+str(i)):
					# 		os.makedirs("best_worst/"+str(num_obj)+"/"+str(g)+"/worst_area/"+str(i))
					# 	if not os.path.exists("best_worst/"+str(num_obj)+"/"+str(g)+"/worst_area/"+str(i)+"area/"):
					# 		os.makedirs("best_worst/"+str(num_obj)+"/"+str(g)+"/worst_area/"+str(i)+"area/")
					# 	test.reset()
					# 	test.step(pts[0], pts[1], "best_worst/"+str(num_obj)+"/"+str(g)+"/worst_area/"+str(i)+"/", display=True)
					# 	test.reset()
					# 	test.step_area(pts[0], pts[1], 0.4, "best_worst/"+str(num_obj)+"/"+str(g)+"/worst_area/"+str(i)+"area/", display=True)


					# if (summary_two['mean separation after push'] - summary_area['mean separation before push']) - (summary['mean separation after push'] - summary['mean separation before push']) > 3:
					# 	if not os.path.exists("best_worst/"+str(num_obj)+"/"+str(g)+"/best_two/"+str(i)):
					# 		os.makedirs("best_worst/"+str(num_obj)+"/"+str(g)+"/best_two/"+str(i))
					# 	if not os.path.exists("best_worst/"+str(num_obj)+"/"+str(g)+"/best_two/"+str(i)+"two"):
					# 		os.makedirs("best_worst/"+str(num_obj)+"/"+str(g)+"/best_two/"+str(i)+"two")
					# 	test.reset()
					# 	test.step(pts[0], pts[1], "best_worst/"+str(num_obj)+"/"+str(g)+"/best_two/"+str(i)+"/", display=True)
					# 	test.reset()
					# 	test.step_two_points(pts[0], pts[1], 0.6 * test.objs[find_best_remove_object(test)].bounding_circle_radius, "best_worst/"+str(num_obj)+"/"+str(g)+"/best_two/"+str(i)+"two/", display=True)


					# if (summary_two['mean separation after push'] - summary_area['mean separation before push']) - (summary['mean separation after push'] - summary['mean separation before push']) < -2:
					# 	if not os.path.exists("best_worst/"+str(num_obj)+"/"+str(g)+"/worst_two/"+str(i)):
					# 		os.makedirs("best_worst/"+str(num_obj)+"/"+str(g)+"/worst_two/"+str(i))
					# 	if not os.path.exists("best_worst/"+str(num_obj)+"/"+str(g)+"/worst_two/"+str(i)+"two"):
					# 		os.makedirs("best_worst/"+str(num_obj)+"/"+str(g)+"/worst_two/"+str(i)+"two")
					# 	test.reset()
					# 	test.step(pts[0], pts[1], "best_worst/"+str(num_obj)+"/"+str(g)+"/worst_two/"+str(i)+"/", display=True)
					# 	test.reset()
					# 	test.step_two_points(pts[0], pts[1], 0.6 * test.objs[find_best_remove_object(test)].bounding_circle_radius, "best_worst/"+str(num_obj)+"/"+str(g)+"/worst_two/"+str(i)+"two/", display=True)



	# 			# group_log["proposed9"].append(time.perf_counter()-timelog_p)
				
	# 			# test.reset()
	# 			# timelog_p = time.perf_counter()
	# 			# pts = boundaryShear(test)
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/boundaryShear/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/boundaryShear/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/boundaryShear_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/boundaryShear_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/boundaryShear_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/boundaryShear_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/boundaryShear/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/boundaryShear_area/"+str(i))
	# 			# 	# for l in np.arange(0.1, 2, 0.1):
	# 			# 	# 	test.reset()
	# 			# 	# 	summary = test.collect_data_two_points_summary(pts[0], pts[1], l * test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/boundaryShear_two/"+str(i)+"_"+str(int(l*10)))
	# 			# group_log["boundaryShear"].append(time.perf_counter()-timelog_p)
				
	# 			# test.reset()
	# 			# timelog_p = time.perf_counter()
	# 			# pts = clusterDiffusion(test)
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/clusterDiffusion/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/clusterDiffusion/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/clusterDiffusion_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/clusterDiffusion_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/clusterDiffusion_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/clusterDiffusion_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/clusterDiffusion/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/clusterDiffusion_area/"+str(i))
	# 			# 	# for l in np.arange(0.1, 2, 0.1):
	# 			# 	# 	test.reset()
	# 			# 	# 	summary = test.collect_data_two_points_summary(pts[0], pts[1], l * test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/clusterDiffusion_two/"+str(i)+"_"+str(int(l*10)))
	# 			# group_log["clusterDiffusion"].append(time.perf_counter()-timelog_p)
				
	# 			# test.reset()
	# 			# timelog_p = time.perf_counter()
	# 			# pts = maximumClearanceRatio(test)
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/maximumClearanceRatio/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/maximumClearanceRatio/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/maximumClearanceRatio_area/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/maximumClearanceRatio_area/")
	# 			# # if not os.path.exists("new_proposed/"+str(num_obj)+"/"+str(g)+"/maximumClearanceRatio_two/"):
	# 			# # 	os.makedirs("new_proposed/"+str(num_obj)+"/"+str(g)+"/maximumClearanceRatio_two/")
	# 			# if pts is not None:
	# 			# 	test.reset()
	# 			# 	test.collect_data_summary(pts[0], pts[1], "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/maximumClearanceRatio/"+str(i))
	# 			# 	# test.reset()
	# 			# 	# summary = test.collect_data_area_summary(pts[0], pts[1], 0.3, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/maximumClearanceRatio_area/"+str(i))
	# 			# 	# for l in np.arange(0.1, 2, 0.1):
	# 			# 	# 	test.reset()
	# 			# 	# 	summary = test.collect_data_two_points_summary(pts[0], pts[1], l * test.objs[find_best_remove_object(test)].bounding_circle_radius, "/", "new_proposed/"+str(num_obj)+"/"+str(g)+"/maximumClearanceRatio_two/"+str(i)+"_"+str(int(l*10)))
	# 			# group_log["maximumClearanceRatio"].append(time.perf_counter()-timelog_p)
	# 	# for key in time_log.keys():
	# 	# 	time_log[key].append(np.mean(group_log[key]))
	# 	# 	print(np.mean(group_log[key]))

	# # for key in group_log.keys():
	# # 	print("Final:")
	# # 	print(key, group_log[key])
				

