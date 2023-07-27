import taichi as ti
import numpy as np

@ti.data_oriented
class ParticleSystem:

	def __init__(self, box_min, box_max, particle_radius):
		# self.water_size = ti.Vector([0.5, 0.8, 0.5])
		self.water_size = ti.Vector([0.3, 0.5, 0.3])
		self.start_pos = ti.Vector([0, 0, 0])
		self.particle_radius = particle_radius
		self.support_radius = 4 * self.particle_radius
		# self.particle_m = 1000 * ((self.particle_radius * 2) ** 3) * ti.math.pi / 6
		self.particle_m = 1000 * ((self.particle_radius ) ** 3) * 8
		# self.particle_m = 1000 * ((self.particle_radius * 2) ** 3) * 0.8

		self.particle_num = int(
			self.water_size.x / self.particle_radius * self.water_size.y / self.particle_radius * self.water_size.z / self.particle_radius)
		print('particle count: ', self.particle_num)
		print('particle m: ', self.particle_m)

		self.pos = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		# self.pos_debug = ti.Vector.field(3, ti.f32, shape=27)
		self.vel = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.acc = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.belong_grid = ti.Vector.field(3, ti.i32, shape=self.particle_num)

		self.box_max = box_max
		self.box_min = box_min

		# Grid
		# grid.from_numpy(a)
		grid = ((self.box_max - self.box_min) / self.particle_radius).to_numpy().astype(np.int32)
		self.max_particle_in_grid = 30 # todo how to cal.
		# The first element is the count of the particle in this grid
		self.grids = ti.field(ti.i32, shape=(grid[0], grid[1], grid[2], self.max_particle_in_grid + 1))
		self.grids.fill(0)
		# self.grids = ti.field(ti.i32, shape=(2, 2, 20, 50))
		self.init_particle()


	@ti.kernel
	def init_particle(self):
		for i in range(self.particle_num):
			x_num = self.water_size.x / self.particle_radius
			z_num = self.water_size.z / self.particle_radius
			xz_num = x_num * z_num

			x = i % x_num
			z = ti.floor(i / x_num) % z_num
			y = int(i / xz_num)
			self.pos[i] = ti.Vector([x, y, z]) * self.particle_radius * 2 + self.start_pos

	@ti.func
	def reset_grid(self):
		# self.grids.fill(0)
		# grid = ((self.box_max - self.box_min) / self.particle_radius)  # .to_numpy().astype(np.int32)
		# print(grid)
		for I in ti.grouped(ti.ndrange(300, 300, 300)):
			if self.grids[I, 0] != 0:
				self.grids[I, 0] = 0

	@ti.kernel
	def test(self):
		self.reset_grid()
		self.update_grid()
		# self.check_all_grid()
		# self.for_all_neighbor(0)

	@ti.func
	def update_grid(self):
		for i in range(self.particle_num):
			grid_index = self.get_particle_grid_index(self.pos[i])

			length = ti.atomic_add(self.grids[grid_index, 0], 1)
			if length > self.max_particle_in_grid:
				print('WARNNING!!!!!')
				continue
			self.grids[grid_index, length + 1] = i
			self.belong_grid[i] = grid_index

	@ti.func
	def for_all_neighbor(self, i, task: ti.template(), ret: ti.template()):
		center = self.belong_grid[i]
		for I in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):

			if not (I + center >= 0).all():
				continue
			count = self.grids[I + center, 0]
			# print(count)
			for index in range(count):
				particle_j = self.grids[I + center, index+1]
				# print(particle_j, I, index + 1, center, count, index, self.grids[I + center, index])
				if particle_j == i:
					continue
				ret += task(i, particle_j)

	@ti.func
	def check_all_grid(self):
		grid = ((self.box_max - self.box_min) / self.particle_radius)#.to_numpy().astype(np.int32)
		# print(grid)
		sum = 0
		cnt = 0
		for I in ti.grouped(ti.ndrange(int(grid[0]), int(grid[1]), int(grid[2]))):
			sum += self.grids[I, 0]
			if self.grids[I, 0] > 8 and cnt == 0:
				pass
				# cnt += 1
				# print("large than 8", self.grids[I, 0], I)
				# for k in range(20):
				# 	if k == 0:
				# 		continue
				# 	index = self.grids[I, k]
				# 	print(self.pos[index], index, I)
				# print('\n')
		if not sum == self.particle_num:
			print("Check no Pass")
		# else:
		# 	print('Check no pass')
		# print('sum is ', sum)

	@ti.func
	def get_particle_grid_index(self, pos) -> ti.types.vector:
		return ti.round( pos / self.support_radius, ti.i32)
		# print(pos)
		# print(ti.floor(pos/ self.support_radius, ti.i32 ))
		# print('----')