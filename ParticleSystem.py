import taichi as ti
import numpy as np

@ti.data_oriented
class ParticleSystem:

	def __init__(self, config):
		scene_config = config.get('scene')
		solver_config = config.get('solver')
		fluid_config = config.get('fluid')

		self.water_size = ti.Vector(fluid_config.get('water_size'))
		self.start_pos = ti.Vector(fluid_config.get('start_pos'))
		self.particle_radius = scene_config.get('particle_radius')
		self.support_radius = 4 * self.particle_radius
		self.particle_m = 1000 * ((self.particle_radius ) ** 3) * 8

		self.particle_num = int(
			self.water_size.x / self.particle_radius * self.water_size.y / self.particle_radius * self.water_size.z / self.particle_radius)

		self.pos = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.vel = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.acc = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.belong_grid = ti.Vector.field(3, ti.i32, shape=self.particle_num)
		# self.belong_grid = ti.field(ti.i32, shape=self.particle_count)

		self.box_max = ti.Vector(scene_config.get('box_max'))
		self.box_min = ti.Vector(scene_config.get('box_min'))

		# Grid
		grid_num_np = np.ceil(((self.box_max - self.box_min) / self.support_radius).to_numpy()).astype(np.int32)
		self.grid_num = ti.Vector([grid_num_np[0], grid_num_np[1], grid_num_np[2]])
		self._3d_to_1d_tran = ti.Vector([1, self.grid_num[0] * self.grid_num[2], self.grid_num[0]])
		self.max_particle_in_grid = 80 		# todo how to cal.
		# The first element is the count of the particle in this grid
		S = ti.root.dense(ti.i, self.grid_num[0] * self.grid_num[1] * self.grid_num[2]).dynamic(ti.j, 512,
																								chunk_size=32)
		self.grids = ti.field(int)
		S.place(self.grids)
		self.init_particle()

		print('particle count: ', self.particle_num)
		print('particle m: ', self.particle_m)
		print('Grid num: ', self.grid_num, self.grid_num[0]* self.grid_num[1]* self.grid_num[2])

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

	@ti.kernel
	def reset_grid(self):
		# pass
		# cnt = 0
		for I in ti.grouped(ti.ndrange(self.grid_num[0], self.grid_num[1], self.grid_num[2])):
			_1d_index = self.get_particle_grid_index_1d(I)
			if self.grids[_1d_index].length() != 0:
				self.grids[_1d_index].deactivate()
		# 	if self.grids[I, 0] != 0:
		# 		self.grids[I, 0] = 0
		# 		cnt +=1
		# print('active grid count: ', cnt)

	# @ti.kernel
	def test(self):
		pass
		# print('213213:', self.grids[0].length())
		# self.grids[0].append(1)
		# print(self.grids[0,0])
		# for i in self.grids[0]:
		# 	print(i)
		# print('213213:', self.x[0].length(), self.x[0])
		self.reset_grid()
		# exit()
		self.update_grid()
		self.check_all_grid()

		# self.for_all_neighbor(0)

	@ti.kernel
	def update_grid(self):
		for i in range(self.particle_num):
			grid_index_3d = self.get_particle_grid_index_3d(self.pos[i])
			grid_index_1d = self.get_particle_grid_index_1d(grid_index_3d)
			self.grids[grid_index_1d].append(i)
			# if self.grids[grid_index, 0] >= self.max_particle_in_grid:
			# 	print("FIRST DETECT WARNNING!!!!!!", grid_index, self.pos[i], self.vel[i], self.acc[i])
			# 	continue
			# length = ti.atomic_add(self.grids[grid_index, 0], 1)
			# if length > self.max_particle_in_grid:
			# 	print('WARNNING!!!!!', length, grid_index)
			# 	continue
			# # if length > 30:
			# # 	print(length, grid_index)
			# self.grids[grid_index, length + 1] = i
			self.belong_grid[i] = grid_index_3d

	@ti.func
	def for_all_neighbor(self, i, task: ti.template(), ret: ti.template()):
		center = self.belong_grid[i]
		for I in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
			if (I + center >= self.grid_num).any():
				continue
			if not (I + center >= 0).all():
				continue
			_1d_index = self.get_particle_grid_index_1d(I+center)
			count = self.grids[_1d_index].length()
			for index in range(count):
				particle_j = self.grids[_1d_index, index]
				if particle_j == i:
					continue
				if (self.pos[i] - self.pos[particle_j]).norm() > self.support_radius:
					continue
				ret += task(i, particle_j)

	@ti.kernel
	def check_all_grid(self):
		sum = 0
		cnt = 0
		for I in ti.grouped(ti.ndrange(self.grid_num[0], self.grid_num[1], self.grid_num[2])):
			# sum += self.grids[I, 0]
			_1d_index = self.get_particle_grid_index_1d(I)
			sum += self.grids[_1d_index].length()
			if self.grids[_1d_index].length() > 8 and cnt == 0:
				print("gird index: ", _1d_index, ", length: ", self.grids[_1d_index].length())
		if not sum == self.particle_num:
			print("Fail!")
		else:
			print('Check pass!')
		# print('sum is ', sum)

	@ti.func
	def get_particle_grid_index_1d(self, _3d_index):
		return _3d_index.dot(self._3d_to_1d_tran)
	@ti.func
	def get_particle_grid_index_3d(self, pos) -> ti.types.vector:
		if (ti.floor(pos / self.support_radius, ti.i32)>= self.grid_num).any():
			print(pos, pos / self.support_radius, ti.floor(pos / self.support_radius, ti.i32)>= self.grid_num)
		return ti.floor(pos / self.support_radius, ti.i32)
		# print(pos)
		# print(ti.floor(pos/ self.support_radius, ti.i32 ))
		# print('----')