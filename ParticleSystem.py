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
		self.particle_diameter = self.particle_radius * 2
		self.support_radius = 4 * self.particle_radius
		self.particle_m = 1000 * ((self.particle_radius ) ** 3) * 8

		self.particle_num = int(
			self.water_size.x / self.particle_diameter * self.water_size.y / self.particle_diameter * self.water_size.z / self.particle_diameter)

		self.pos = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.vel = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.acc = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.belong_grid = ti.Vector.field(3, ti.i32, shape=self.particle_num)

		self.box_max = ti.Vector(scene_config.get('box_max'))
		self.box_min = ti.Vector(scene_config.get('box_min'))

		# Grid
		grid_num_np = np.ceil(((self.box_max - self.box_min) / self.support_radius).to_numpy()).astype(np.int32)
		self.grid_num = ti.Vector([grid_num_np[0], grid_num_np[1], grid_num_np[2]])
		self._3d_to_1d_tran = ti.Vector([1, self.grid_num[0] * self.grid_num[2], self.grid_num[0]])
		S = ti.root.dense(ti.i, self.grid_num[0] * self.grid_num[1] * self.grid_num[2]).dynamic(ti.j, 512,
																								chunk_size=32)
		self.grids = ti.field(int)
		S.place(self.grids)
		self.init_particle()

		print('Particle count: {}k'.format(self.particle_num/1000))
		print('Particle mass: {}'.format(self.particle_m))
		print('Grid: {}, Grid count: {}'.format( self.grid_num, self.grid_num[0]* self.grid_num[1]* self.grid_num[2]))

	@ti.kernel
	def init_particle(self):
		for i in range(self.particle_num):
			x_num = self.water_size.x / self.particle_diameter
			z_num = self.water_size.z / self.particle_diameter
			xz_num = x_num * z_num

			x = i % x_num
			z = ti.floor(i / x_num) % z_num
			y = int(i / xz_num)
			self.pos[i] = ti.Vector([x, y, z]) * self.particle_radius * 2 + self.start_pos

	@ti.kernel
	def reset_grid(self):
		for I in ti.grouped(ti.ndrange(self.grid_num[0], self.grid_num[1], self.grid_num[2])):
			_1d_index = self.get_particle_grid_index_1d(I)
			if self.grids[_1d_index].length() != 0:
				self.grids[_1d_index].deactivate()

	# @ti.kernel
	def test(self):
		self.reset_grid()
		self.update_grid()
		self.check_all_grid()

		# self.for_all_neighbor(0)

	@ti.kernel
	def update_grid(self):
		for i in range(self.particle_num):
			grid_index_3d = self.get_particle_grid_index_3d(self.pos[i])
			grid_index_1d = self.get_particle_grid_index_1d(grid_index_3d)
			self.grids[grid_index_1d].append(i)
			self.belong_grid[i] = grid_index_3d

	@ti.kernel
	def get_max_neighbor_particle_index(self) -> ti.int32:

		max_count = -1
		max_index = -1
		for i in range(self.particle_num):
			neighbor_cnt = 0
			center = self.get_particle_grid_index_3d(self.pos[i])
			for I in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
				if (I + center >= self.grid_num).any():
					continue
				if not (I + center >= 0).all():
					continue
				_1d_index = self.get_particle_grid_index_1d(I + center)
				count = self.grids[_1d_index].length()
				for index in range(count):
					particle_j = self.grids[_1d_index, index]
					if particle_j == i:
						continue
					if (self.pos[i] - self.pos[particle_j]).norm() > self.support_radius:
						continue
					neighbor_cnt += 1
			# grid_index_1d = self.get_particle_grid_index_1d(grid_index_3d)
			# count = self.grids[grid_index_1d].length()
			new_max = ti.atomic_max(max_count, neighbor_cnt)
			if new_max == neighbor_cnt:
				max_index = i
		print('max_index is {}, length is {}'.format(max_index, max_count))
		return max_index

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

	@ti.func
	def get_particle_grid_index_1d(self, _3d_index):
		return _3d_index.dot(self._3d_to_1d_tran)
	@ti.func
	def get_particle_grid_index_3d(self, pos) -> ti.types.vector:
		if (ti.floor(pos / self.support_radius, ti.i32)>= self.grid_num).any():
			print(pos, pos / self.support_radius, ti.floor(pos / self.support_radius, ti.i32)>= self.grid_num)
		return ti.floor(pos / self.support_radius, ti.i32)