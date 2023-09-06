import taichi as ti
import numpy as np
import trimesh as tm
from solver_base import solver_base

Particles = ti.types.struct(
	pos=ti.math.vec3,
	vel=ti.math.vec3,
	acc=ti.math.vec3,
	mass=float,
	material=ti.int32,
	volume=float,
	belong_grid=ti.math.ivec3,
	index_offset=int,
	force=ti.math.vec3,
	index=int,
	rgb=ti.math.vec3
	# omega=ti.math.vec3
)

ParticlesBlocks = ti.types.struct(
	block=Particles,
	particles_count=int,
	pos_offset=ti.math.vec3
)

@ti.data_oriented
class ParticleSystem:

	def __init__(self, config):
		scene_config = config.get('scene')
		solver_config = config.get('solver')
		fluid_config = config.get('fluid')
		solid_config = config.get('solid', {})

		self.exist_rigid = ti.field(int, shape=())
		self.exist_rigid[None] = 1 if solid_config else 0
		if self.exist_rigid[None] == 1:
			mesh = tm.load_mesh(solid_config.get('mesh'))
			mesh = mesh.apply_scale(solid_config.get('scale'))
			self.voxel_radius = solid_config.get('voxel_radius')
			if solid_config.get('fill', True):
				voxelized_mesh = mesh.voxelized(pitch=self.voxel_radius).fill()
			else:
				voxelized_mesh = mesh.voxelized(pitch=self.voxel_radius)
			voxelized_points_np = voxelized_mesh.points
			self.rigid_pos_offset = solid_config.get('pos_offset')
			self.rigid_attitude_offset = ti.Vector(solid_config.get('attitude_offset')) / 180.0 * ti.math.pi
			self.rigid_rho = solid_config.get('rho_0')
			self.rigid_particles_num = voxelized_points_np.shape[0]
			self.rigid_particles = Particles.field(shape=self.rigid_particles_num)
			self.rigid_particles.pos.from_numpy(voxelized_points_np)
			self.rigid_centriod = ti.field(ti.math.vec3, shape=())
			self.rigid_inertia_tensor = ti.field(ti.math.mat3, shape=())
			self.active_rigid = ti.field(int, shape=())
			self.active_rigid[None] = 1 if solid_config.get('active', False) else 0
		else:
			self.rigid_particles_num = 0
			self.rigid_particles = Particles.field(shape=1)
			# self.rigid_particles = None
			# self.rigid_particles = Particles.field(shape=self.rigid_particles_num)
		# self.voxelized_points = ti.Vector.field(3, ti.f32, self.rigid_particles_num)
		# self.voxelized_points.from_numpy(voxelized_points_np)

		self.material_fluid = 0
		self.material_solid_boundary = 1
		self.material_solid = 2

		self.water_size = ti.Vector(fluid_config.get('water_size'))
		self.start_pos = ti.Vector(fluid_config.get('start_pos'))
		self.particle_radius = scene_config.get('particle_radius')
		self.particle_diameter = self.particle_radius * 2
		self.support_radius = 4 * self.particle_radius
		self.particle_m = 1000 * (self.particle_radius ** 3) * 8

		self.particle_num = int(
			self.water_size.x / self.particle_diameter * self.water_size.y / self.particle_diameter * self.water_size.z / self.particle_diameter)

		self.fluid_particles = Particles.field(shape=self.particle_num)

		# self.belong_grid = ti.Vector.field(3, ti.i32, shape=self.particle_num)

		self.box_max = ti.Vector(scene_config.get('box_max'))
		self.box_min = ti.Vector(scene_config.get('box_min'))

		self.boundary_particles_num = self.compute_boundary_particles_count()
		print('Boundary particle count: {}'.format(self.boundary_particles_num))
		self.boundary_particles = Particles.field(shape=self.boundary_particles_num)

		# Grid
		grid_num_np = np.ceil(((self.box_max - self.box_min) / self.support_radius).to_numpy()).astype(np.int32)
		self.grid_num = ti.Vector([grid_num_np[0] + 1, grid_num_np[1] + 1, grid_num_np[2] + 1])
		self._3d_to_1d_tran = ti.Vector([1, self.grid_num[0] * self.grid_num[2], self.grid_num[0]])
		S = ti.root.dense(ti.i, self.grid_num[0] * self.grid_num[1] * self.grid_num[2]).dynamic(ti.j, 512, chunk_size=32)
		self.grids = ti.field(int)
		S.place(self.grids)

		# Boundary Grids
		S = ti.root.dense(ti.i, self.grid_num[0] * self.grid_num[1] * self.grid_num[2]).dynamic(ti.j, 512, chunk_size=32)
		self.boundary_grids = ti.field(int)
		S.place(self.boundary_grids)

		# Output
		self.rgba = ti.Vector.field(4, dtype=ti.f32, shape=self.particle_num)

		# Visualization
		self.rgb = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.rgb.fill(ti.Vector([0.0, 0.28, 1.0]))

		self.init_particle_pos()
		if self.exist_rigid[None] == 1:
			self.init_rigid_particles_pos()
		self.init_particles_data()

		print('Fluid particle count: {}k'.format(self.particle_num/1000))
		print('Solid particle count: {}k'.format(self.rigid_particles_num/1000))
		print('Particle mass: {}'.format(self.particle_m))
		print('Grid: {}, Grid count: {}'.format( self.grid_num, self.grid_num[0]* self.grid_num[1]* self.grid_num[2]))

	def compute_boundary_particles_count(self):
		box = (self.box_max - self.box_min)
		x_cnt = int(box.x / self.particle_diameter + 1)
		z_cnt = int(box.z / self.particle_diameter + 1)
		bottom_layer_particle_cnt = x_cnt * z_cnt
		one_round_particle_cnt = x_cnt * z_cnt - (x_cnt - 2) * (z_cnt - 2)
		layer = int(ti.ceil((box.y - self.particle_diameter) / self.particle_diameter))
		particles_count = layer * one_round_particle_cnt + bottom_layer_particle_cnt * 2
		return particles_count

	@ti.kernel
	def init_particle_pos(self):
		# Init fluid
		for i in range(self.particle_num):
			x_num = self.water_size.x / self.particle_diameter
			z_num = self.water_size.z / self.particle_diameter
			xz_num = x_num * z_num

			x = i % x_num
			z = ti.floor(i / x_num) % z_num
			y = int(i / xz_num)
			self.fluid_particles.pos[i] = ti.Vector([x, y, z]) * self.particle_radius * 2 + self.start_pos
			self.fluid_particles.index[i] = i
		self.rgba.fill(ti.Vector([0.0, 0.26, 0.68, 1.0]))

		# Init boundary
		box = (self.box_max - self.box_min)
		x_cnt = int(box.x / self.particle_diameter + 1)
		z_cnt = int(box.z / self.particle_diameter + 1)
		x_cnt_round = x_cnt - 1
		z_cnt_round = z_cnt - 1
		bottom_layer_particle_cnt = x_cnt * z_cnt
		one_round_particle_cnt = x_cnt * z_cnt - (x_cnt - 2) * (z_cnt - 2)
		for i in range(self.boundary_particles_num):
			self.boundary_particles.index[i] = i
			if i < bottom_layer_particle_cnt:
				x = i % x_cnt * self.particle_diameter
				y = 0.0
				z = ti.floor(i / x_cnt) * self.particle_diameter
				self.boundary_particles[i].pos = ti.Vector([x, y, z])
			elif bottom_layer_particle_cnt <= i < self.boundary_particles_num - bottom_layer_particle_cnt:
				index = i - bottom_layer_particle_cnt
				layer = ti.floor(index / one_round_particle_cnt, ti.i32)
				y = self.particle_diameter * (layer + 1)
				index -= layer * one_round_particle_cnt
				x = 0.0
				z = 0.0
				index += 1
				if index <= x_cnt_round:
					x = index % x_cnt_round * self.particle_diameter
					z = 0.0
				elif x_cnt_round < index <= x_cnt_round + z_cnt_round:
					x = x_cnt_round * self.particle_diameter
					z = (index - x_cnt) % z_cnt_round * self.particle_diameter
				elif x_cnt_round + z_cnt_round < index <= 2 * x_cnt_round + z_cnt_round:
					x = ((2 * x_cnt_round + z_cnt_round - index) % x_cnt_round + 1) * self.particle_diameter
					z = z_cnt_round * self.particle_diameter
				elif 2 * x_cnt_round + z_cnt_round < index <= 2 * (x_cnt_round + z_cnt_round):
					x = 0.0
					z = ((2 * (x_cnt_round + z_cnt_round) - index) % z_cnt_round + 1) * self.particle_diameter
				self.boundary_particles[i].pos = ti.Vector([x, y, z])
			else:
				index = i - (self.boundary_particles_num - bottom_layer_particle_cnt)
				x = index % x_cnt * self.particle_diameter
				y = self.box_max.y
				z = int(index / x_cnt) * self.particle_diameter
				self.boundary_particles[i].pos = ti.Vector([x, y, z])

		# Init rigid
	@ti.kernel
	def init_rigid_particles_pos(self):
		m = ti.math.rotation3d(self.rigid_attitude_offset.x, self.rigid_attitude_offset.z, self.rigid_attitude_offset.y)
		for i in range(self.rigid_particles_num):
			if self.exist_rigid[None] == 1:

				pos = self.rigid_particles.pos[i]
				pos4 = ti.Vector([pos.x, pos.y, pos.z, 1])
				pos4 = m @ pos4
				self.rigid_particles.pos[i] = ti.Vector([pos4.x, pos4.y, pos4.z])

		for i in range(self.rigid_particles_num):
			if self.exist_rigid[None] == 1:
				self.rigid_particles.pos[i] += ti.Vector(self.rigid_pos_offset)
				self.rigid_particles.index[i] = i

	def init_particles_data(self):
		self.fluid_particles.material.fill(self.material_fluid)
		self.fluid_particles.index_offset.fill(0)

		self.boundary_particles.index_offset.fill(self.particle_num)
		self.boundary_particles.material.fill(self.material_solid_boundary)

		if self.exist_rigid[None] == 1:
			self.rigid_particles.index_offset.fill(self.particle_num + self.boundary_particles_num)
			self.rigid_particles.material.fill(self.material_solid)

		self.reset_boundary_grids()
		self.update_boundary_grids()

		self.reset_grid()
		self.update_grid()

		self.compute_all_boundary_volume()

		# self.compute_all_rigid_volume()
		if self.exist_rigid[None]:
			self.init_rigid_particles_data()

	@ti.kernel
	def init_rigid_particles_data(self):
		# volume
		for i in range(self.rigid_particles_num):
			volume = 0.0
			self.for_all_neighbor(i + self.rigid_particles.index_offset[i], self.compute_rigid_volume, volume)
			if volume < 1e-6:
				print('WARNNING, volume too small')
				self.rigid_particles.volume[i] = 0.0
			else:
				self.rigid_particles.volume[i] = 1.0 / volume

		# mass
		for i in range(self.rigid_particles_num):
			self.rigid_particles.mass[i] = self.rigid_rho * self.rigid_particles.volume[i]

		# Centroid
		centroid = ti.Vector([0.0, 0.0, 0.0])
		sum_mass = 0.0
		for i in range(self.rigid_particles_num):
			centroid += self.rigid_particles.pos[i] * self.rigid_particles.mass[i]
			sum_mass += self.rigid_particles.mass[i]
		self.rigid_centriod[None] = centroid / sum_mass
		print("Centroid: {}".format(centroid / sum_mass))

		# Inertia tensor
		Ixx = 0.0
		Iyy = 0.0
		Izz = 0.0
		Ixy = 0.0
		Ixz = 0.0
		Iyz = 0.0
		for i in range(self.rigid_particles_num):
			pos = self.rigid_particles.pos[i] - self.rigid_centriod[None]
			Ixx += self.rigid_particles.mass[i] * (pos.y ** 2 + pos.z ** 2)
			Iyy += self.rigid_particles.mass[i] * (pos.x ** 2 + pos.z ** 2)
			Izz += self.rigid_particles.mass[i] * (pos.x ** 2 + pos.y ** 2)
			Ixy += - self.rigid_particles.mass[i] * (pos.x * pos.y)
			Ixz += - self.rigid_particles.mass[i] * (pos.x * pos.z)
			Iyz += - self.rigid_particles.mass[i] * (pos.z * pos.y)

		self.rigid_inertia_tensor[None] = ti.math.mat3([Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz])
		print("Intertia tensor: {}".format(self.rigid_inertia_tensor[None]))
	# @ti.kernel
	# def compute_all_rigid_volume(self):
		self.rigid_particles.rgb.fill(ti.Vector([1.0, 0.0, 0.0]))


	@ti.func
	def compute_rigid_volume(self, particle_i, particle_j):
		ret = 0.0
		if particle_j.material == self.material_solid:
			q = (particle_i.pos - particle_j.pos).norm()
			ret = solver_base.cubic_kernel(q, self.support_radius)
		return ret

	@ti.kernel
	def compute_all_boundary_volume(self):
		for i in range(self.boundary_particles_num):
			volume = 0.0
			self.for_all_boundary_neighbor(i + self.boundary_particles.index_offset[i], self.compute_boundary_volume, volume)
			self.boundary_particles.volume[i] = 1.0 / volume

	@ti.func
	def compute_boundary_volume(self, i, j):
		q = (self.boundary_particles.pos[i] - self.boundary_particles.pos[j]).norm()
		ret = solver_base.cubic_kernel(q, self.support_radius)
		return ret

	@ti.kernel
	def reset_boundary_grids(self):
		for I in ti.grouped(ti.ndrange(self.grid_num[0], self.grid_num[1], self.grid_num[2])):
			_1d_index = self.get_particle_grid_index_1d(I)
			if self.boundary_grids[_1d_index].length() != 0:
				self.boundary_grids[_1d_index].deactivate()

	@ti.kernel
	def update_boundary_grids(self):
		for i in range(self.boundary_particles_num):
			grid_index_3d = self.get_particle_grid_index_3d(self.boundary_particles.pos[i])
			grid_index_1d = self.get_particle_grid_index_1d(grid_index_3d)
			self.boundary_grids[grid_index_1d].append(i)
			self.boundary_particles.belong_grid[i] = grid_index_3d

	@ti.func
	def for_all_boundary_neighbor(self, i, task: ti.template(), ret: ti.template()):

		center = ti.Vector([0, 0, 0])
		center_pos = ti.Vector([0.0, 0.0, 0.0])
		is_same_material = 1
		if i >= self.particle_num:
			i -= self.particle_num
			center = self.boundary_particles.belong_grid[i]
			center_pos = self.boundary_particles.pos[i]
		else:
			is_same_material = 0
			center = self.fluid_particles.belong_grid[i]
			center_pos = self.fluid_particles.pos[i]

		for I in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
			if (I + center >= self.grid_num).any():
				continue
			if not (I + center >= 0).all():
				continue

			_1d_index = self.get_particle_grid_index_1d(I + center)
			count = self.boundary_grids[_1d_index].length()
			for index in range(count):
				particle_j = self.boundary_grids[_1d_index, index]
				if particle_j == i and is_same_material == 1:
					continue
				if (center_pos - self.boundary_particles.pos[particle_j]).norm() > self.support_radius:
					continue
				ret += task(i, particle_j)

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

	# @ti.kernel
	def update_grid(self):
		self.update_grid_fluid_particles()

		if self.exist_rigid[None] == 1:
			self.update_grid_rigid_particles()

	@ti.kernel
	def update_grid_fluid_particles(self):
		for i in range(self.particle_num):
			grid_index_3d = self.get_particle_grid_index_3d(self.fluid_particles.pos[i])
			grid_index_1d = self.get_particle_grid_index_1d(grid_index_3d)
			self.grids[grid_index_1d].append(i)
			self.fluid_particles.belong_grid[i] = grid_index_3d

	@ti.kernel
	def update_grid_rigid_particles(self):
		for i in range(self.rigid_particles_num):
			if self.active_rigid[None] == 0:
				continue
			grid_index_3d = self.get_particle_grid_index_3d(self.rigid_particles.pos[i])
			grid_index_1d = self.get_particle_grid_index_1d(grid_index_3d)
			self.grids[grid_index_1d].append(i + self.rigid_particles.index_offset[i])
			self.rigid_particles.belong_grid[i] = grid_index_3d

	@ti.kernel
	def get_max_neighbor_particle_index(self) -> ti.int32:

		max_count = -1
		max_index = -1
		for i in range(self.particle_num):
			neighbor_cnt = self.get_neighbour_count(i)
			# grid_index_1d = self.get_particle_grid_index_1d(grid_index_3d)
			# count = self.grids[grid_index_1d].length()
			new_max = ti.atomic_max(max_count, neighbor_cnt)
			if new_max == neighbor_cnt:
				max_index = i
		print('max_index is {}, length is {}'.format(max_index, max_count))
		return max_index

	@ti.func
	def get_neighbour_count(self, i):
		neighbor_cnt = 0
		center = self.get_particle_grid_index_3d(self.fluid_particles.pos[i])
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
				if (self.fluid_particles.pos[i] - self.fluid_particles.pos[particle_j]).norm() > self.support_radius:
					continue
				neighbor_cnt += 1
		return neighbor_cnt

	@ti.func
	def for_all_neighbor(self, i, task: ti.template(), ret: ti.template()):
		# center = self.fluid_particles.belong_grid[i]
		particle = self.get_particle(i)
		center = particle.belong_grid
		for I in ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2))):
			if (I + center >= self.grid_num).any():
				continue
			if not (I + center >= 0).all():
				continue
			_1d_index = self.get_particle_grid_index_1d(I+center)
			count = self.grids[_1d_index].length()
			for index in range(count):
				neighbor_index = self.grids[_1d_index, index]
				if neighbor_index == i:
					continue
				particle_j = self.get_particle(neighbor_index)
				# if not particle_j.material == self.material_fluid:
				# 	continue
				if (particle.pos - particle_j.pos).norm() > self.support_radius:
					continue
				if particle_j.material == self.material_solid and particle.material == self.material_fluid:
					print('fluid index {}, solid index {}'.format(particle.index, particle_j.index))
				# ret += task(i, neighbor_index)
				ret += task(particle, particle_j)

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
			print('Position illegal ', pos, pos / self.support_radius, ti.floor(pos / self.support_radius, ti.i32)>= self.grid_num)
		return ti.floor(pos / self.support_radius, ti.i32)

	@ti.func
	def get_particle(self, index):
		ret = self.fluid_particles[0]
		if index < self.particle_num:
			ret = self.fluid_particles[index]
		elif self.particle_num <= index < self.particle_num + self.boundary_particles_num:
			ret = self.boundary_particles[index - self.particle_num]
		elif self.particle_num + self.boundary_particles_num <= index < self.particle_num + self.boundary_particles_num + self.rigid_particles_num:
			ret = self.rigid_particles[index - self.particle_num - self.boundary_particles_num]
		else:
			print('WARNNING!')
		return ret