import taichi as ti


@ti.data_oriented
class rigid_solver:

	def __init__(self, particle_system, config):
		solid_config = config.get('solid')
		solver_config = config.get('solver')
		scene_config = config.get('scene')
		self.ps = particle_system

		self.delta_time = ti.field(ti.float32, shape=())
		self.delta_time[None] = solver_config.get('delta_time')
		self.simulate_cnt = ti.field(ti.int32, shape=())
		self.gravity = scene_config.get('gravity')

		self.rho = solid_config.get('rho_0')
		self.particle_count = self.ps.rigid_particles_num
		self.displacement = ti.Vector([0.0, 0.0, 0.0])
		self.omega = ti.field(ti.math.vec3, shape=())
		self.attitude = ti.field(ti.math.vec3, shape=())

		self.v_decay_proportion = 0.9

		self.omega_decay_proportion = 0.0

		self.temp_pos = ti.field(ti.math.vec3, shape=self.particle_count)
		self.collision_iteration_cnt = 5

	@ti.kernel
	def kinematic(self):
		force = ti.Vector([0.0, 0.0, 0.0])
		for i in range(self.particle_count):
			force += self.ps.rigid_particles.force[i]
			self.ps.rigid_particles.force[i] = ti.Vector([0.0, 0.0, 0.0])

		sum_mass = 0.0
		for i in range(self.particle_count):
			sum_mass += self.ps.rigid_particles.mass[i]

		self.ps.rigid_particles.acc.fill(force / sum_mass + self.gravity * ti.Vector([0.0, -1.0, 0.0]))
		# self.ps.rigid_particles.vel.fill()
		vel = self.ps.rigid_particles.acc[0] * self.delta_time[None] + self.ps.rigid_particles.vel[0]
		displacement = vel * self.delta_time[None]

		# self.displacement_distance = -ti.math.inf
		# displacement_min = displacement
		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.ps.rigid_particles.pos[i][j] + displacement[j] <= self.ps.box_min[j] + self.ps.particle_radius:
					# self.ps.rigid_particles.rgb[i] = ti.Vector([1.0, 1.0, 1.0])
					ti.atomic_max(displacement[j], self.ps.box_min[j] + self.ps.particle_radius - self.ps.rigid_particles.pos[i][j])
					ti.atomic_max(vel[j], - vel[j] * self.v_decay_proportion)

				if self.ps.rigid_particles.pos[i][j] + displacement[j] >= self.ps.box_max[j] - self.ps.particle_radius:
					# self.ps.rigid_particles.rgb[i] = ti.Vector([1.0, 1.0, 1.0])
					ti.atomic_min(displacement[j], self.ps.box_max[j] - self.ps.particle_radius - self.ps.rigid_particles.pos[i][j])
					ti.atomic_min(vel[j], - vel[j] * self.v_decay_proportion)

		self.ps.rigid_particles.vel.fill(vel)
		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] += displacement

		centroid = ti.Vector([0.0, 0.0, 0.0])
		sum_mass = 0.0
		for i in range(self.ps.rigid_particles_num):
			centroid += self.ps.rigid_particles.pos[i] * self.ps.rigid_particles.mass[i]
			sum_mass += self.ps.rigid_particles.mass[i]
		self.ps.rigid_centriod[None] = centroid / sum_mass

	@ti.kernel
	def compute_attitude(self):
		torque = ti.Vector([0.0, 0.0, 0.0])
		for i in range(self.particle_count):
			pos = self.ps.rigid_particles.pos[i] - self.ps.rigid_centriod[None]
			torque += ti.math.cross(pos, self.ps.rigid_particles.force[i])
		alpha = self.ps.rigid_inertia_tensor[None] @ torque
		self.omega[None] += alpha * self.delta_time[None]
		self.attitude[None] = self.omega[None] * self.delta_time[None]

	@ti.kernel
	def _iterate_collision(self, rad: ti.f32, axis: ti.i32) -> ti.int32:
		m = ti.math.mat4(1)
		for i in range(self.particle_count):
			self.temp_pos[i] = self.ps.rigid_particles.pos[i]
			self.temp_pos[i] -= self.ps.rigid_centriod[None]
		if axis == 0:
			m = ti.math.rotation3d(rad, 0.0, 0.0)
		elif axis == 1:
			m = ti.math.rotation3d(0.0, 0.0, rad)
		elif axis == 2:
			m = ti.math.rotation3d(0.0, rad, 0.0)

		for i in range(self.particle_count):
			pos = self.temp_pos[i]
			pos4 = ti.Vector([pos.x, pos.y, pos.z, 1])
			pos4 = m @ pos4
			self.temp_pos[i] = ti.Vector([pos4.x, pos4.y, pos4.z])

		for i in range(self.particle_count):
			self.temp_pos[i] += self.ps.rigid_centriod[None]

		is_collision = 0
		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.temp_pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
					is_collision = 1
				if self.temp_pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
					is_collision = 1
		return is_collision

	def rotate_collision(self, axis):
		assert 0 <= axis < 3, "axis must larger than 0 and less than 3"
		rad = self.attitude[None][axis]
		is_collision = 0
		iter_cnt = 0
		while iter_cnt < self.collision_iteration_cnt:
			is_collision = self._iterate_collision(rad, axis)
			# print('iter count {}, is collision {}'.format(iter_cnt, is_collision))
			if is_collision == 0 and iter_cnt == 0:
				break
			if is_collision == 1:
				rad /= 2
			elif is_collision == 0:
				rad += 0.5 * rad
			iter_cnt += 1

		if is_collision == 1:
			self.attitude[None][axis] = rad / 2
		elif is_collision == 0:
			self.attitude[None][axis] = rad
		if iter_cnt > 0:
			# print('collision ! axis {}'.format(axis))
			self.omega[None][axis] *= -self.omega_decay_proportion

	@ti.kernel
	def rotation(self):
		m = ti.math.rotation3d(self.attitude[None].x, self.attitude[None].z, self.attitude[None].y)

		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] -= self.ps.rigid_centriod[None]

		for i in range(self.particle_count):
			pos = self.ps.rigid_particles.pos[i]
			pos4 = ti.Vector([pos.x, pos.y, pos.z, 1])
			pos4 = m @ pos4
			self.ps.rigid_particles.pos[i] = ti.Vector([pos4.x, pos4.y, pos4.z])

		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] += self.ps.rigid_centriod[None]

		# Artificial rotation friction
		self.omega[None] *= 0.99

	@ti.kernel
	def reset(self):
		self.ps.rigid_particles.acc.fill(0.0)

		# Debug & Visualization
		# self.ps.rigid_particles.rgb.fill(ti.Vector([1.0, 0.0, 0.0]))

	def step(self):
		self.simulate_cnt[None] += 1

		self.reset()

		self.compute_attitude()

		self.rotate_collision(0)
		self.rotate_collision(1)
		self.rotate_collision(2)

		self.rotation()

		self.kinematic()