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

	@ti.kernel
	def kinematic(self):
		force = ti.Vector([0.0, 0.0, 0.0])
		for i in range(self.particle_count):
			force += self.ps.rigid_particles.force[i]
			# self.ps.rigid_particles.force[i] = ti.Vector([0.0, 0.0, 0.0])

		# rotation
		torque = ti.Vector([0.0, 0.0, 0.0])
		for i in range(self.particle_count):
			pos = self.ps.rigid_particles.pos[i] - self.ps.rigid_centriod[None]
			torque += ti.math.cross(pos, self.ps.rigid_particles.force[i])
			self.ps.rigid_particles.force[i] = ti.Vector([0.0, 0.0, 0.0])

		alpha = self.ps.rigid_inertia_tensor[None] @ torque
		# m = ti.math.rotation3d(0.0, 0.0, 0.0)
		# m = ti.math.rot_yaw_pitch_roll(1.0 / 180.0 * ti.math.pi, 0, 0)
		# m[0, 3] = -self.ps.rigid_centriod[None].x
		# m[1, 3] = -self.ps.rigid_centriod[None].y
		# m[2, 3] = -self.ps.rigid_centriod[None].z
		# print(m)
		self.omega[None] += alpha * self.delta_time[None]
		self.attitude[None] = self.omega[None] * self.delta_time[None]
		# m = ti.math.rotation3d(self.omega[None].x, self.omega[None].z, self.omega[None].y)
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

		sum_mass = 0.0
		for i in range(self.particle_count):
			sum_mass+=self.ps.rigid_particles.mass[i]

		self.ps.rigid_particles.acc.fill(force / sum_mass + self.gravity * ti.Vector([0.0, -1.0, 0.0]))
		# self.ps.rigid_particles.vel.fill()
		vel = self.ps.rigid_particles.acc[0] * self.delta_time[None] + self.ps.rigid_particles.vel[0]
		displacement = vel * self.delta_time[None]

		# self.displacement_distance = -ti.math.inf
		# displacement_min = displacement
		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.ps.rigid_particles.pos[i][j] + displacement[j] <= self.ps.box_min[j] + self.ps.particle_radius:
					ti.atomic_max(displacement[j], self.ps.box_min[j] + self.ps.particle_radius - self.ps.rigid_particles.pos[i][j])
					ti.atomic_max(vel[j], - vel[j] * self.v_decay_proportion)

				if self.ps.rigid_particles.pos[i][j] + displacement[j] >= self.ps.box_max[j] - self.ps.particle_radius:
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
	def reset(self):
		self.ps.rigid_particles.acc.fill(0.0)

	def step(self):
		self.simulate_cnt[None] += 1

		self.reset()

		self.kinematic()