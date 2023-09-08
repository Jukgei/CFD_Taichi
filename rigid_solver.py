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

		self.v_decay_proportion = 0.5

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
		ori_displacement = vel * self.delta_time[None]
		# self.displacement_distance = -ti.math.inf
		# displacement_min = displacement
		collision_point_cnt = 0
		collision_point = ti.Vector([0.0, 0.0, 0.0])
		collision_particle_v = ti.Vector([0.0, 0.0, 0.0])
		collision_norm = ti.Vector([0.0, 0.0, 0.0])
		collision_mass = 0.0
		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				collision = 0
				if self.ps.rigid_particles.pos[i][j] + ori_displacement[j] <= self.ps.box_min[j] + self.ps.particle_diameter:
					# self.ps.rigid_particles.rgb[i] = ti.Vector([1.0, 1.0, 1.0])
					ti.atomic_max(displacement[j], self.ps.box_min[j] + self.ps.particle_diameter - self.ps.rigid_particles.pos[i][j])
					v = vel + ti.math.cross(self.omega[None], (self.ps.rigid_particles.pos[i] + ori_displacement - self.ps.rigid_centriod[None]))
					if v[j] < 0:
						collision = 1
						collision_particle_v += v
						collision_norm[j] = -1

				if self.ps.rigid_particles.pos[i][j] + ori_displacement[j] >= self.ps.box_max[j] - self.ps.particle_diameter:
					# self.ps.rigid_particles.rgb[i] = ti.Vector([1.0, 1.0, 1.0])
					ti.atomic_min(displacement[j], self.ps.box_max[j] - self.ps.particle_diameter - self.ps.rigid_particles.pos[i][j])
					v = vel + ti.math.cross(self.omega[None], (self.ps.rigid_particles.pos[i] + ori_displacement - self.ps.rigid_centriod[None]))
					if v[j] > 0:
						collision = 1
						collision_particle_v += v
						collision_norm[j] = 1

				if collision == 1:
					collision_point += (self.ps.rigid_particles.pos[i] - self.ps.rigid_centriod[None])
					collision_point_cnt += 1
					collision_mass += self.ps.rigid_particles.mass[i]

		collision_particle_v_new = ti.Vector([0.0, 0.0, 0.0])

		if collision_point_cnt > 0:
			collision_point = (collision_point + ori_displacement) / collision_point_cnt
			collision_particle_v /= collision_point_cnt

			collision_particle_v_new = self.compute_new_vel(collision_particle_v, collision_norm)

		sum_mass = 0.0
		for i in range(self.ps.rigid_particles_num):
			sum_mass += self.ps.rigid_particles.mass[i]

		if collision_point_cnt > 0:
			collision_point_matrix = ti.math.mat3([[0.0, - collision_point.z, collision_point.y], [collision_point.z, 0.0, -collision_point.x], [-collision_point.y, collision_point.x, 0.0]])
			K = ti.Matrix.identity(ti.f32, 3) / sum_mass - collision_point_matrix @ ti.math.inverse(self.ps.rigid_inertia_tensor[None]) @ collision_point_matrix

			inv_K = ti.math.inverse(K)
			j = inv_K @ (collision_particle_v_new - collision_particle_v)
			vel += j / sum_mass
			self.omega[None] += ti.math.inverse(self.ps.rigid_inertia_tensor[None]) @ ti.math.cross(collision_point, j)

		self.ps.rigid_particles.vel.fill(vel)
		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] += displacement

		centroid = ti.Vector([0.0, 0.0, 0.0])

		for i in range(self.ps.rigid_particles_num):
			centroid += self.ps.rigid_particles.pos[i] * self.ps.rigid_particles.mass[i]

		self.ps.rigid_centriod[None] = centroid / sum_mass

	@ti.func
	def compute_new_vel(self, v, n) -> ti.math.vec3:
		mu_t = 0.8 # friction
		mu_n = self.v_decay_proportion
		v_n = v.dot(n) * n
		v_t = v - v_n
		a = ti.max(1 - mu_t * (1 + mu_n) * v_n.norm() / v_t.norm(), 0.0)
		v_n_new = - mu_n * v_n
		v_t_new = a * v_t
		v_new = v_t_new + v_n_new
		return v_new

	@ti.kernel
	def compute_attitude(self):
		torque = ti.Vector([0.0, 0.0, 0.0])
		for i in range(self.particle_count):
			pos = self.ps.rigid_particles.pos[i] - self.ps.rigid_centriod[None]
			torque += ti.math.cross(pos, self.ps.rigid_particles.force[i])

		alpha = ti.math.inverse(self.ps.rigid_inertia_tensor[None]) @ torque
		self.omega[None] += alpha * self.delta_time[None]
		self.attitude[None] = self.omega[None] * self.delta_time[None]

	@ti.kernel
	def rotation(self):
		m = ti.math.rotation3d(-self.attitude[None].x, -self.attitude[None].z, -self.attitude[None].y)
		R = ti.math.mat3([[m[0, 0], m[0, 1], m[0, 2]], [m[1, 0], m[1, 1], m[1, 2]], [m[2, 0], m[2, 1], m[2, 2]]])

		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] = R @(self.ps.rigid_particles.pos[i] - self.ps.rigid_centriod[None]) + self.ps.rigid_centriod[None]

		self.ps.rigid_inertia_tensor[None] = R @ self.ps.rigid_inertia_tensor[None] @ R.transpose()

	@ti.kernel
	def reset(self):
		self.ps.rigid_particles.acc.fill(0.0)

		# Debug & Visualization
		# self.ps.rigid_particles.rgb.fill(ti.Vector([1.0, 0.0, 0.0]))

	def step(self):
		self.simulate_cnt[None] += 1

		self.reset()

		self.compute_attitude()

		self.rotation()

		self.kinematic()