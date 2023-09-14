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
		self.mass = ti.field(ti.f32, shape=())

		self.v_decay_proportion = 0.1

		self.run_once_flag = False

		self.max_boundary = ti.field(ti.math.vec3, shape=())
		self.min_boundary = ti.field(ti.math.vec3, shape=())
		self.debug_particles = ti.field(ti.math.vec3, shape=100)
		self.debug_index = ti.field(ti.i32, shape=())

	@ti.kernel
	def kinematic(self):
		force = ti.Vector([0.0, 0.0, 0.0])
		for i in range(self.particle_count):
			force += self.ps.rigid_particles.force[i]
			self.ps.rigid_particles.force[i] = ti.Vector([0.0, 0.0, 0.0])

		acc = force / self.mass[None] + self.gravity * ti.Vector([0.0, -1.0, 0.0])
		self.ps.rigid_particles.acc.fill(acc)
		# self.ps.rigid_particles.vel.fill()
		vel = self.ps.rigid_particles.acc[0] * self.delta_time[None] + self.ps.rigid_particles.vel[0]
		# print('rigid acc {}, force {}, vel {}'.format(acc, force, vel))
		displacement = vel * self.delta_time[None]
		ori_displacement = displacement
		# self.displacement_distance = -ti.math.inf
		# displacement_min = displacement
		collision_point_cnt = 0
		collision_point = ti.Vector([0.0, 0.0, 0.0])
		collision_particle_v = ti.Vector([0.0, 0.0, 0.0])
		collision_norm = ti.Vector([0.0, 0.0, 0.0])
		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				collision = 0
				if self.ps.rigid_particles.pos[i][j] + ori_displacement[j] <= self.ps.box_min[j] + self.ps.particle_diameter:
					# self.ps.rigid_particles.rgb[i] = ti.Vector([1.0, 1.0, 1.0])
					ti.atomic_max(displacement[j], self.ps.box_min[j] + self.ps.particle_diameter - self.ps.rigid_particles.pos[i][j])
					v = vel + ti.math.cross(self.omega[None], (self.ps.rigid_particles.pos[i] + ori_displacement - self.ps.rigid_centriod[None]))
					if v[j] < 0:
						collision = 1
						# collision_particle_v += v
						collision_norm[j] = -1

				if self.ps.rigid_particles.pos[i][j] + ori_displacement[j] >= self.ps.box_max[j] - self.ps.particle_diameter:
					# self.ps.rigid_particles.rgb[i] = ti.Vector([1.0, 1.0, 1.0])
					ti.atomic_min(displacement[j], self.ps.box_max[j] - self.ps.particle_diameter - self.ps.rigid_particles.pos[i][j])
					v = vel + ti.math.cross(self.omega[None], (self.ps.rigid_particles.pos[i] + ori_displacement - self.ps.rigid_centriod[None]))
					if v[j] > 0:
						collision = 1
						# collision_particle_v += v
						collision_norm[j] = 1

				if collision == 1:
					collision_point += self.ps.rigid_particles.pos[i]
					collision_point_cnt += 1

		collision_particle_v_new = ti.Vector([0.0, 0.0, 0.0])

		if collision_point_cnt > 0:
			collision_point = (collision_point + ori_displacement) / collision_point_cnt - self.ps.rigid_centriod[None]
			# collision_particle_v /= collision_point_cnt
			collision_particle_v = vel + ti.math.cross(self.omega[None], collision_point)

			collision_particle_v_new = self.compute_new_vel(collision_particle_v, collision_norm)

		if collision_point_cnt > 0:
			collision_point_matrix = ti.math.mat3([[0.0, - collision_point.z, collision_point.y], [collision_point.z, 0.0, -collision_point.x], [-collision_point.y, collision_point.x, 0.0]])
			K = ti.Matrix.identity(ti.f32, 3) / self.mass[None] - collision_point_matrix @ self.ps.rigid_inertia_tensor_inv[None] @ collision_point_matrix

			inv_K = ti.math.inverse(K)
			j = inv_K @ (collision_particle_v_new - collision_particle_v)
			vel += j / self.mass[None]
			self.omega[None] += self.ps.rigid_inertia_tensor_inv[None] @ ti.math.cross(collision_point, j)

		self.ps.rigid_particles.omega.fill(self.omega[None])
		self.ps.rigid_particles.vel.fill(vel)
		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] += displacement

		for i in range(self.ps.rigid_vertex_count):
			self.ps.rigid_vertices[i] += displacement

		self.ps.rigid_centriod[None] += displacement

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

		alpha = self.ps.rigid_inertia_tensor_inv[None] @ torque
		self.omega[None] += alpha * self.delta_time[None]
		self.attitude[None] = self.omega[None] * self.delta_time[None]
		self.ps.rigid_particles.alpha.fill(alpha)

	@ti.kernel
	def rotation(self):
		m = ti.math.rotation3d(-self.attitude[None].x, -self.attitude[None].z, -self.attitude[None].y)
		R = ti.math.mat3([[m[0, 0], m[0, 1], m[0, 2]], [m[1, 0], m[1, 1], m[1, 2]], [m[2, 0], m[2, 1], m[2, 2]]])

		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] = R @(self.ps.rigid_particles.pos[i] - self.ps.rigid_centriod[None]) + self.ps.rigid_centriod[None]

		for i in range(self.ps.rigid_vertex_count):
			self.ps.rigid_vertices[i] = R@(self.ps.rigid_vertices[i] - self.ps.rigid_centriod[None]) + self.ps.rigid_centriod[None]

		self.ps.rigid_inertia_tensor_inv[None] = R @ self.ps.rigid_inertia_tensor_inv[None] @ R.transpose()

	@ti.kernel
	def reset(self):
		pass
		# self.ps.rigid_particles.acc.fill(0.0)

		# Debug & Visualization
		# self.ps.rigid_particles.rgb.fill(ti.Vector([1.0, 0.0, 0.0]))

		for i in range(self.debug_index[None]):
			self.debug_particles[i] = ti.Vector([0.0, 0.0, 0.0])

		self.debug_index[None] = 0

	@ti.kernel
	def compute_sum_mass(self):
		sum_mass = 0.0
		for i in range(self.particle_count):
			sum_mass += self.ps.rigid_particles.mass[i]
		self.mass[None] = sum_mass
		print('rigid mass is {}'.format(sum_mass))

	@ti.kernel
	def check_penetrate(self):
		offset = self.ps.rigid_particles[0].index_offset
		for i in range(self.particle_count):
			cnt = 0
			self.ps.for_all_neighbor(i+offset, self.judge_penetrate, cnt)
			# if cnt > 0:
			# 	print('penetrate')

	@ti.func
	def judge_penetrate(self, particle_i, particle_j):
		cnt = 0
		if particle_j.material == self.ps.material_fluid:
			pos = particle_j.pos - self.ps.rigid_centriod[None]
			if pos.x < self.max_boundary[None].x and pos.y < self.max_boundary[None].y and pos.z < self.max_boundary[None].z:
				# cnt += 1
				# j = particle_j.index
				# self.ps.fluid_particles[j].rgb = ti.Vector([1.0, 1.0, 1.0])
				# print('up {}, {}, {}'.format(particle_j.pos, self.ps.rigid_centriod[None], pos))
				if pos.x > self.min_boundary[None].x and pos.y > self.min_boundary[None].y and pos.z > self.min_boundary[None].z:
					cnt += 1
					j = particle_j.index

					self.ps.fluid_particles[j].rgb = ti.Vector([1.0, 1.0, 1.0])
					if self.debug_index[None] < 100:
						self.debug_particles[self.debug_index[None]] = particle_j.pos
						self.debug_index[None] += 1
					# print('down {}, {}, {}'.format(particle_j.pos, self.ps.rigid_centriod[None], pos))
		return cnt

	@ti.kernel
	def init_boundary(self):
		for i in range(self.particle_count):
			pos = self.ps.rigid_particles[i].pos - self.ps.rigid_centriod[None]
			ti.atomic_max(self.max_boundary[None].x, pos.x)
			ti.atomic_max(self.max_boundary[None].y, pos.y)
			ti.atomic_max(self.max_boundary[None].z, pos.z)

			ti.atomic_min(self.min_boundary[None].x, pos.x)
			ti.atomic_min(self.min_boundary[None].y, pos.y)
			ti.atomic_min(self.min_boundary[None].z, pos.z)

		diameter = self.ps.particle_diameter
		self.max_boundary[None] -= ti.Vector([diameter, diameter, diameter])
		self.max_boundary[None] += ti.Vector([diameter, diameter, diameter])

		print('Rigid boundary max: {}, min: {}'.format(self.max_boundary[None], self.min_boundary[None]))

	def run_once(self):
		self.compute_sum_mass()
		self.init_boundary()

	def step(self):
		if not self.run_once_flag:
			self.run_once()
			self.run_once_flag = True

		self.simulate_cnt[None] += 1

		if self.ps.delta_time[None] > 0.0:
			self.delta_time[None] = self.ps.delta_time[None]

		self.reset()

		self.compute_attitude()

		self.rotation()

		self.kinematic()

		# self.check_penetrate()
