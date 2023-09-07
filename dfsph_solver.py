import taichi as ti
from solver_base import solver_base


class dfsph_solver(solver_base):

	def __init__(self, particle_system, config):
		super(dfsph_solver, self).__init__(particle_system, config)

		self.alpha = ti.field(ti.float32, shape=self.particle_count)
		self.rho_adv = ti.field(ti.float32, shape=self.particle_count)
		self.rho_derivative = ti.field(ti.float32, shape=self.particle_count)
		self.vel_adv = ti.Vector.field(3, ti.float32, shape=self.particle_count)
		self.vel_adv_delta = ti.Vector.field(3, ti.float32, shape=self.particle_count)
		self.force_ext = ti.Vector.field(3, ti.float32, shape=self.particle_count)
		self.force_ext.fill(self.gravity * ti.Vector([0.0, -1.0, 0.0]) * self.ps.particle_m)

		self.delta_time_2 = self.delta_time[None] ** 2
		self.min_iteration_density = 2
		self.density_threshold = 0.1
		self.min_iteration_density_divergence = 1
		self.max_iteration_density_divergence = 25
		self.density_divergence_threshold = 1
		# self.tension_k = 5

	@ti.func
	def compute_all_alpha(self):
		for i in range(self.particle_count):
			sum_square = ti.Vector([0.0, 0.0, 0.0])
			square_sum = 0.0
			denominator = 0.0
			self.ps.for_all_neighbor(i, self.compute_sum, sum_square)
			self.ps.for_all_neighbor(i, self.compute_square_sum, square_sum)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				sum_square_boundary = ti.Vector([0.0, 0.0, 0.0])
				square_sum_boundary = 0.0
				self.ps.for_all_boundary_neighbor(i, self.compute_sum_boundary, sum_square_boundary)
				self.ps.for_all_boundary_neighbor(i, self.compute_square_sum_boundary, square_sum_boundary)
				denominator = (sum_square.dot(sum_square) + square_sum + square_sum_boundary + sum_square_boundary.dot(sum_square_boundary))
			else:
				denominator = (sum_square.dot(sum_square) + square_sum)
			if ti.abs(denominator) < 1e-6:
				self.alpha[i] = 0.0
			else:
				self.alpha[i] = self.rho[i] / denominator

	@ti.func
	def compute_sum(self, particle_i, particle_j):
		ret = ti.Vector([0.0, 0.0, 0.0])
		if particle_j.material == self.ps.material_fluid:
			q = particle_i.pos - particle_j.pos
			ret = self.ps.particle_m * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret

	@ti.func
	def compute_square_sum(self, particle_i, particle_j):
		ans = 0.0
		if particle_j.material == self.ps.material_fluid:
			q = particle_i.pos - particle_j.pos
			ret = self.ps.particle_m * self.cubic_kernel_derivative(q, self.kernel_h)
			ans = ret.dot(ret)
		return ans

	@ti.func
	def compute_sum_boundary(self, i, j):
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		ret = self.ps.boundary_particles.volume[j] * self.rho_0 * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret

	@ti.func
	def compute_square_sum_boundary(self, i, j):
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		ret = self.ps.boundary_particles.volume[j] * self.rho_0 * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret.dot(ret)

	@ti.kernel
	def compute_all_ext_force(self):
		self.solve_all_tension()
		self.solve_all_viscosity()
		for i in range(self.particle_count):
			self.force_ext[i] = self.gravity * ti.Vector([0, -1, 0]) + self.tension[i] + self.viscosity[i]

	@ti.kernel
	def compute_all_vel_adv(self):
		max_vel = -ti.math.inf
		for i in range(self.particle_count):
			self.vel_adv[i] = self.ps.fluid_particles.vel[i] + self.delta_time[None] * self.force_ext[i] / self.ps.particle_m
			ti.atomic_max(max_vel, self.vel_adv[i].norm())
			# self.delta_time[None] = 0.4 * self.ps.particle_radius * 2 / max_vel * 0.5
		if self.delta_time[None] > 0.4 * self.ps.particle_radius * 2 / max_vel:
			print('WARNING! DELTA TIME TOO SMALL. According to CLF condition, delta time must larger than {}'.format(0.4 * self.ps.particle_radius * 2/ max_vel))

	@ti.kernel
	def compute_all_rho_adv(self) -> ti.float32:
		rho_avg = 0.0
		for i in range(self.particle_count):
			delta = 0.0
			self.ps.for_all_neighbor(i, self.compute_rho_adv, delta)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				delta_boundary = 0.0
				self.ps.for_all_boundary_neighbor(i, self.compute_rho_adv_boundary, delta_boundary)
				self.rho_adv[i] = ti.max(self.rho[i] + self.delta_time[None] * (delta + delta_boundary * self.rho_0), self.rho_0)
			else:
				self.rho_adv[i] = ti.max(self.rho[i] + self.delta_time[None] * delta, self.rho_0)
			# self.rho_adv[i] = self.rho[i] + self.delta_time[None] * delta
			rho_avg += self.rho_adv[i]
		return rho_avg / self.particle_count

	@ti.func
	def compute_rho_adv(self, particle_i, particle_j):
		ret = 0.0
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			q = particle_i.pos - particle_j.pos
			kernel = self.cubic_kernel_derivative(q, self.kernel_h)
			ret = self.ps.particle_m * (self.vel_adv[i] - self.vel_adv[j]).dot(kernel)
		return ret

	@ti.func
	def compute_rho_adv_boundary(self, i, j):
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		return self.ps.boundary_particles.volume[j] * self.vel_adv[i].dot(self.cubic_kernel_derivative(q, self.kernel_h))

	@ti.kernel
	def iter_all_vel_adv(self):

		for i in range(self.particle_count):
			vel_adv = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.iter_vel_adv, vel_adv)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				vel_adv_boundary = ti.Vector([0.0, 0.0, 0.0])
				self.ps.for_all_boundary_neighbor(i, self.iter_vel_adv_boundary, vel_adv_boundary)
				self.vel_adv_delta[i] = vel_adv + vel_adv_boundary * self.rho_0
			else:
				self.vel_adv_delta[i] = vel_adv
		for i in range(self.particle_count):
			self.vel_adv[i] -= self.vel_adv_delta[i] * self.delta_time[None]

	@ti.func
	def iter_vel_adv(self, particle_i, particle_j):
		ret = ti.Vector([0.0, 0.0, 0.0])
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			k_i = (self.rho_adv[i] - self.rho_0) * self.alpha[i] / self.delta_time_2
			k_j = (self.rho_adv[j] - self.rho_0) * self.alpha[j] / self.delta_time_2
			q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
			ret = self.ps.particle_m * (k_i / self.rho[i] + k_j / self.rho[j]) * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret

	@ti.func
	def iter_vel_adv_boundary(self, i, j):
		k_i = (self.rho_adv[i] - self.rho_0) * self.alpha[i] / self.delta_time_2
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		return self.ps.boundary_particles.volume[j] * k_i / self.rho[i] * self.cubic_kernel_derivative(q, self.kernel_h)

	def correct_density_error(self):
		rho_avg = ti.math.inf
		iter_cnt = 0

		while iter_cnt < self.min_iteration_density or rho_avg - self.rho_0 > self.density_threshold * self.rho_0 * 0.01:

			rho_avg = self.compute_all_rho_adv()

			self.iter_all_vel_adv()

			iter_cnt += 1

		print('[density iteration] count: {}, error {}'.format(iter_cnt, rho_avg - self.rho_0))

	@ti.kernel
	def compute_all_position(self):
		for i in range(self.particle_count):
			self.ps.fluid_particles.pos[i] = self.ps.fluid_particles.pos[i] + self.delta_time[None] * self.vel_adv[i]
			self.ps.fluid_particles.vel[i] = self.vel_adv[i]

		if self.boundary_handle == self.clamp_boundary_handle:
			for i in range(self.particle_count):
				for j in ti.static(range(3)):
					if self.ps.fluid_particles.pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
						self.ps.fluid_particles.pos[i][j] = self.ps.box_min[j] + self.ps.particle_radius
						self.ps.fluid_particles.vel[i][j] *= -self.v_decay_proportion

					if self.ps.fluid_particles.pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
						self.ps.fluid_particles.pos[i][j] = self.ps.box_max[j] - self.ps.particle_radius
						self.ps.fluid_particles.vel[i][j] *= -self.v_decay_proportion

	@ti.kernel
	def derivative_iter_all_rho(self) -> ti.float32:
		avg = 0.0
		for i in range(self.particle_count):
			neighbor_count = self.ps.get_neighbour_count(i)
			if neighbor_count < 20:
				self.rho_derivative[i] = 0.0
				continue
			rho_derivative = 0.0
			self.ps.for_all_neighbor(i, self.compute_rho_derivative, rho_derivative)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				rho_derivative_boundary = 0.0
				self.ps.for_all_boundary_neighbor(i, self.compute_rho_derivative_boundary, rho_derivative_boundary)
				self.rho_derivative[i] = ti.max(rho_derivative + rho_derivative_boundary * self.rho_0, 0.0)
			else:
				self.rho_derivative[i] = ti.max(rho_derivative, 0.0)
			# self.rho_derivative[i] = -self.rho[i] * rho_derivative
			avg += self.rho_derivative[i]
		return avg / self.particle_count

	@ti.func
	def compute_rho_derivative(self, particle_i, particle_j):
		ret = 0.0
		if particle_j.material == self.ps.material_fluid:
			q = particle_i.pos - particle_j.pos
			ret = self.ps.particle_m * (particle_i.vel - particle_j.vel).dot(self.cubic_kernel_derivative(q, self.kernel_h))
		return ret

	@ti.func
	def compute_rho_derivative_boundary(self, i, j):
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		return self.ps.boundary_particles.volume[j] * self.ps.fluid_particles.vel[i].dot(self.cubic_kernel_derivative(q, self.kernel_h))

	@ti.kernel
	def divergence_iter_all_vel_adv(self):
		for i in range(self.particle_count):
			vel_adv = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.divergence_iter_vel_adv, vel_adv)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				vel_adv_boundary = ti.Vector([0.0, 0.0, 0.0])
				self.ps.for_all_boundary_neighbor(i, self.divergence_iter_vel_adv_boundary, vel_adv_boundary)
				self.ps.fluid_particles.vel[i] -= (vel_adv + vel_adv_boundary * self.rho_0) * self.delta_time[None]
			else:
				self.ps.fluid_particles.vel[i] -= vel_adv * self.delta_time[None]

	@ti.func
	def divergence_iter_vel_adv(self, particle_i, particle_j):
		ret = ti.Vector([0.0, 0.0, 0.0])
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			k_i = self.rho_derivative[i] * self.alpha[i] / self.delta_time[None]
			k_j = self.rho_derivative[j] * self.alpha[j] / self.delta_time[None]
			q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
			kernel = self.cubic_kernel_derivative(q, self.kernel_h)
			ret = self.ps.particle_m * (k_i / self.rho[i] + k_j / self.rho[j]) * kernel
		return ret

	@ti.func
	def divergence_iter_vel_adv_boundary(self, i, j):
		k_i = self.rho_derivative[i] * self.alpha[i] / self.delta_time[None]
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		ret = self.ps.boundary_particles.volume[j] * k_i / self.rho[i] * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret

	def correct_divergence_error(self):
		rho_divergence_avg = ti.math.inf
		iter_cnt = 0

		while (iter_cnt < self.min_iteration_density_divergence or rho_divergence_avg > self.density_divergence_threshold) and iter_cnt < self.max_iteration_density_divergence:

			rho_divergence_avg = self.derivative_iter_all_rho()

			self.divergence_iter_all_vel_adv()

			iter_cnt += 1

		print('[divergence iteration] count: {}, error {}'.format(iter_cnt, rho_divergence_avg))

	@ti.kernel
	def initialize(self):
		self.compute_all_rho()
		self.compute_all_alpha()

	def iterate(self):

		self.correct_divergence_error()

		self.compute_all_ext_force()

		self.compute_all_vel_adv()

		self.correct_density_error()

		self.compute_all_position()

	def step(self):
		super(dfsph_solver, self).step()

		self.initialize()

		self.iterate()
