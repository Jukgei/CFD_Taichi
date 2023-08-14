import taichi as ti
from solver_base import solver_base

class dfsph_solver(solver_base):

	def __init__(self, particle_system, config):
		super(dfsph_solver, self).__init__(particle_system, config)

		self.alpha = ti.field(ti.float32, shape=self.particle_count)
		self.rho_derivative = ti.field(ti.float32, shape=self.particle_count)
		self.vel_adv = ti.Vector.field(3, ti.float33, shape=self.particle_count)
		self.rho_adv = ti.Vector.field(3, ti.float33, shape=self.particle_count)
		self.force_ext = ti.Vector.field(3, ti.float32, shape=self.particle_count)
		self.force_ext.fill(self.gravity * ti.Vector([0.0, -1.0, 0.0]) * self.ps.particle_m)

		self.delta_time_2 = self.delta_time ** 2
		self.min_iteration_density = 2
		self.density_threshold = 0.1
		self.min_iteration_density_divergence = 1
		self.density_divergence_threshold = 1

	@ti.func
	def compute_all_alpha(self):
		for i in range(self.particle_count):
			sum_square = ti.Vector([0.0, 0.0, 0.0])
			square_sum = 0.0
			self.ps.for_all_neighbor(i, self.compute_sum, sum_square)
			self.ps.for_all_neighbor(i, self.compute_square_sum, square_sum)
			self.alpha[i] = sum_square.dot(sum_square) + square_sum

	@ti.func
	def compute_sum(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		ret = self.ps.particle_m * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret

	@ti.func
	def compute_square_sum(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		ret = self.ps.particle_m * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret.dot(ret)

	@ti.kernel
	def compute_all_ext_force(self):
		# TODO: viscosity and surface tension
		pass

	@ti.func
	def compute_all_vel_adv(self):
		for i in range(self.particle_count):
			self.vel_adv[i] = self.ps.vel[i] + self.delta_time * self.force_ext[i] / self.ps.particle_m

	@ti.kernel
	def compute_all_rho_adv(self) -> ti.float32:
		rho_avg = 0.0
		for i in range(self.particle_count):
			rho_adv = 0
			self.ps.for_all_neighbor(i, self.compute_rho_adv, rho_adv)
			self.rho_adv[i] = self.rho[i] + self.delta_time * rho_adv
			rho_avg += self.rho_adv[i]
		return rho_avg / self.particle_count

	@ti.func
	def compute_rho_adv(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		return self.ps.particle_m * (self.vel_adv[i] - self.vel_adv[j]).dot(self.cubic_kernel_derivative(q, self.kernel_h))

	@ti.kernel
	def iter_all_vel_adv(self):

		for i in range(self.particle_count):

			vel_adv = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.iter_vel_adv, vel_adv)
			self.vel_adv[i] -= vel_adv * self.delta_time

	@ti.func
	def iter_vel_adv(self, i, j):
		# todo camp to zero?
		k_i = (self.rho_adv[i] - self.rho_0) * self.alpha[i] / self.delta_time_2
		k_j = (self.rho_adv[j] - self.rho_0) * self.alpha[j] / self.delta_time_2
		q = self.ps.pos[i] - self.ps.pos[j]
		ret = self.ps.particle_m * (k_i / self.rho[i] + k_j / self.rho[j]) * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret

	def correct_density_error(self):
		rho_avg = ti.math.inf
		iter_cnt = 0

		while iter_cnt < 0 or rho_avg - self.rho_0 > self.density_threshold * self.rho_0 * 0.01:

			rho_avg = self.compute_all_rho_adv()

			self.iter_all_vel_adv()

			iter_cnt += 1

	@ti.kernel
	def compute_all_position(self):
		for i in range(self.particle_count):
			self.ps.pos[i] = self.ps.pos[i] + self.delta_time * self.vel_adv[i]

		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.ps.pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_min[j] + self.ps.particle_radius
					self.vel_adv[i][j] *= -self.v_decay_proportion

				if self.ps.pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_max[j] - self.ps.particle_radius
					self.vel_adv[i][j] *= -self.v_decay_proportion

	@ti.kernel
	def update_density(self):
		self.compute_all_rho()

	@ti.kernel
	def update_alpha(self):
		self.compute_all_alpha()

	@ti.kernel
	def iter_all_rho_derivative(self) -> ti.float32:
		avg = 0.0
		for i in range(self.particle_count):
			rho_derivative = 0.0
			self.ps.for_all_neighbor(i, self.compute_rho_derivative, rho_derivative)
			self.rho_derivative[i] = rho_derivative
			avg += rho_derivative
		return avg / self.particle_count

	@ti.kernel
	def iter_all_vel_adv_about_divergence(self):
		for i in range(self.particle_count):

			vel_adv = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.iter_vel_adv_about_divergence, vel_adv)
			self.vel_adv[i] -= vel_adv * self.delta_time

	@ti.func
	def iter_vel_adv_about_divergence(self, i, j):
		# todo camp to zero?
		k_i = (self.rho_derivative[i] * self.alpha[i] / self.delta_time)
		k_j = (self.rho_derivative[j] * self.alpha[j] / self.delta_time)
		q = self.ps.pos[i] - self.ps.pos[j]
		ret = self.ps.particle_m * (k_i / self.rho[i] + k_j / self.rho[j]) * self.cubic_kernel_derivative(q,
																										  self.kernel_h)
		return ret

	def correct_divergence_error(self):
		rho_divergence_avg = ti.math.inf
		iter_cnt = 0

		while iter_cnt < 0 or rho_divergence_avg > self.density_divergence_threshold:

			rho_divergence_avg = self.iter_all_rho_derivative()

			self.iter_all_vel_adv_about_divergence()

			iter_cnt += 1

	@ti.kernel
	def update_velocity(self):
		for i in range(self.particle_count):
			self.ps.vel[i] = self.vel_adv[i]

	@ti.kernel
	def initialize(self):
		self.compute_all_rho()
		self.compute_all_alpha()

	def iterate(self):
		self.compute_all_ext_force()

		self.compute_all_vel_adv()

		self.correct_density_error()

		self.compute_all_position()

		self.update_density()

		self.update_alpha()

		self.correct_divergence_error()

		self.update_velocity()

	def step(self):
		self.initialize()

		self.iterate()