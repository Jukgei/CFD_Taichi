import taichi as ti
from solver_base import solver_base


class iisph_solver(solver_base):

	def __init__(self, particle_system, config):
		super(iisph_solver, self).__init__(particle_system, config)

		self.v_adv = ti.Vector.field(3, dtype=ti.float32, shape=self.particle_count)
		self.f_adv = ti.Vector.field(3, dtype=ti.float32, shape=self.particle_count)  # include gravity, surface tension, viscosity
		self.d_ii = ti.Vector.field(3, dtype=ti.float32, shape=self.particle_count)
		self.a_ii = ti.field(ti.float32, shape=self.particle_count)
		self.d_ij = ti.Vector.field(3, dtype=ti.float32, shape=self.particle_count)

		self.f_adv.fill(9.8 * ti.Vector([0.0, -1.0, 0.0]) * self.ps.particle_m)
		self.rho_adv = ti.field(ti.float32, shape=self.particle_count)
		self.rho_iter = ti.field(ti.float32, shape=self.particle_count)
		self.p_iter = ti.field(ti.float32, shape=self.particle_count)
		self.p_past = ti.field(ti.float32, shape=self.particle_count)
		self.r_sum = ti.field(ti.float32, shape=self.particle_count)
		self.f_press = ti.Vector.field(3, dtype=ti.float32, shape=self.particle_count)

		self.omega = 0.5

	@ti.kernel
	def reset(self):
		# self.ps.acc.fill(9.8 * ti.Vector([0.0, -1.0, 0.0]))
		# self.f_adv.fill(9.8 * ti.Vector([0.0, -1.0, 0.0]) * self.ps.particle_m)
		pass

	@ti.kernel
	def predict_advection(self):
		self.compute_all_rho()
		# rho_avg = 0.0
		# for i in range(self.particle_count):
		# 	rho_avg += self.rho[i]
		# print("rho avg ", rho_avg / self.particle_count)
		for i in range(self.particle_count):
			self.v_adv[i] = self.ps.vel[i] + self.delta_time * self.f_adv[i] / self.ps.particle_m
			d_ii = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_d_ii, d_ii)
			self.d_ii[i] = d_ii * self.delta_time * self.delta_time

		for i in range(self.particle_count):
			rho_adv = 0.0
			self.ps.for_all_neighbor(i, self.compute_rho_adv, rho_adv)
			# self.rho_adv[i] = ti.max(rho_adv * self.delta_time + self.rho[i], self.rho[i])
			self.rho_adv[i] = rho_adv * self.delta_time + self.rho[i]
			self.p_iter[i] = 0.5 * self.p_past[i]
			a_ii = 0.0
			self.ps.for_all_neighbor(i, self.compute_a_ii, a_ii)
			self.a_ii[i] = a_ii

	# @ti.kernel
	def pressure_solve(self):
		l = 0
		# rho_avg = self.rho.sum() / self.particle_count
		# rho_avg = 0.0
		# self.compute_all_rho_adv()
		# for i in range(self.particle_count):
		# 	rho_avg += self.rho_iter[i]
		# rho_avg /= self.particle_count
		# print("Iter 0 ", rho_avg)
		# while ti.abs(rho_avg - self.rho_0) > self.rho_0 * 0.05:# or l < 2:
		residual = 10000
		while residual > 1 and l < 500:

			self.compute_all_d_ij()

			self.update_p()

			l += 1

			residual = self.compute_residual()
			# rho_avg = 0.0
			# self.compute_all_rho_adv()
			# for i in range(self.particle_count):
			# 	rho_avg += self.rho_iter[i]
			# rho_avg /= self.particle_count
			# # print("Iter 0 ", rho_avg)
			# print("l is ", l, rho_avg)

		print("Iter cnt: ", l, residual)

	@ti.kernel
	def compute_residual(self) -> ti.f32:
		residual = 0.0
		for i in range(self.particle_count):
			residual += (self.a_ii[i] * self.p_iter[i] + self.r_sum[i] + self.rho_adv[i] - 1000) ** 2

		# print('residual: ', residual / self.particle_count)
		return residual / self.particle_count

	@ti.kernel
	def compute_all_rho_adv(self):
		self.compute_all_press_force()
		self.compute_all_rho_iter()

	@ti.func
	def compute_all_rho_iter(self):
		for i in range(self.particle_count):
			rho_iter = 0.0
			self.ps.for_all_neighbor(i, self.compute_rho_iter, rho_iter)
			self.rho_iter[i] = self.delta_time * self.delta_time * rho_iter + self.rho_adv[i]

	@ti.func
	def compute_rho_iter(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		# todo dot?
		return self.ps.particle_m * (self.f_press[i] / self.ps.particle_m - self.f_press[j] / self.ps.particle_m).dot(self.cubic_kernel_derivative(q, self.kernel_h))

	@ti.kernel
	def compute_all_d_ij(self):
		for i in range(self.particle_count):
			d_ij = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_d_ij, d_ij)
			self.d_ij[i] = d_ij * self.delta_time * self.delta_time

	@ti.kernel
	def update_p(self):
		for i in range(self.particle_count):
			sum = 0.0
			self.ps.for_all_neighbor(i, self.sum_factor, sum)
			self.r_sum[i] = sum
			self.p_iter[i] = (1 - self.omega) * self.p_iter[i] + self.omega * (self.rho_0 - self.rho_adv[i] - sum) / (self.a_ii[i])

	@ti.func
	def compute_all_press_force(self):

		for i in range(self.particle_count):
			f_press = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_press_force, f_press)
			self.f_press[i] = - f_press * self.ps.particle_m

	@ti.kernel
	def intergation(self):
		self.compute_all_press_force()

		for i in range(self.particle_count):
			self.ps.vel[i] = self.v_adv[i] + self.delta_time * self.f_press[i] / self.ps.particle_m
			self.ps.pos[i] = self.ps.pos[i] + self.delta_time * self.ps.vel[i]

		self.check_valid()
		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.ps.pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_min[j] + self.ps.particle_radius
					self.ps.vel[i][j] *= -self.v_decay_proportion

				if self.ps.pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_max[j] - self.ps.particle_radius
					self.ps.vel[i][j] *= -self.v_decay_proportion

		for i in range(self.particle_count):
			self.p_past[i] = self.p_iter[i]

	@ti.func
	def compute_press_force(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]

		return self.ps.particle_m * (self.p_iter[i] / ti.pow(self.rho[i], 2) + self.p_iter[j] / ti.pow(self.rho[j], 2)) * self.cubic_kernel_derivative(q, self.kernel_h)

	@ti.func
	def sum_factor(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		w_ij = self.cubic_kernel_derivative(q, self.kernel_h)
		w_ji = self.cubic_kernel_derivative(-q, self.kernel_h)
		d_ji = - self.delta_time * self.delta_time * self.ps.particle_m / (
				self.rho[i] * self.rho[i]) * w_ji * self.p_iter[i]
		return self.ps.particle_m * (self.d_ij[i] - self.d_ii[j] * self.p_iter[j] - (self.d_ij[j] - d_ji)).dot(w_ij)


	@ti.func
	def compute_d_ii(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		return - self.ps.particle_m / (self.rho[i] * self.rho[i]) * self.cubic_kernel_derivative(q, self.kernel_h)

	@ti.func
	def compute_a_ii(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		cubic_kernel_derivative = self.cubic_kernel_derivative(q, self.kernel_h)
		# todo  check is rho[i] ? or rho[j]?
		d_ji = - self.delta_time * self.delta_time * self.ps.particle_m / (
					self.rho[i] * self.rho[i]) * self.cubic_kernel_derivative(-q, self.kernel_h)
		return self.ps.particle_m * (self.d_ii[i] - d_ji).dot(cubic_kernel_derivative)

	@ti.func
	def compute_d_ij(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		w_ij = self.cubic_kernel_derivative(q, self.kernel_h)
		return - self.ps.particle_m * self.p_iter[j] * w_ij / (self.rho[j] * self.rho[j])

	@ti.func
	def compute_rho_adv(self, i, j):
		v_ij = self.v_adv[i] - self.v_adv[j]
		q = self.ps.pos[i] - self.ps.pos[j]
		#TODO dot?
		return self.ps.particle_m * v_ij.dot(self.cubic_kernel_derivative(q, self.kernel_h))

	def step(self):
		super(iisph_solver, self).step()

		self.predict_advection()

		self.pressure_solve()

		self.intergation()
