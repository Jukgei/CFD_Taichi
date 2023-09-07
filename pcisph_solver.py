import taichi as ti
from solver_base import solver_base


class pcisph_solver(solver_base):

	def __init__(self, particle_system, config):
		super(pcisph_solver, self).__init__(particle_system, config)
		self.pos_predict = ti.Vector.field(3, ti.f32, shape=self.ps.particle_num)
		self.vel_predict = ti.Vector.field(3, ti.f32, shape=self.ps.particle_num)
		self.ext_force = ti.Vector.field(3, ti.f32, shape=self.ps.particle_num)
		self.ext_force.fill(self.gravity * ti.Vector([0, -1, 0]))
		self.press_force = ti.Vector.field(3, ti.f32, shape=self.ps.particle_num)
		self.rho_predict = ti.field(ti.float32, shape=self.ps.particle_num)
		self.rho_err = ti.field(ti.float32, shape=self.ps.particle_num)
		self.press_iter = ti.field(ti.float32, shape=self.ps.particle_num)
		self.tension = ti.Vector.field(n=3, dtype=ti.float32, shape=self.ps.particle_num)

		self.rho_max_err_percent = .1
		self.min_iteration = 1
		self.max_iteration = 80

		self.beta = self.delta_time[None] * self.delta_time[None] * self.ps.particle_m * self.ps.particle_m * 2 / (self.rho_0 ** 2)
		self.delta = ti.field(ti.float32, shape=())

		self.pre_compute()

	def pre_compute(self):
		self.ps.reset_grid()

		self.ps.update_grid()

		max_index = self.ps.get_max_neighbor_particle_index()

		self.pre_compute_delta(max_index)

		print('PCISPH parameter delta: {}, beta: {}'.format(self.delta[None], self.beta))

	@ti.kernel
	def pre_compute_delta(self, full_neighbor_index: ti.int32):
		sum = ti.Vector([0.0, 0.0, 0.0])
		square_sum = 0.0
		self.ps.for_all_neighbor(full_neighbor_index, self.compute_sum, sum)
		self.ps.for_all_neighbor(full_neighbor_index, self.compute_square_sum, square_sum)
		self.delta[None] = 1 / ((sum.dot(sum) + square_sum) * self.beta)

	def iteration(self):

		iter_cnt = 0
		self.predict_vel_pos()

		self.predict_rho()

		rho_err_avg = self.compute_residual()

		while (rho_err_avg > self.rho_0 * self.rho_max_err_percent * 0.01 or iter_cnt < self.min_iteration) and iter_cnt < self.max_iteration:

			self.iter_press()

			self.update_press_force()

			self.predict_vel_pos()

			self.predict_rho()

			# error
			rho_err_avg = self.compute_residual()
			iter_cnt += 1
			# print('		Iter cnt: {}, error: {}'.format(iter_cnt, rho_err_avg))

	@ti.kernel
	def predict_vel_pos(self):
		for i in range(self.particle_count):
			self.vel_predict[i] = self.ps.fluid_particles.vel[i] + self.delta_time[None] * (self.ext_force[i] + self.press_force[i]) / self.ps.particle_m
			self.pos_predict[i] = self.ps.fluid_particles.pos[i] + self.delta_time[None] * self.vel_predict[i]

		if self.boundary_handle == self.clamp_boundary_handle:
			for i in range(self.particle_count):
				for j in ti.static(range(3)):
					if self.pos_predict[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
						self.pos_predict[i][j] = self.ps.box_min[j] + self.ps.particle_radius
						self.vel_predict[i][j] *= -self.v_decay_proportion

					if self.pos_predict[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
						self.pos_predict[i][j] = self.ps.box_max[j] - self.ps.particle_radius
						self.vel_predict[i][j] *= -self.v_decay_proportion

	@ti.kernel
	def predict_rho(self):
		for i in range(self.particle_count):
			rho_predict = 0.0

			self.ps.for_all_neighbor(i, self.compute_rho_predict, rho_predict)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				rho_boundary = 0.0
				self.ps.for_all_boundary_neighbor(i, self.compute_rho_boundary_predict, rho_boundary)
				self.rho_predict[i] = rho_predict + rho_boundary * self.rho_0
			else:
				self.rho_predict[i] = rho_predict
			self.rho_err[i] = self.rho_predict[i] - self.rho_0

	@ti.kernel
	def iter_press(self):
		for i in range(self.particle_count):
			self.press_iter[i] += self.rho_err[i] * self.delta[None]
			self.press_iter[i] = ti.max(0.0, self.press_iter[i])

	@ti.kernel
	def update_press_force(self):
		for i in range(self.particle_count):
			press_force = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_press_force, press_force)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				boundary_acc = ti.Vector([0.0, 0.0, 0.0])
				self.ps.for_all_boundary_neighbor(i, self.compute_boundary_pressure, boundary_acc)
				self.press_force[i] = - press_force + boundary_acc * self.rho_0 * self.ps.particle_m
			else:
				self.press_force[i] = - press_force

	@ti.kernel
	def compute_residual(self) -> ti.f32:
		rho_err_sum = 0.0
		for i in range(self.particle_count):
			rho_err_sum += self.rho_err[i]
		return rho_err_sum / self.particle_count

	@ti.func
	def compute_rho_predict(self, particle_i, particle_j):
		w_ij = 0.0
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			q = (self.pos_predict[i] - self.pos_predict[j]).norm()
			w_ij = self.cubic_kernel(q, self.kernel_h) * self.ps.particle_m
		elif particle_j.material == self.ps.material_solid:
			if self.boundary_handle == self.akinci2012_boundary_handle:
				i = particle_i.index
				q = (self.pos_predict[i] - particle_j.pos).norm()
				w_ij = self.cubic_kernel(q, self.kernel_h) * particle_j.volume * self.rho_0
		return w_ij

	@ti.func
	def compute_rho_boundary_predict(self, i, j):
		q = (self.pos_predict[i] - self.ps.boundary_particles.pos[j]).norm()
		w_ij = self.cubic_kernel(q, self.kernel_h) * self.ps.boundary_particles.volume[j]
		return w_ij

	@ti.func
	def compute_sum(self, particle_i, particle_j):
		q = particle_i.pos - particle_j.pos
		dw_ij = self.cubic_kernel_derivative(q, self.kernel_h)
		return dw_ij

	@ti.func
	def compute_square_sum(self, particle_i, particle_j):
		q = particle_i.pos - particle_j.pos
		dw_ij = self.cubic_kernel_derivative(q, self.kernel_h)
		dw_ij_2 = dw_ij.dot(dw_ij)
		return dw_ij_2

	@ti.func
	def compute_press_force(self, particle_i, particle_j):
		ret = ti.Vector([0.0, 0.0, 0.0])
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
			dw_ij = self.cubic_kernel_derivative(q, self.kernel_h)
			ret = (self.press_iter[i] + self.press_iter[j]) * dw_ij / (self.rho_0 ** 2) * self.ps.particle_m * self.ps.particle_m
		elif particle_j.material == self.ps.material_solid:
			if self.boundary_handle == self.akinci2012_boundary_handle:
				i = particle_i.index
				j = particle_j.index
				q = particle_i.pos - particle_j.pos
				rho_i = self.rho[i]
				kernel = self.cubic_kernel_derivative(q, self.kernel_h)
				ret = self.ps.particle_m * particle_j.volume * self.rho_0 * self.press_iter[i] * kernel / (rho_i ** 2)
				self.ps.rigid_particles[j].force += ret
		return ret

	@ti.func
	def compute_boundary_pressure(self, i, j):
		ret = ti.Vector([0.0, 0.0, 0.0])
		p_i = self.press_iter[i]
		rho_i = self.rho[i]
		rho_i_2 = rho_i ** 2
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		ret -= self.ps.boundary_particles.volume[j] * p_i / rho_i_2 * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret

	@ti.kernel
	def integration(self):
		for i in range(self.particle_count):
			self.ps.fluid_particles.vel[i] = self.ps.fluid_particles.vel[i] + self.delta_time[None] * (
						self.ext_force[i] + self.press_force[i]) / self.ps.particle_m
			self.ps.fluid_particles.pos[i] = self.ps.fluid_particles.pos[i] + self.delta_time[None] * self.ps.fluid_particles.vel[i]

		if self.boundary_handle == self.clamp_boundary_handle:
			for i in range(self.particle_count):
				for j in ti.static(range(3)):
					if self.ps.fluid_particles.pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
						self.ps.fluid_particles.pos[i][j] = self.ps.box_min[j] + self.ps.particle_radius
						self.ps.fluid_particles.vel[i][j] *= -self.v_decay_proportion

					if self.ps.fluid_particles.pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
						self.ps.fluid_particles.pos[i][j] = self.ps.box_max[j] - self.ps.particle_radius
						self.ps.fluid_particles.vel[i][j] *= -self.v_decay_proportion
		# self.check_valid()

	@ti.kernel
	def compute_ext_force(self):
		self.compute_all_rho()
		self.solve_all_tension()
		self.solve_all_viscosity()
		for i in range(self.particle_count):
			self.ext_force[i] = self.gravity * ti.Vector([0, -1, 0]) + self.tension[i] + self.viscosity[i]

	@ti.kernel
	def reset(self):
		self.press_iter.fill(0)
		self.press_force.fill(ti.Vector([0.0, 0.0, 0.0]))

	def step(self):
		super(pcisph_solver, self).step()

		self.compute_ext_force()

		self.iteration()

		self.integration()
