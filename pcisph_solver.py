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

		self.rho_max_err_percent = 1
		self.min_iteration = 1
		self.max_iteration = 500

		self.beta = self.delta_time * self.delta_time * self.ps.particle_m * self.ps.particle_m * 2 / self.rho_0
		self.delta = ti.field(ti.float32, shape=())

		self.pre_compute()
		self.delta[None] = 100000
		# self.delta[None] = 1

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
		# self.ps.for_all_neighbor(i, self.compute_press, press_iter)
		self.ps.for_all_neighbor(full_neighbor_index, self.compute_sum, sum)
		self.ps.for_all_neighbor(full_neighbor_index, self.compute_square_sum, square_sum)
		self.delta[None] = 1 / ((sum.dot(sum) + square_sum) * self.beta)
		print('sum {}'.format(sum))



	# @ti.kernel
	def iteration(self):
		rho_err_avg = ti.math.inf
		iter_cnt = 0
		self.predict_vel_pos()

		self.predict_rho()

		while (rho_err_avg > self.rho_0 * self.rho_max_err_percent * 0.01 or iter_cnt < self.min_iteration) and iter_cnt < self.max_iteration:

			self.iter_press()

			self.update_press_force()

			self.predict_vel_pos()

			self.predict_rho()

			# error
			rho_err_avg = self.compute_residual()
			iter_cnt += 1
			print('		Iter cnt: {}, error: {}'.format(iter_cnt, rho_err_avg))

	@ti.kernel
	def predict_vel_pos(self):
		for i in range(self.particle_count):
			self.vel_predict[i] = self.ps.vel[i] + self.delta_time * (self.ext_force[i] + self.press_force[i]) / self.ps.particle_m
			self.pos_predict[i] = self.ps.pos[i] + self.delta_time * self.vel_predict[i]
			#TODO BUG
			# delta_x = self.pos_predict[i] - self.ps.pos[i]
			# print('delta_x: {}, accuracy {}'.format(delta_x,  (self.delta_time ** 2) * ( self.ext_force[i] + self.press_force[i]) / self.ps.particle_m - delta_x))

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
			# rho_predict = self.cubic_kernel(0.0, self.kernel_h)
			rho_predict = 0.0
			# delta_rho_predict = 0.0
			# self.ps.for_all_neighbor(i, self.compute_delta_rho_predict, delta_rho_predict)
			self.ps.for_all_neighbor(i, self.compute_rho_predict, rho_predict)
			self.rho_predict[i] = rho_predict * self.ps.particle_m
			print('rho_predict[{}]:{}'.format(i, rho_predict))
			self.rho_err[i] = self.rho_predict[i] - self.rho_0
			self.rho_err[i] = ti.max(self.rho_predict[i] - self.rho_0, 0.0)

	@ti.kernel
	def iter_press(self):
		for i in range(self.particle_count):
			# sum = ti.Vector([0.0, 0.0, 0.0])
			# square_sum = 0.0
			# # self.ps.for_all_neighbor(i, self.compute_press, press_iter)
			# self.ps.for_all_neighbor(i, self.compute_sum_pre, sum)
			# self.ps.for_all_neighbor(i, self.compute_square_sum_pre, square_sum)
			# # TODO: numberical problem self.beta is less than 1e-9
			# press_iter = - self.rho_err[i] / ((- sum.dot(sum) - square_sum) * self.beta)

			self.press_iter[i] += self.rho_err[i] * self.delta[None]
			self.press_iter[i] = ti.max(0.0, self.press_iter[i])
			# if i == 0:
			# 	print('press_iter {}'.format(- sum.dot(sum) - square_sum))

	@ti.kernel
	def update_press_force(self):
		for i in range(self.particle_count):
			press_force = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_press_force, press_force)
			self.press_force[i] = - press_force * self.ps.particle_m * self.ps.particle_m

			# if i == 0:
			# 	print('press force[0] {}'.format(press_force))

	@ti.kernel
	def compute_residual(self) -> ti.f32:
		rho_err_sum = 0.0
		cnt = 0
		avg = 0.0
		for i in range(self.particle_count):
			# if self.press_iter[i] > 0:

			# rho_err_sum += ti.abs(self.rho_err[i])
			rho_err_sum += self.rho_err[i]
			cnt += 1
			# print('rho_err[{}]: {}'.format(i, self.rho_err[i]))
		# avg = 0.0
		# if cnt > 0:
		avg = rho_err_sum / cnt
		# else:
		# 	avg = 0.0
		return avg

	@ti.func
	def compute_rho_predict(self, i, j):
		q = (self.pos_predict[i] - self.pos_predict[j]).norm()
		w_ij = self.cubic_kernel(q, self.kernel_h)
		return w_ij

	@ti.func
	def compute_sum(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		dw_ij = self.cubic_kernel_derivative(q, self.kernel_h)
		# print('i: {}, j:{}, q: {}, dw_ij: {}'.format(i, j, q.norm(), dw_ij))
		return dw_ij

	@ti.func
	def compute_square_sum(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		dw_ij = self.cubic_kernel_derivative(q, self.kernel_h)
		dw_ij_2 = dw_ij.dot(dw_ij)
		return dw_ij_2

	@ti.func
	def compute_press_force(self, i, j):
		q = self.ps.pos[i] - self.ps.pos[j]
		dw_ij = self.cubic_kernel_derivative(q, self.kernel_h)
		# TODO: why use rho_0
		return (self.press_iter[i] + self.press_iter[j]) * dw_ij / (self.rho_0 ** 2)

	@ti.kernel
	def integration(self):
		for i in range(self.particle_count):
			self.ps.vel[i] = self.ps.vel[i] + self.delta_time * (
						self.ext_force[i] + self.press_force[i]) / self.ps.particle_m
			self.ps.pos[i] = self.ps.pos[i] + self.delta_time * self.ps.vel[i]

		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.ps.pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_min[j] + self.ps.particle_radius
					self.ps.vel[i][j] *= -self.v_decay_proportion

				if self.ps.pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_max[j] - self.ps.particle_radius
					self.ps.vel[i][j] *= -self.v_decay_proportion
		self.check_valid()

	@ti.kernel
	def compute_ext_force(self):
		# todo: add surface tension and viscosity
		pass

	@ti.kernel
	def reset(self):
		self.press_iter.fill(0)
		self.press_force.fill(ti.Vector([0.0, 0.0, 0.0]))


	def step(self):
		super(pcisph_solver, self).step()

		self.compute_ext_force()

		self.iteration()

		self.integration()
