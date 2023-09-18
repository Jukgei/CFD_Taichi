import taichi as ti
from solver_base import solver_base


class iisph_solver(solver_base):

	def __init__(self, particle_system, config):
		super(iisph_solver, self).__init__(particle_system, config)

		self.v_adv = ti.Vector.field(3, dtype=ti.float32, shape=self.particle_count)
		self.f_adv = ti.Vector.field(3, dtype=ti.float32,
									 shape=self.particle_count)  # include gravity, surface tension, viscosity
		self.d_ii = ti.Vector.field(3, dtype=ti.float32, shape=self.particle_count)
		self.a_ii = ti.field(ti.float32, shape=self.particle_count)
		self.d_ij = ti.Vector.field(3, dtype=ti.float32, shape=self.particle_count)

		self.f_adv.fill(self.gravity * ti.Vector([0.0, -1.0, 0.0]) * self.ps.particle_m)
		self.rho_adv = ti.field(ti.float32, shape=self.particle_count)
		self.rho_iter = ti.field(ti.float32, shape=self.particle_count)
		self.p_iter = ti.field(ti.float32, shape=self.particle_count)
		self.p_past = ti.field(ti.float32, shape=self.particle_count)
		self.p_new_buff = ti.field(ti.float32, shape=self.particle_count)
		self.r_sum = ti.field(ti.float32, shape=self.particle_count)
		self.f_press = ti.Vector.field(3, dtype=ti.float32, shape=self.particle_count)

		self.omega = 0.5
		self.max_iter_cnt = 180
		self.min_iter_cnt = 1
		self.rho_err_percent = .1

	@ti.kernel
	def reset(self):
		pass

	@ti.kernel
	def predict_advection(self):
		self.compute_all_rho()
		# rho_avg = 0.0
		# for i in range(self.particle_count):
		# 	rho_avg += self.rho[i]
		# print("rho avg ", rho_avg / self.particle_count)
		self.solve_all_tension()
		self.solve_all_viscosity()
		for i in range(self.particle_count):
			self.f_adv[i] = self.gravity * ti.Vector([0, -1, 0]) + self.tension[i] + self.viscosity[i]
		for i in range(self.particle_count):
			self.v_adv[i] = self.ps.fluid_particles.vel[i] + self.delta_time[None] * self.f_adv[i] / self.ps.particle_m
			d_ii = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_d_ii, d_ii)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				d_ii_boundary = ti.Vector([0.0, 0.0, 0.0])
				self.ps.for_all_boundary_neighbor(i, self.compute_boundary_d_ii, d_ii_boundary)
				self.d_ii[i] = (d_ii + d_ii_boundary * self.rho_0) * self.delta_time[None] * self.delta_time[None]
			else:
				self.d_ii[i] = d_ii * self.delta_time[None] * self.delta_time[None]

		for i in range(self.particle_count):
			rho_adv = 0.0
			self.ps.for_all_neighbor(i, self.compute_rho_adv, rho_adv)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				rho_adv_boundary = 0.0
				self.ps.for_all_boundary_neighbor(i, self.compute_rho_adv_boundary, rho_adv_boundary)
				self.rho_adv[i] = (rho_adv + rho_adv_boundary * self.rho_0) * self.delta_time[None] + self.rho[i]
			else:
				# self.rho_adv[i] = ti.max(rho_adv * self.delta_time[None] + self.rho[i], self.rho[i])
				self.rho_adv[i] = rho_adv * self.delta_time[None] + self.rho[i]
			self.p_iter[i] = 0.5 * self.p_past[i]
			a_ii = 0.0
			self.ps.for_all_neighbor(i, self.compute_a_ii, a_ii)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				a_ii_boundary = 0.0
				self.ps.for_all_boundary_neighbor(i, self.compute_a_ii_boundary, a_ii_boundary)
				self.a_ii[i] = a_ii + a_ii_boundary * self.rho_0
			else:
				self.a_ii[i] = a_ii

	# @ti.kernel
	def pressure_solve(self):
		l = 0
		residual = ti.math.inf
		residuals = []
		err = self.rho_err_percent * self.rho_0 * 0.01
		while (residual > err or l < self.min_iter_cnt) and l < self.max_iter_cnt:
			self.compute_all_d_ij()

			self.update_p()

			l += 1

			residual = self.compute_residual()
			if len(residuals) > 0 and residual - residuals[-1] > 0:
				print('Iteration trend to divergence')
				break
			residuals.append(residual)

		print("Iter cnt: ", l, residual)

		if l == 50:
			print(residuals)
			# exit()

	@ti.kernel
	def compute_residual(self) -> ti.f32:
		residual = 0.0
		cnt = 0
		avg = 0.0
		for i in range(self.particle_count):
			if self.p_iter[i] > 0.0:
				residual += (self.a_ii[i] * self.p_iter[i] + self.r_sum[i] + self.rho_adv[i] - 1000)  # ** 2
				cnt += 1
		if cnt > 0:
			avg = residual / cnt
		return avg

	@ti.func
	def compute_rho_iter(self, i, j):
		q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
		return self.ps.particle_m * (self.f_press[i] / self.ps.particle_m - self.f_press[j] / self.ps.particle_m).dot(
			self.cubic_kernel_derivative(q, self.kernel_h))

	@ti.kernel
	def compute_all_d_ij(self):
		for i in range(self.particle_count):
			d_ij = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_d_ij, d_ij)
			self.d_ij[i] = d_ij * self.delta_time[None] * self.delta_time[None]

	@ti.kernel
	def update_p(self):
		for i in range(self.particle_count):
			sum = 0.0
			boundary_sum = 0.0
			self.ps.for_all_neighbor(i, self.sum_factor, sum)
			if self.boundary_handle == self.akinci2012_boundary_handle:
				self.ps.for_all_boundary_neighbor(i, self.sum_factor_boundary, boundary_sum)
				self.r_sum[i] = sum + boundary_sum
			else:
				self.r_sum[i] = sum
		for i in range(self.particle_count):
			if ti.abs(self.a_ii[i]) > 1e-7:
				self.p_new_buff[i] = (1 - self.omega) * self.p_iter[i] + self.omega * (
						self.rho_0 - self.rho_adv[i] - self.r_sum[i]) / (self.a_ii[i])
			else:
				self.p_new_buff[i] = 0.0

		for i in range(self.particle_count):
			self.p_iter[i] = ti.max(self.p_new_buff[i], 0.0)

	@ti.func
	def compute_rigid_force(self, particle_i, particle_j):
		force = ti.Vector([0.0, 0.0, 0.0])
		if particle_j.material == self.ps.material_solid:
			if self.fs_couple == self.two_way_couple:
				j = particle_j.index
				i = particle_i.index
				q = particle_i.pos - particle_j.pos
				w_ij = self.cubic_kernel_derivative(q, self.kernel_h)
				force = particle_j.volume * self.rho_0 / (self.rho[i] ** 2) * w_ij * self.p_iter[i]
				self.ps.rigid_particles[j].force += force * self.ps.particle_m
		return force

	@ti.func
	def compute_all_press_force(self):

		for i in range(self.particle_count):
			force = ti.Vector([0.0, 0.0, 0.0])
			self.f_press[i] = (self.d_ij[i] + self.d_ii[i] * self.p_iter[i]) * self.ps.particle_m / (self.delta_time[None] ** 2)
			# f_press = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_rigid_force, force)
			# if self.fs_couple == self.two_way_couple:
			# 	f_press_boundary = ti.Vector([0.0, 0.0, 0.0])
			# 	self.ps.for_all_boundary_neighbor(i, self.compute_press_force_boundary, f_press_boundary)
			# 	self.f_press[i] = - f_press + f_press_boundary * self.rho_0
			# else:
			# 	self.f_press[i] = - f_press

	@ti.func
	def compute_press_force_boundary(self, i, j):
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		rho_2 = self.rho[i] * self.rho[i]
		ret = - self.ps.particle_m * self.ps.boundary_particles.volume[j] * self.p_iter[i] / rho_2 * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret

	@ti.kernel
	def intergation(self):
		self.compute_all_press_force()

		for i in range(self.particle_count):
			self.ps.fluid_particles.vel[i] = self.v_adv[i] + self.delta_time[None] * self.f_press[i] / self.ps.particle_m
			self.ps.fluid_particles.pos[i] = self.ps.fluid_particles.pos[i] + self.delta_time[None] * self.ps.fluid_particles.vel[i]

		# self.check_valid()
		if self.boundary_handle == self.clamp_boundary_handle:
			for i in range(self.particle_count):
				for j in ti.static(range(3)):
					if self.ps.fluid_particles.pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
						self.ps.fluid_particles.pos[i][j] = self.ps.box_min[j] + self.ps.particle_radius
						self.ps.fluid_particles.vel[i][j] *= -self.v_decay_proportion

					if self.ps.fluid_particles.pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
						self.ps.fluid_particles.pos[i][j] = self.ps.box_max[j] - self.ps.particle_radius
						self.ps.fluid_particles.vel[i][j] *= -self.v_decay_proportion

		for i in range(self.particle_count):
			self.p_past[i] = self.p_iter[i]

	@ti.func
	def compute_press_force(self, particle_i, particle_j):
		ret = ti.Vector([0.0, 0.0, 0.0])
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
			kernel = self.cubic_kernel_derivative(q, self.kernel_h)
			ret = self.ps.particle_m * (self.p_iter[i] / ti.pow(self.rho[i], 2) + self.p_iter[j] / ti.pow(self.rho[j], 2)) * kernel * self.ps.particle_m
		elif particle_j.material == self.ps.material_solid:
			if self.fs_couple == self.two_way_couple:
				i = particle_i.index
				j = particle_j.index
				q = particle_i.pos - particle_j.pos
				rho_2 = self.rho[i] ** 2
				kernel = self.cubic_kernel_derivative(q, self.kernel_h)
				ret = self.ps.particle_m * particle_j.volume * self.p_iter[i] / rho_2 * kernel * self.rho_0
				self.ps.rigid_particles[j].force += ret
		return ret

	@ti.func
	def sum_factor_boundary(self, i, j):
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		w_ij = self.cubic_kernel_derivative(q, self.kernel_h)
		ret = self.d_ij[i].dot(w_ij) * self.ps.boundary_particles[j].volume * self.rho_0
		return ret

	@ti.func
	def sum_factor(self, particle_i, particle_j):
		ret = 0.0
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
			w_ij = self.cubic_kernel_derivative(q, self.kernel_h)
			w_ji = self.cubic_kernel_derivative(-q, self.kernel_h)
			d_ji = - self.delta_time[None] * self.delta_time[None] * self.ps.particle_m / (
					self.rho[i] * self.rho[i]) * w_ji * self.p_iter[i]
			ret = self.ps.particle_m * (self.d_ij[i] - self.d_ii[j] * self.p_iter[j] - (self.d_ij[j] - d_ji)).dot(w_ij)
		elif particle_j.material == self.ps.material_solid:
			if self.fs_couple == self.two_way_couple:
				i = particle_i.index
				q = particle_i.pos - particle_j.pos
				w_ij = self.cubic_kernel_derivative(q, self.kernel_h)
				ret = self.d_ij[i].dot(w_ij) * particle_j.volume * self.rho_0
		return ret

	@ti.func
	def compute_d_ii(self, particle_i, particle_j):
		ret = ti.Vector([0.0, 0.0, 0.0])
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			q = particle_i.pos - particle_j.pos
			ret = - self.ps.particle_m / (self.rho[i] * self.rho[i]) * self.cubic_kernel_derivative(q, self.kernel_h)
		elif particle_j.material == self.ps.material_solid:
			if self.fs_couple == self.two_way_couple:
				i = particle_i.index
				q = particle_i.pos - particle_j.pos
				kernel = self.cubic_kernel_derivative(q, self.kernel_h)
				ret = - particle_j.volume * self.rho_0 / (self.rho[i] ** 2) * kernel
		return ret

	@ti.func
	def compute_boundary_d_ii(self, i, j):
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		return - self.ps.boundary_particles.volume[j] / (self.rho[i] * self.rho[i]) * self.cubic_kernel_derivative(q, self.kernel_h)

	@ti.func
	def compute_a_ii(self, particle_i, particle_j):
		ret = 0.0
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
			cubic_kernel_derivative = self.cubic_kernel_derivative(q, self.kernel_h)
			d_ji = - self.delta_time[None] * self.delta_time[None] * self.ps.particle_m / (
					self.rho[i] * self.rho[i]) * self.cubic_kernel_derivative(-q, self.kernel_h)
			ret = self.ps.particle_m * (self.d_ii[i] - d_ji).dot(cubic_kernel_derivative)
		elif particle_j.material == self.ps.material_solid:
			if self.fs_couple == self.two_way_couple:
				i = particle_i.index
				q = particle_i.pos - particle_j.pos
				kernel = self.cubic_kernel_derivative(q, self.kernel_h)
				d_ji = - self.delta_time[None] * self.delta_time[None] * self.ps.particle_m / (
						self.rho[i] * self.rho[i]) * self.cubic_kernel_derivative(-q, self.kernel_h)
				ret = particle_j.volume * (self.d_ii[i] - d_ji).dot(kernel) * self.rho_0

		return ret

	@ti.func
	def compute_a_ii_boundary(self, i, j):
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		cubic_kernel_derivative = self.cubic_kernel_derivative(q, self.kernel_h)
		d_ji = - self.delta_time[None] * self.delta_time[None] * self.ps.particle_m / (
				self.rho[i] * self.rho[i]) * self.cubic_kernel_derivative(-q, self.kernel_h)
		return self.ps.boundary_particles.volume[j] * (self.d_ii[i] - d_ji).dot(cubic_kernel_derivative)

	@ti.func
	def compute_d_ij(self, particle_i, particle_j):
		ret = ti.Vector([0.0, 0.0, 0.0])
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
			w_ij = self.cubic_kernel_derivative(q, self.kernel_h)
			ret = - self.ps.particle_m * self.p_iter[j] * w_ij / (self.rho[j] * self.rho[j])
		return ret

	@ti.func
	def compute_rho_adv(self, particle_i, particle_j):
		ret = 0.0
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			v_ij = self.v_adv[i] - self.v_adv[j]
			q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
			ret = self.ps.particle_m * v_ij.dot(self.cubic_kernel_derivative(q, self.kernel_h))
		elif particle_j.material == self.ps.material_solid:
			if self.fs_couple == self.two_way_couple:
				i = particle_i.index
				v_omega = ti.math.cross(particle_j.omega + particle_j.alpha * self.delta_time[None],
										particle_j.pos - self.ps.rigid_centriod[None])
				v_j = particle_j.vel + particle_j.acc * self.delta_time[None] + v_omega
				v_ij = self.v_adv[i] - v_j
				q = particle_i.pos - particle_j.pos
				ret = particle_j.volume * v_ij.dot(self.cubic_kernel_derivative(q, self.kernel_h)) * self.rho_0
		return ret

	@ti.func
	def compute_rho_adv_boundary(self, i, j):
		v_ij = self.v_adv[i]
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		return self.ps.boundary_particles.volume[j] * v_ij.dot(self.cubic_kernel_derivative(q, self.kernel_h))

	def step(self):
		super(iisph_solver, self).step()

		self.predict_advection()

		self.pressure_solve()

		self.intergation()
