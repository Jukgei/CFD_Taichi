import taichi as ti
from solver_base import solver_base


@ti.data_oriented
class pbf_solver(solver_base):

	def __init__(self, particle_system, config):
		super(pbf_solver, self).__init__(particle_system, config)
		particle_count = particle_system.particle_num

		self.constrain = ti.field(ti.float32, shape=particle_count)
		self.pos_predict = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.delta_pos = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.constrain_derivative = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.pbf_lambda = ti.field(ti.float32, shape=particle_count)
		self.epsilon = 1e8

		self.k = 1e-9 	# tension
		self.c = 5e-8 	# viscosity
		self.s_corr_factor = 0.1

		self.rho_0 = 1000

	@ti.kernel
	def externel_force_predict_pos(self):
		for i in range(self.particle_count):
			self.ps.vel[i] += self.delta_time * self.ps.acc[i]
			self.pos_predict[i] = self.ps.pos[i] + self.delta_time * self.ps.vel[i]

	@ti.kernel
	def compute_all_lambda(self):
		self.compute_all_rho()
		self.compute_all_constrain()
		self.compute_all_constrain_derivative()

		for i in range(self.particle_count):
			sum = 0.0

			self.ps.for_all_neighbor(i, self.compute_derivative_around_sum, sum)
			sum = self.constrain_derivative[i].dot(self.constrain_derivative[i]) + sum

			self.pbf_lambda[i] = - self.constrain[i] / (sum + self.epsilon)


	@ti.kernel
	def compute_all_delta_pos(self):
		for i in range(self.particle_count):
			delta_pos = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_delta_pos, delta_pos)
			self.delta_pos[i] = delta_pos / self.rho_0

	@ti.kernel
	def update_all_pos(self):
		for i in range(self.particle_count):
			self.pos_predict[i] += self.delta_pos[i]
			self.ps.vel[i] = (self.pos_predict[i] - self.ps.pos[i]) / self.delta_time

			# for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.pos_predict[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
					self.pos_predict[i][j] = self.ps.box_min[j] + self.ps.particle_radius
					self.ps.vel[i][j] *= self.v_decay_proportion

				if self.pos_predict[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
					self.pos_predict[i][j] = self.ps.box_max[j] - self.ps.particle_radius
					self.ps.vel[i][j] *= self.v_decay_proportion

			# self.ps.vel[i] = (self.pos_predict[i] - self.ps.pos[i]) / self.delta_time
			self.ps.pos[i] = self.pos_predict[i]

			v = ti.Vector([0, 0, 0])
			self.ps.for_all_neighbor(i, self.update_vel, v)
			self.ps.vel[i] += self.c * v

	@ti.func
	def update_vel(self, i, j):
		return (self.ps.vel[j] - self.ps.vel[i] ) * self.poly_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), self.kernel_h)

	@ti.func
	def compute_all_constrain_derivative(self):
		for i in range(self.particle_count):
			constrain_derivative = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_constrain_derivative, constrain_derivative)
			self.constrain_derivative[i] = constrain_derivative

	@ti.func
	def compute_constrain_derivative(self, i, j) -> ti.types.vector:
		return self.spiky_kernel_derivative(self.ps.pos[i] - self.ps.pos[j], self.kernel_h) / self.rho_0

	@ti.func
	def compute_all_constrain(self):
		for i in range(self.particle_count):
			self.constrain[i] = self.rho[i] / self.rho_0 - 1

	@ti.func
	def compute_derivative_around_sum(self, i, j):

		ret = self.spiky_kernel_derivative(self.ps.pos[i] - self.ps.pos[j], self.kernel_h) / self.rho_0
		return ret.dot(ret)

	@ti.func
	def compute_delta_pos(self, i, j) -> ti.types.vector:
		x_ij = self.ps.pos[i] - self.ps.pos[j]
		x_ij_norm = x_ij.norm()
		s_corr = self.poly_kernel(x_ij_norm, self.kernel_h) / self.poly_kernel(self.s_corr_factor * self.kernel_h, self.kernel_h)
		s_corr *= s_corr
		s_corr *= s_corr
		s_corr *= -self.k

		return (self.pbf_lambda[i] + self.pbf_lambda[j] + s_corr) * self.spiky_kernel_derivative(x_ij, self.kernel_h)

	@ti.func
	def compute_rho(self, i, j):
		return self.ps.particle_m * self.poly_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), self.kernel_h)

	def step(self):

		super(pbf_solver, self).step()

		self.externel_force_predict_pos()

		self.compute_all_lambda()

		self.compute_all_delta_pos()

		self.update_all_pos()

