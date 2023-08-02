import taichi as ti
from solver_base import solver_base

kernel_h = 0.01 * 4


@ti.data_oriented
class pbf_solver(solver_base):

	def __init__(self, particle_system):
		super(pbf_solver, self).__init__(particle_system)
		particle_count = particle_system.particle_num

		self.pressure_gradient = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.pressure = ti.field(ti.float32, shape=particle_count)
		self.viscosity = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.tension = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)

		self.constrain = ti.field(ti.float32, shape=particle_count)
		self.pos_predict = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.delta_pos = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.constrain_derivative = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.derivative_sum = ti.field(ti.float32, shape=particle_count)
		self.pbf_lambda = ti.field(ti.float32, shape=particle_count)
		self.epsilon = 100

		self.particle_count = particle_count
		self.delta_time = 1e-4

		self.rho_0 = 1000
		self.viscosity_epsilon = 0.01
		self.viscosity_c_s = 5  # TODO 31
		self.viscosity_alpha = 0.08
		self.tension_k = 0.1

	@ti.kernel
	def reset(self):
		self.ps.acc.fill(9.8 * ti.Vector([0.0, -1.0, 0.0]))
		# self.ps.acc.fill(0 * ti.Vector([0.0, -1.0, 0.0]))

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
			# print(self.compute_derivative_sum(0, i))

			self.ps.for_all_neighbor(i, self.compute_derivative_around_sum, sum)
			# self.derivative_sum[i] -= sum
			# self.derivative_sum[i] += self.constrain_derivative[i].norm()
			sum = self.constrain_derivative[i].dot(self.constrain_derivative[i]) + sum
		# for i in range(self.particle_count):
			self.pbf_lambda[i] = - self.constrain[i] / (sum + self.epsilon)
			# self.derivative_sum[i] = 0.0
			# if i == 15077:
			# 	print('lambda[{}]: {}'.format(i, self.pbf_lambda[i]))
			# 	print('constrain[{}]: {}'.format(i, self.constrain[i]))
			# 	print(self.derivative_sum[i], sum / self.rho_0)
			# 	print('\n')


	@ti.kernel
	def compute_all_delta_pos(self):
		for i in range(self.particle_count):
			delta_pos = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_delta_pos, delta_pos)
			self.delta_pos[i] = delta_pos / self.rho_0
			# if self.delta_pos[i].norm() > 1:
			# 	print('delta_pos', self.delta_pos[i], self.constrain[i], self.delta_pos[i].norm(), i)

	@ti.kernel
	def update_all_pos(self):
		for i in range(self.particle_count):
			self.pos_predict[i] += self.delta_pos[i]
			self.ps.vel[i] = (self.pos_predict[i] - self.ps.pos[i]) / self.delta_time

			# for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.pos_predict[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
					self.pos_predict[i][j] = self.ps.box_min[j] + self.ps.particle_radius
					self.ps.vel[i][j] *= -0.5

				if self.pos_predict[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
					self.pos_predict[i][j] = self.ps.box_max[j] - self.ps.particle_radius
					self.ps.vel[i][j] *= -0.5

			# self.ps.vel[i] = (self.pos_predict[i] - self.ps.pos[i]) / self.delta_time
			self.ps.pos[i] = self.pos_predict[i]

			v = ti.Vector([0, 0, 0])
			self.ps.for_all_neighbor(i, self.update_vel, v)
			self.ps.vel[i] += 0.00000009 * v

	@ti.func
	def update_vel(self, i, j):
		# ret = ti.Vector([0.0, 0.0, 0.0])
		return (self.ps.vel[j] - self.ps.vel[i] ) * self.poly_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), kernel_h)

	@ti.func
	def compute_all_constrain_derivative(self):
		for i in range(self.particle_count):
			constrain_derivative = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_constrain_derivative, constrain_derivative)
			self.constrain_derivative[i] = constrain_derivative

	@ti.func
	def compute_constrain_derivative(self, i, j) -> ti.types.vector:
		return self.spiky_kernel_derivative(self.ps.pos[i] - self.ps.pos[j], kernel_h) / self.rho_0

	@ti.func
	def compute_all_constrain(self):
		for i in range(self.particle_count):
			self.constrain[i] = self.rho[i] / self.rho_0 - 1

	@ti.func
	def compute_derivative_around_sum(self, i, j):

		ret = self.spiky_kernel_derivative(self.ps.pos[i] - self.ps.pos[j], kernel_h) / self.rho_0
		return ret.dot(ret)

	@ti.func
	def compute_delta_pos(self, i, j) -> ti.types.vector:
		# ret = ti.Vector([0.0, 0.0, 0.0])
		# if i ==59765 and j == 58864:
		# 	# print(self.pbf_lambda[i], self.pbf_lambda[j])
		# 	# print(self.spiky_kernel_derivative(self.ps.pos[i] - self.ps.pos[j], kernel_h))
		# 	# print(i, j, self.spiky_kernel_derivative(self.ps.pos[i] - self.ps.pos[j], kernel_h) )
		# 	# print(self.ps.pos[59765], self.ps.pos[58864], (self.ps.pos[i] - self.ps.pos[j]).norm())
		# 	print((self.pbf_lambda[i] + self.pbf_lambda[j]) * self.spiky_kernel_derivative(self.ps.pos[i] - self.ps.pos[j],
		# 																				kernel_h), j)
		# 	print("lambda:", self.pbf_lambda[i], self.pbf_lambda[j])
		# 	print("kernel: ", self.spiky_kernel_derivative(self.ps.pos[i] - self.ps.pos[j], kernel_h) )
		# 	print("distant", (self.ps.pos[i] - self.ps.pos[j]).norm())
		# 	print('\n')
		k = 0.000000005
		n = 4
		s_corr = -k * ((self.poly_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), kernel_h) / self.poly_kernel(0.1 * kernel_h, kernel_h))** n)
		# print((self.spiky_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), kernel_h) / self.spiky_kernel(0.3 * kernel_h, kernel_h)))
		return (self.pbf_lambda[i] + self.pbf_lambda[j] + s_corr) * self.spiky_kernel_derivative(self.ps.pos[i] - self.ps.pos[j],
																						kernel_h)

	@ti.func
	def compute_rho(self, i, j):
		return self.ps.particle_m * self.poly_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), kernel_h)

	def step(self):

		self.ps.reset_grid()

		self.ps.update_grid()

		self.reset()

		self.externel_force_predict_pos()

		self.compute_all_lambda()

		self.compute_all_delta_pos()

		self.update_all_pos()

