import taichi as ti


v_decay_proportion = 0.9
kernel_h = 0.01 * 4
gamma = 7
# B = 140000 # ((math.sqrt(2 * 9.8 * 0.25) / 0.1) ** 2 ) * 1000 / 7
B = 70000

@ti.data_oriented
class wcsph_solver:

	def __init__(self, particle_system):
		particle_count = particle_system.particle_num
		self.ps = particle_system
		self.rho = ti.field(ti.float32, shape=particle_count)
		self.pressure_gradient = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.pressure = ti.field(ti.float32, shape=particle_count)
		self.viscosity = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)

		self.particle_count = particle_count
		self.delta_time = 1e-4
		self.rho_0 = ti.field(dtype=float, shape=())

	@ti.kernel
	def sample_a_rho(self):
		self.solve_all_rho()
		print('Init Rho: ', self.rho[self.particle_count//2])
		self.rho_0[None] = 1000
		self.reset()

	@ti.kernel
	def step(self):
		self.semi_implicit_euler_step()

	@ti.func
	def semi_implicit_euler_step(self):
		# pass
		self.reset()

		self.pressure_phase()

		self.kinematic_phase()

	@ti.func
	def reset(self):
		self.rho.fill(0)
		self.pressure_gradient.fill(ti.Vector([0.0, 0.0, 0.0]))
		self.pressure.fill(0)
		self.viscosity.fill(ti.Vector([0.0, 0.0, 0.0]))

		self.ps.acc.fill(9.8 * ti.Vector([0, -1, 0]))


	@ti.func
	def pressure_phase(self):
		self.solve_all_rho()
		self.solve_all_pressure()
		self.solve_all_pressure_gradient()
		# self.solve_all_viscosity()

	@ti.func
	def kinematic_phase(self):
		# print(self.ps.vel[0], self.ps.vel[0].norm(), self.ps.pos[0], self.ps.acc[0], self.pressure_gradient[0])
		for i in range(self.particle_count):
			self.ps.acc[i] += - self.pressure_gradient[i] + self.viscosity[i]

		for i in range(self.particle_count):
			self.ps.vel[i] += self.ps.acc[i] * self.delta_time
			self.ps.pos[i] += self.ps.vel[i] * self.delta_time
		# print(self.ps.vel[0], self.ps.vel[0].norm(), self.ps.pos[0], self.ps.acc[0], self.pressure_gradient[0])
		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.ps.pos[i][j] <= self.ps.box_min[j]:
					self.ps.pos[i][j] = self.ps.box_min[j]
					self.ps.vel[i][j] *= -v_decay_proportion

				if self.ps.pos[i][j] >= self.ps.box_max[j]:
					self.ps.pos[i][j] = self.ps.box_max[j]
					self.ps.vel[i][j] *= -v_decay_proportion

	@ti.func
	def solve_all_rho(self):
		for i in range(self.particle_count):
			self.rho[i] = self.solve_rho(i)

	@ti.func
	def solve_rho(self, i):
		rho = 0.001
		for j in range(self.particle_count):
			if j != i:
				# if True:
				# rho += particle_m * solve_gradient_kernel(x[particle_i], x[j])
				# if j==1 and i ==2:
				# 	print((self.ps.pos[i] - self.ps.pos[j]).norm() * self.ps.distance_scale)
				rho += self.ps.particle_m * self.spiky_kernel((self.ps.pos[i] - self.ps.pos[j]).norm() * self.ps.distance_scale, kernel_h)
		# print("rho ", rho, )
		return rho


	@ti.func
	def solve_all_pressure(self):
		for i in range(self.particle_count):
			self.pressure[i] = self.solve_p(i)


	@ti.func
	def solve_all_pressure_gradient(self):
		for i in range(self.particle_count):
			self.pressure_gradient[i] = self.solve_gradient_p(i)


	@ti.func
	def solve_all_viscosity(self):
		pass

	@ti.func
	def spiky_kernel(self, r, h):
		ret = 0.0
		if r <= h:
			ret = 15 / (ti.math.pi * h ** 6) * ((h - r) ** 3)
		return ret

	@ti.func
	def gradient_spiky_kernel(self, r, h):
		ret = 0.0
		if r <= h:
			ret = - 45.0 / (ti.math.pi * h ** 6) * ((h - r) ** 2)
		return ret

	@ti.func
	def solve_p(self, i):
		rho_i = ti.max(self.rho[i], self.rho_0[None])

		p = B * ((rho_i / self.rho_0[None]) ** gamma - 1.0)
		return p

	@ti.func
	def solve_gradient_p(self, i):
		rho_i = self.rho[i]
		rho_i_2 = rho_i ** 2
		sum = ti.Vector([0.0, 0.0, 0.0])
		p_i = self.pressure[i]
		for j in range(self.particle_count):
			if j != i:
				# if True:
				p_j = self.pressure[j]
				rho_j = self.rho[j]
				q = (self.ps.pos[i] - self.ps.pos[j]).norm() * self.ps.distance_scale
				dir = (self.ps.pos[i] - self.ps.pos[j]).normalized()

				# sum += particle_m * (p_i/ (rho_i_2) + p_j / (rho_j ** 2)) * solve_smooth_kernel(q) * solve_gradient_kernel(q) * dir
				sum += self.ps.particle_m * (p_i / (rho_i_2) + p_j / (rho_j ** 2)) * self.gradient_spiky_kernel(q, kernel_h) * dir


		return sum