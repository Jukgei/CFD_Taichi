import taichi as ti


v_decay_proportion = 0.5
kernel_h = 0.01 * 4
gamma = 7
B = 70000 # ((math.sqrt(2 * 9.8 * 0.25) / 0.1) ** 2 ) * 1000 / 7
# B = 70000

@ti.data_oriented
class wcsph_solver:

	def __init__(self, particle_system):
		particle_count = particle_system.particle_num
		self.ps = particle_system
		self.rho = ti.field(ti.float32, shape=particle_count)
		self.pressure_gradient = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.pressure = ti.field(ti.float32, shape=particle_count)
		self.viscosity = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.tension = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)

		self.particle_count = particle_count
		self.delta_time = 1e-4

		self.rho_0 = 1000
		self.viscosity_epsilon = 0.01
		self.viscosity_c_s = 5 #TODO 31
		self.viscosity_alpha = 0.08
		self.tension_k = 0.1


	@ti.kernel
	def sample_a_rho(self):
		# self.solve_all_rho()
		# print('Init Rho: ', self.rho[self.particle_count//2])
		# self.rho_0 = 1000
		# print('GGGGGG' , self.cubic_kernel_derivative(0, kernel_h) * self.ps.particle_m)
		# self.reset()
		pass

	# @ti.kernel
	def step(self):
		# self.semi_implicit_euler_step()
		self.ps.reset_grid()

		self.ps.update_grid()

		self.reset()

		self.pressure_phase()

		self.kinematic_phase()

	@ti.func
	def semi_implicit_euler_step(self):
		# pass

		# self.ps.reset_grid()
		# #
		# self.ps.update_grid()

		self.reset()

		self.pressure_phase()

		self.kinematic_phase()



	@ti.kernel
	def reset(self):
		# self.rho.fill(0)
		# self.pressure_gradient.fill(ti.Vector([0.0, 0.0, 0.0]))
		# self.pressure.fill(0)
		# self.viscosity.fill(ti.Vector([0.0, 0.0, 0.0]))

		self.ps.acc.fill(9.8 * ti.Vector([0, -1, 0]))

	@ti.kernel
	def pressure_phase(self):
		self.solve_all_rho()
		self.solve_all_pressure()
		self.solve_all_pressure_gradient()
		self.solve_all_viscosity()
		self.solve_all_tension()

	@ti.kernel
	def kinematic_phase(self):
		# print(self.ps.vel[0], self.ps.vel[0].norm(), self.ps.pos[0], self.ps.acc[0], self.pressure_gradient[0])
		for i in range(self.particle_count):
			self.ps.acc[i] += - self.pressure_gradient[i] + self.viscosity[i] + self.tension[i]

		for i in range(self.particle_count):
			self.ps.vel[i] += self.ps.acc[i] * self.delta_time
			self.ps.pos[i] += self.ps.vel[i] * self.delta_time

		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.ps.pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_min[j] + self.ps.particle_radius
					self.ps.vel[i][j] *= -v_decay_proportion

				if self.ps.pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_max[j] - self.ps.particle_radius
					self.ps.vel[i][j] *= -v_decay_proportion

	@ti.func
	def solve_all_rho(self):
		for i in range(self.particle_count):
			# self.rho[i] = self.solve_rho(i)
			rho = 0.001
			self.ps.for_all_neighbor(i, self.compute_rho, rho)
			self.rho[i] = rho

	@ti.func
	def solve_rho(self, i):
		rho = 0.001
		for j in range(self.particle_count):
			if j != i:
				rho += self.ps.particle_m * self.cubic_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), kernel_h)
		return rho


	@ti.func
	def compute_rho(self, i, j):
		return self.ps.particle_m * self.cubic_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), kernel_h)


	@ti.func
	def solve_all_pressure(self):
		for i in range(self.particle_count):
			self.pressure[i] = self.solve_p(i)


	@ti.func
	def solve_all_pressure_gradient(self):
		# for loop
		# for i in range(self.particle_count):
		# 	self.pressure_gradient[i] = self.solve_gradient_p(i)

		# neighbor
		for i in range(self.particle_count):
			ret = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_pressure_gradient, ret)
			self.pressure_gradient[i] = ret

	@ti.func
	def solve_all_viscosity(self):
		for i in range(self.particle_count):
			viscousity = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_viscousity, viscousity)
			self.viscosity[i] = viscousity

	@ti.func
	def solve_all_tension(self):
		for i in range(self.particle_count):
			tension = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_tension, tension)
			self.tension[i] = tension

	@ti.func
	def compute_tension(self, i, j) -> ti.math.vec3:
		q = self.ps.pos[i] - self.ps.pos[j]
		return - self.tension_k / self.ps.particle_m * self.ps.particle_m * self.cubic_kernel(q.norm(), kernel_h) * q

	@ti.func
	def cubic_kernel(self, r, h):
		ret = 0.0
		q = r / h
		k = 8 / (ti.math.pi * ti.pow(h, 3))
		if 0 <= q <= 0.5:
			q2 = q * q
			q3 = q2 * q
			ret = k * (6 * (q3 - q2) + 1)
		elif 0.5 < q <= 1:
			ret = 2 * k * ((1 - q) ** 3)
		else:
			ret = 0.0
		return ret

	@ti.func
	def cubic_kernel_derivative(self, r, h):
		r_norm = r.norm()
		q = r_norm / h
		ret = ti.Vector([0.0, 0.0, 0.0])
		k = 48 / (ti.math.pi * ti.pow(h, 3))
		if 1e-5 < q <= 0.5:
			q2 = q * q
			ret = k * 6 * (3 * q2 - 2 * q) * r / (h * r_norm)
		elif 0.5 < q <= 1:
			ret = -k * 6 * ((1 - q) ** 2) * r / (h * r_norm)
		else:
			ret = ti.Vector([0.0, 0.0, 0.0])
		return ret

	@ti.func
	def spiky_kernel(self, r, h):
		ret = 0.0
		# if r <= h:
		# 	ret = 15 / (ti.math.pi * h ** 6) * ((h - r) ** 3)
		q = r / h
		if q <= 1:
			ret = 15 * ((1 - q) ** 3) / (ti.math.pi * h * h * h)
		return ret

	@ti.func
	def gradient_spiky_kernel(self, r, h):
		# ret = 0.0
		# if r <= h:
		# 	ret = - 45.0 / (ti.math.pi * h ** 6) * ((h - r) ** 2)
		# return ret
		r_norm = r.norm()
		q = r_norm / h
		ret = ti.Vector([0.0, 0.0, 0.0])
		if q <= 1 and q > 0:
			ret = - (45 * (1 - q) ** 2) * r / (ti.math.pi * (h ** 4) * r_norm)
		return ret

	@ti.func
	def solve_p(self, i):
		rho_i = ti.max(self.rho[i], self.rho_0)
		p = B * ((rho_i / self.rho_0) ** gamma - 1.0)
		return p

	@ti.func
	def compute_pressure_gradient(self, i, j) -> ti.types.vector:
		ret = ti.Vector([0.0, 0.0, 0.0])
		rho_i = self.rho[i]
		rho_i_2 = rho_i ** 2

		p_i = self.pressure[i]

		p_j = self.pressure[j]
		rho_j = self.rho[j]
		q = self.ps.pos[i] - self.ps.pos[j]
		ret += self.ps.particle_m * (p_i / (rho_i_2) + p_j / (rho_j ** 2)) * self.cubic_kernel_derivative(q, kernel_h)# * dir / (dir.norm() * kernel_h)
		return ret

	@ti.func
	def compute_viscousity(self, i, j) -> ti.types.vector:
		ret = ti.Vector([0.0, 0.0, 0.0])
		v_ij = self.ps.vel[i] - self.ps.vel[j]
		x_ij = self.ps.pos[i] - self.ps.pos[j]
		shear = v_ij @ x_ij
		if shear < 0:
			q = x_ij.norm()
			q2 = q * q
			nu = (2 * self.viscosity_alpha * kernel_h * self.viscosity_c_s) / (self.rho[i] + self.rho[j])
			pi = -nu * shear / (q2 + self.viscosity_epsilon * kernel_h * kernel_h)
			ret += - self.ps.particle_m * pi * self.cubic_kernel_derivative(x_ij, kernel_h)
		return ret

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
				q = self.ps.pos[i] - self.ps.pos[j]
				sum += self.ps.particle_m * (p_i / (rho_i_2) + p_j / (rho_j ** 2)) * self.cubic_kernel_derivative(q, kernel_h)
		return sum