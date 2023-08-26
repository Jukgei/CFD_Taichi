import taichi as ti
from solver_base import solver_base


class wcsph_solver(solver_base):

	def __init__(self, particle_system, config):
		super(wcsph_solver, self).__init__(particle_system, config)
		particle_count = particle_system.particle_num
		self.rho = ti.field(ti.float32, shape=particle_count)
		self.pressure_gradient = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.pressure = ti.field(ti.float32, shape=particle_count)
		self.viscosity = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.tension = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)

		self.viscosity_epsilon = 0.01
		self.viscosity_c_s = 5 #TODO 31
		self.viscosity_alpha = 0.08
		self.tension_k = 0.2
		self.gamma = 7
		self.B = 70000 #TODO ((math.sqrt(2 * 9.8 * 0.25) / 0.1) ** 2 ) * 1000 / 7

	# @ti.kernel
	def step(self):
		super(wcsph_solver, self).step()

		self.pressure_phase()

		self.kinematic_phase()

	@ti.kernel
	def pressure_phase(self):
		self.compute_all_rho()
		self.solve_all_pressure()
		self.solve_all_pressure_gradient()
		self.solve_all_viscosity()
		self.solve_all_tension()

	@ti.kernel
	def kinematic_phase(self):
		for i in range(self.particle_count):
			self.ps.acc[i] += - self.pressure_gradient[i] + self.viscosity[i] + self.tension[i]

		for i in range(self.particle_count):
			self.ps.vel[i] += self.ps.acc[i] * self.delta_time[None]
			self.ps.pos[i] += self.ps.vel[i] * self.delta_time[None]

		for i in range(self.particle_count):
			for j in ti.static(range(3)):
				if self.ps.pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_min[j] + self.ps.particle_radius
					self.ps.vel[i][j] *= -self.v_decay_proportion

				if self.ps.pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
					self.ps.pos[i][j] = self.ps.box_max[j] - self.ps.particle_radius
					self.ps.vel[i][j] *= -self.v_decay_proportion

	@ti.func
	def solve_rho(self, i):
		rho = 0.001
		for j in range(self.particle_count):
			if j != i:
				rho += self.ps.particle_m * self.cubic_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), self.kernel_h)
		return rho


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
		return - self.tension_k / self.ps.particle_m * self.ps.particle_m * self.cubic_kernel(q.norm(), self.kernel_h) * q

	@ti.func
	def solve_p(self, i):
		rho_i = ti.max(self.rho[i], self.rho_0)
		p = self.B * ((rho_i / self.rho_0) ** self.gamma - 1.0)
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
		ret += self.ps.particle_m * (p_i / (rho_i_2) + p_j / (rho_j ** 2)) * self.cubic_kernel_derivative(q, self.kernel_h)# * dir / (dir.norm() * self.kernel_h)
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
			nu = (2 * self.viscosity_alpha * self.kernel_h * self.viscosity_c_s) / (self.rho[i] + self.rho[j])
			pi = -nu * shear / (q2 + self.viscosity_epsilon * self.kernel_h * self.kernel_h)
			ret += - self.ps.particle_m * pi * self.cubic_kernel_derivative(x_ij, self.kernel_h)
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
				sum += self.ps.particle_m * (p_i / (rho_i_2) + p_j / (rho_j ** 2)) * self.cubic_kernel_derivative(q, self.kernel_h)
		return sum