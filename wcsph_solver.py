import taichi as ti
from solver_base import solver_base


class wcsph_solver(solver_base):

	def __init__(self, particle_system, config):
		super(wcsph_solver, self).__init__(particle_system, config)
		particle_count = particle_system.particle_num
		self.rho = ti.field(ti.float32, shape=particle_count)
		self.pressure_gradient = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.boundary_acc = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.pressure = ti.field(ti.float32, shape=particle_count)
		self.viscosity = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.tension = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)

		self.viscosity_epsilon = 0.01
		self.viscosity_c_s = 5
		self.viscosity_alpha = 0.08
		self.tension_k = 0.2
		self.gamma = 7
		self.B = 70000

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
			if self.boundary_handle == self.akinci2012_boundary_handle:
				self.ps.fluid_particles.acc[i] += self.pressure_gradient[i] + self.viscosity[i] + self.tension[i] + \
												  self.boundary_acc[i]
			else:
				self.ps.fluid_particles.acc[i] += self.pressure_gradient[i] + self.viscosity[i] + self.tension[i]

		for i in range(self.particle_count):
			self.ps.fluid_particles.vel[i] += self.ps.fluid_particles.acc[i] * self.delta_time[None]
			self.ps.fluid_particles.pos[i] += self.ps.fluid_particles.vel[i] * self.delta_time[None]

		if self.boundary_handle == self.clamp_boundary_handle:
			for i in range(self.particle_count):
				for j in ti.static(range(3)):
					if self.ps.fluid_particles.pos[i][j] <= self.ps.box_min[j] + self.ps.particle_radius:
						self.ps.fluid_particles.pos[i][j] = self.ps.box_min[j] + self.ps.particle_radius
						self.ps.fluid_particles.vel[i][j] *= -self.v_decay_proportion

					if self.ps.fluid_particles.pos[i][j] >= self.ps.box_max[j] - self.ps.particle_radius:
						self.ps.fluid_particles.pos[i][j] = self.ps.box_max[j] - self.ps.particle_radius
						self.ps.fluid_particles.vel[i][j] *= -self.v_decay_proportion

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
			if self.boundary_handle == self.akinci2012_boundary_handle:
				boundary_acc = ti.Vector([0.0, 0.0, 0.0])
				self.ps.for_all_boundary_neighbor(i, self.compute_boundary_pressure, boundary_acc)
				self.boundary_acc[i] = boundary_acc * self.rho_0
			self.pressure_gradient[i] = ret

	@ti.func
	def solve_p(self, i):
		rho_i = ti.max(self.rho[i], self.rho_0)
		p = self.B * ((rho_i / self.rho_0) ** self.gamma - 1.0)
		return p

	@ti.func
	def compute_boundary_pressure(self, i, j):
		ret = ti.Vector([0.0, 0.0, 0.0])
		p_i = self.pressure[i]
		rho_i = self.rho[i]
		rho_i_2 = rho_i ** 2
		q = self.ps.fluid_particles.pos[i] - self.ps.boundary_particles.pos[j]
		ret -= self.ps.boundary_particles.volume[j] * p_i / rho_i_2 * self.cubic_kernel_derivative(q, self.kernel_h)
		return ret

	@ti.func
	def compute_pressure_gradient(self, particle_i, particle_j) -> ti.types.vector:
		ret = ti.Vector([0.0, 0.0, 0.0])
		if particle_j.material == self.ps.material_fluid:
			i = particle_i.index
			j = particle_j.index
			rho_i = self.rho[i]
			rho_i_2 = rho_i ** 2

			p_i = self.pressure[i]

			p_j = self.pressure[j]
			rho_j = self.rho[j]
			q = particle_i.pos - particle_j.pos
			ret -= self.ps.particle_m * (p_i / rho_i_2 + p_j / (rho_j ** 2)) * self.cubic_kernel_derivative(q, self.kernel_h)  # * dir / (dir.norm() * self.kernel_h)
		elif particle_j.material == self.ps.material_solid:
			if self.boundary_handle == self.akinci2012_boundary_handle:
				i = particle_i.index
				p_i = self.pressure[i]
				rho_i = self.rho[i]
				rho_i_2 = rho_i ** 2
				q = particle_i.pos - particle_j.pos
				ret = - particle_j.volume * p_i / rho_i_2 * self.cubic_kernel_derivative(q, self.kernel_h) * self.rho_0
				# particle_j.force += -ret * particle_j.mass
				self.ps.rigid_particles[particle_j.index].force += -ret * self.ps.particle_m
				# if ret.norm() > 0:
				# 	print(particle_j.index, particle_j.force, self.ps.rigid_particles[particle_j.index].force)
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
				q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
				sum += self.ps.particle_m * (p_i / rho_i_2 + p_j / (rho_j ** 2)) * self.cubic_kernel_derivative(q, self.kernel_h)
		return sum
