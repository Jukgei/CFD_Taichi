import taichi as ti


@ti.data_oriented
class solver_base:

	def __init__(self, particle_system, config):
		particle_count = particle_system.particle_num
		scene_config = config.get('scene')
		solver_config = config.get('solver')
		fluid_config = config.get('fluid')
		self.particle_count = particle_count
		self.ps = particle_system
		self.rho = ti.field(ti.float32, shape=particle_count)
		self.delta_time = ti.field(ti.float32, shape=())
		self.delta_time[None] = solver_config.get('delta_time')
		self.kernel_h = self.ps.particle_radius * 4
		self.v_decay_proportion = 0.9
		self.rho_0 = 1000
		self.gravity = scene_config.get('gravity')
		self.simulate_cnt = ti.field(ti.int32, shape=())

		self.viscosity_epsilon = 0.01
		self.viscosity_c_s = 5
		self.viscosity_alpha = 0.08
		self.tension_k = 0.5

		self.viscosity = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)
		self.tension = ti.Vector.field(n=3, dtype=ti.float32, shape=particle_count)

		print("\033[32m[Solver]: {}\033[0m".format(solver_config.get('name')))

	@ti.func
	def compute_all_rho(self):
		for i in range(self.particle_count):
			# self.rho[i] = self.solve_rho(i)
			rho = 0.001
			self.ps.for_all_neighbor(i, self.compute_rho, rho)
			self.rho[i] = rho

	@ti.func
	def compute_all_task(self):
		#todo reflect
		pass

	@ti.func
	def compute_rho(self, i, j):
		return self.ps.particle_m * self.cubic_kernel((self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]).norm(), self.kernel_h)

	@staticmethod
	@ti.func
	def cubic_kernel(r, h):
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
		q = r / h
		if q <= 1:
			ret = 15 * ((1 - q) ** 3) / (ti.math.pi * h * h * h)
		return ret

	@ti.func
	def spiky_kernel_derivative(self, r, h):
		r_norm = r.norm()
		q = r_norm / h
		ret = ti.Vector([0.0, 0.0, 0.0])
		if q <= 1 and q > 0:
			ret = - (45 * (1 - q) ** 2) * r / (ti.math.pi * (h ** 4) * r_norm)
		return ret

	@ti.func
	def poly_kernel(self, r, h):
		q = r / h
		q2 = q * q
		ret = 0.0
		if q <= 1:
			ret = 315 / (64 * ti.math.pi * h ** 3) * ((1 - q2) ** 3)
		return ret

	@ti.kernel
	def reset(self):
		self.ps.fluid_particles.acc.fill(9.8 * ti.Vector([0.0, -1.0, 0.0]))

	def step(self):
		self.simulate_cnt[None] += 1

		self.ps.reset_grid()

		self.ps.update_grid()

		self.reset()

	@ti.func
	def check_valid(self):
		for i in range(self.particle_count):

			is_false = 0
			if ti.math.isnan(self.ps.fluid_particles.pos[i]).any() or ti.math.isinf(self.ps.fluid_particles.pos[i]).any():
				print("Position {} invalid, {}".format(i, self.ps.fluid_particles.pos[i]))
				is_false = 1

			if ti.math.isnan(self.ps.fluid_particles.vel[i]).any() or ti.math.isinf(self.ps.fluid_particles.vel[i]).any():
				print("Velocity {} invalid, {}".format(i, self.ps.fluid_particles.vel[i]))
				is_false = 1

			if ti.math.isnan(self.ps.fluid_particles.acc[i]).any() or ti.math.isinf(self.ps.fluid_particles.acc[i]).any():
				print("Acceleration {} invalid, {}".format(i, self.ps.fluid_particles.acc[i]))
				is_false = 1

			if is_false == 1:
				self.print_debug_info(i)
				# exit()

	@ti.func
	def print_debug_info(self, i):
		print(self.rho[i])

	@ti.func
	def solve_all_viscosity(self):
		for i in range(self.particle_count):
			viscosity = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_viscosity, viscosity)
			self.viscosity[i] = viscosity * self.ps.particle_m

	@ti.func
	def compute_viscosity(self, i, j) -> ti.types.vector:
		ret = ti.Vector([0.0, 0.0, 0.0])
		v_ij = self.ps.fluid_particles.vel[i] - self.ps.fluid_particles.vel[j]
		x_ij = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
		shear = v_ij @ x_ij
		if shear < 0:
			q = x_ij.norm()
			q2 = q * q
			nu = (2 * self.viscosity_alpha * self.kernel_h * self.viscosity_c_s) / (self.rho[i] + self.rho[j])
			pi = -nu * shear / (q2 + self.viscosity_epsilon * self.kernel_h * self.kernel_h)
			ret += - self.ps.particle_m * pi * self.cubic_kernel_derivative(x_ij, self.kernel_h)
		return ret

	@ti.func
	def solve_all_tension(self):
		for i in range(self.particle_count):
			tension = ti.Vector([0.0, 0.0, 0.0])
			self.ps.for_all_neighbor(i, self.compute_tension, tension)
			self.tension[i] = tension * self.ps.particle_m

	@ti.func
	def compute_tension(self, i, j) -> ti.math.vec3:
		q = self.ps.fluid_particles.pos[i] - self.ps.fluid_particles.pos[j]
		return - self.tension_k / self.ps.particle_m * self.ps.particle_m * self.cubic_kernel(q.norm(), self.kernel_h) * q

	@ti.kernel
	def visualize_rho(self):
		max_rho = - ti.math.inf
		min_rho = ti.math.inf
		for i in range(self.particle_count):
			ti.atomic_max(max_rho, self.rho[i])
			ti.atomic_min(min_rho, self.rho[i])
		# print('max rho {}, min rho {}'.format(max_rho, min_rho))

		for i in range(self.particle_count):
			if max_rho - min_rho > 0:
				b = (self.rho[i] - min_rho)/(max_rho - min_rho)
				self.ps.rgb[i] = ti.Vector([0.0, 0.28, b])

	@ti.kernel
	def visualize_neighbour(self):
		max_neighbour = - ti.math.inf
		min_neighbour = ti.math.inf
		for i in range(self.particle_count):
			neighbour = self.ps.get_neighbour_count(i)
			ti.atomic_max(max_neighbour, neighbour)
			ti.atomic_min(min_neighbour, neighbour)

		for i in range(self.particle_count):
			neighbour = self.ps.get_neighbour_count(i)
			if max_neighbour - min_neighbour > 0:
				self.ps.rgb[i] = ti.Vector([0.0, 0.28, (neighbour - min_neighbour) / (max_neighbour - min_neighbour)])