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
		return self.ps.particle_m * self.cubic_kernel((self.ps.pos[i] - self.ps.pos[j]).norm(), self.kernel_h)

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
		self.ps.acc.fill(9.8 * ti.Vector([0.0, -1.0, 0.0]))

	def step(self):
		self.simulate_cnt[None] += 1

		self.ps.reset_grid()

		self.ps.update_grid()

		self.reset()

	@ti.func
	def check_valid(self):
		for i in range(self.particle_count):

			is_false = 0
			if ti.math.isnan(self.ps.pos[i]).any() or ti.math.isinf(self.ps.pos[i]).any():
				print("Position {} invalid, {}".format(i, self.ps.pos[i]))
				is_false = 1

			if ti.math.isnan(self.ps.vel[i]).any() or ti.math.isinf(self.ps.vel[i]).any():
				print("Velocity {} invalid, {}".format(i, self.ps.vel[i]))
				is_false = 1

			if ti.math.isnan(self.ps.acc[i]).any() or ti.math.isinf(self.ps.acc[i]).any():
				print("Acceleration {} invalid, {}".format(i, self.ps.acc[i]))
				is_false = 1

			if is_false == 1:
				self.print_debug_info(i)
				# exit()

	@ti.func
	def print_debug_info(self, i):
		print(self.rho[i])