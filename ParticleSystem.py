import taichi as ti


@ti.data_oriented
class ParticleSystem:

	def __init__(self, box_min, box_max, particle_radius):
		# self.water_size = ti.Vector([0.5, 0.8, 0.5])
		self.water_size = ti.Vector([0.05, 0.51, 0.5])
		self.start_pos = ti.Vector([0, 0, 0])
		self.particle_radius = particle_radius
		self.particle_m = 1000 * ((self.particle_radius * 2) ** 3) * ti.math.pi / 6
		# self.particle_m = 1000 * ((self.particle_radius * 2) ** 3) * 0.8
		self.distance_scale = 1
		# self.distance_scale = 6.25
		# self.distance_scale = 10

		self.particle_num = int(
			self.water_size.x / self.particle_radius * self.water_size.y / self.particle_radius * self.water_size.z / self.particle_radius)
		print('particle count: ', self.particle_num)
		print('particle m: ', self.particle_m)
		self.pos = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.vel = ti.Vector.field(3, ti.f32, shape=self.particle_num)
		self.acc = ti.Vector.field(3, ti.f32, shape=self.particle_num)

		self.box_max = box_max
		self.box_min = box_min

		self.solver = None

		self.init_particle()


	@ti.kernel
	def init_particle(self):
		for i in range(self.particle_num):
			x_num = self.water_size.x / self.particle_radius
			y_num = self.water_size.y / self.particle_radius
			z_num = self.water_size.z / self.particle_radius
			xz_num = x_num * z_num

			# x = i % xz_num % x_num
			x = i % x_num
			# z = int(i % xz_num / y_num)
			z = i /x_num % z_num
			y = int(i / xz_num)
			self.pos[i] = ti.Vector([x, y, z]) * self.particle_radius * 2 + self.start_pos
			# print( (ti.Vector([x, y, z]) + self.start_pos))