import taichi as ti


@ti.data_oriented
class rigid_solver:

	def __init__(self, particle_system, config):
		solid_config = config.get('solid')
		solver_config = config.get('solver')
		scene_config = config.get('scene')
		self.ps = particle_system

		self.delta_time = ti.field(ti.float32, shape=())
		self.delta_time[None] = solver_config.get('delta_time')
		self.simulate_cnt = ti.field(ti.int32, shape=())
		self.gravity = scene_config.get('gravity')

		self.rho = solid_config.get('rho_0')
		self.particle_count = self.ps.rigid_particles_num
		self.displacement = ti.Vector([0.0, 0.0, 0.0])
		self.omega = ti.field(ti.math.vec3, shape=())
		self.attitude = ti.field(ti.math.vec3, shape=())

		self.v_decay_proportion = 0.9

	@ti.kernel
	def kinematic(self):
		force = ti.Vector([0.0, 0.0, 0.0])
		for i in range(self.particle_count):
			force += self.ps.rigid_particles.force[i]
			# self.ps.rigid_particles.force[i] = ti.Vector([0.0, 0.0, 0.0])

		# rotation
		torque = ti.Vector([0.0, 0.0, 0.0])
		for i in range(self.particle_count):
			pos = self.ps.rigid_particles.pos[i] - self.ps.rigid_centriod[None]
			torque += ti.math.cross(pos, self.ps.rigid_particles.force[i])
			self.ps.rigid_particles.force[i] = ti.Vector([0.0, 0.0, 0.0])

		alpha = self.ps.rigid_inertia_tensor[None] @ torque
		if self.simulate_cnt[None] == 1:
			alpha = ti.Vector([-2, 0.0, 0.0])
		# m = ti.math.rotation3d(0.0, 0.0, 0.0)
		# m = ti.math.rot_yaw_pitch_roll(1.0 / 180.0 * ti.math.pi, 0, 0)
		# m[0, 3] = -self.ps.rigid_centriod[None].x
		# m[1, 3] = -self.ps.rigid_centriod[None].y
		# m[2, 3] = -self.ps.rigid_centriod[None].z
		# print(m)
		self.omega[None] += alpha * self.delta_time[None]
		self.attitude[None] = self.omega[None] * self.delta_time[None]

		# m = ti.math.rotation3d(self.omega[None].x, self.omega[None].z, self.omega[None].y)
		# m = ti.math.rotation3d(self.attitude[None].x, self.attitude[None].z, self.attitude[None].y)
		m = ti.math.rotation3d(self.attitude[None].x, self.attitude[None].z, self.attitude[None].y)
		x_rotation_constrain = self.rotation_collision(self.attitude[None])
		print('x_rotation constrain {}, attitude {}'.format(x_rotation_constrain, self.attitude[None].x))
		if self.attitude[None].x > 0:
			if x_rotation_constrain.x < self.attitude[None].x:
				m = ti.math.rotation3d(x_rotation_constrain.x, self.attitude[None].z, self.attitude[None].y)
				self.omega[None].x *= -self.v_decay_proportion
				print('Collision! omega {}, attitude {}'.format(self.omega[None].x, self.attitude[None].x))
		elif self.attitude[None].x < 0:
			if x_rotation_constrain.y > self.attitude[None].x:
				m = ti.math.rotation3d(x_rotation_constrain.y, self.attitude[None].z, self.attitude[None].y)
				self.omega[None].x *= -self.v_decay_proportion
				print('Collision!! omega {}, attitude {}'.format(self.omega[None].x, self.attitude[None].x))

		# elif max_x <= 0:
		# 	m = ti.math.rotation3d(0.0, self.attitude[None].z, self.attitude[None].y)

		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] -= self.ps.rigid_centriod[None]

		for i in range(self.particle_count):
			pos = self.ps.rigid_particles.pos[i]
			pos4 = ti.Vector([pos.x, pos.y, pos.z, 1])
			pos4 = m @ pos4
			self.ps.rigid_particles.pos[i] = ti.Vector([pos4.x, pos4.y, pos4.z])

		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] += self.ps.rigid_centriod[None]

		sum_mass = 0.0
		for i in range(self.particle_count):
			sum_mass+=self.ps.rigid_particles.mass[i]

		self.ps.rigid_particles.acc.fill(force / sum_mass + self.gravity * ti.Vector([0.0, 0.0, 0.0]))
		# self.ps.rigid_particles.vel.fill()
		vel = self.ps.rigid_particles.acc[0] * self.delta_time[None] + self.ps.rigid_particles.vel[0]
		displacement = vel * self.delta_time[None]

		# self.displacement_distance = -ti.math.inf
		# displacement_min = displacement
		# for i in range(self.particle_count):
		# 	for j in ti.static(range(3)):
		# 		if self.ps.rigid_particles.pos[i][j] + displacement[j] <= self.ps.box_min[j] + self.ps.particle_radius:
		# 			self.ps.rigid_particles.rgb[i] = ti.Vector([1.0, 1.0, 1.0])
		# 			# ti.atomic_max(displacement[j], self.ps.box_min[j] + self.ps.particle_radius - self.ps.rigid_particles.pos[i][j])
		# 			# ti.atomic_max(vel[j], - vel[j] * self.v_decay_proportion)
		#
		# 		if self.ps.rigid_particles.pos[i][j] + displacement[j] >= self.ps.box_max[j] - self.ps.particle_radius:
		# 			self.ps.rigid_particles.rgb[i] = ti.Vector([1.0, 1.0, 1.0])
					# ti.atomic_min(displacement[j], self.ps.box_max[j] - self.ps.particle_radius - self.ps.rigid_particles.pos[i][j])
					# ti.atomic_min(vel[j], - vel[j] * self.v_decay_proportion)

		self.ps.rigid_particles.vel.fill(vel)
		for i in range(self.particle_count):
			self.ps.rigid_particles.pos[i] += displacement

		centroid = ti.Vector([0.0, 0.0, 0.0])
		sum_mass = 0.0
		for i in range(self.ps.rigid_particles_num):
			centroid += self.ps.rigid_particles.pos[i] * self.ps.rigid_particles.mass[i]
			sum_mass += self.ps.rigid_particles.mass[i]
		self.ps.rigid_centriod[None] = centroid / sum_mass

	@ti.func
	def rotation_collision(self, attitude: ti.math.vec3) -> ti.math.vec2:
		x_min_particle = self.ps.rigid_particles[0]
		x_max_particle = self.ps.rigid_particles[0]
		y_min_particle = self.ps.rigid_particles[0]
		y_max_particle = self.ps.rigid_particles[0]
		z_min_particle = self.ps.rigid_particles[0]
		z_max_particle = self.ps.rigid_particles[0]
		x_min = ti.math.inf
		y_min = ti.math.inf
		z_min = ti.math.inf
		x_max = - ti.math.inf
		y_max = - ti.math.inf
		z_max = - ti.math.inf
		center2boundary_min = self.ps.box_min - self.ps.rigid_centriod[None]
		center2boundary_max = self.ps.box_max - self.ps.rigid_centriod[None]
		for i in range(self.particle_count):
			particle = self.ps.rigid_particles[i]
			delta_position = particle.pos - self.ps.rigid_centriod[None]

			ti.atomic_max(x_max, delta_position.x)
			if x_max == delta_position.x:
				x_max_particle = particle

			ti.atomic_max(y_max, delta_position.y)
			if y_max == delta_position.y:
				y_max_particle = particle

			ti.atomic_max(z_max, delta_position.z)
			if z_max == delta_position.z:
				z_max_particle = particle

			ti.atomic_min(x_min, delta_position.x)
			if x_min == delta_position.x:
				x_min_particle = particle

			ti.atomic_min(y_min, delta_position.y)
			if y_min == delta_position.y:
				y_min_particle = particle

			ti.atomic_min(z_min, delta_position.z)
			if z_min == delta_position.z:
				z_min_particle = particle

		# self.ps.rigid_particles.rgb[x_min_particle.index] = ti.Vector([1.0, 1.0, 1.0])
		# self.ps.rigid_particles.rgb[x_max_particle.index] = ti.Vector([1.0, 1.0, 1.0])
		# self.ps.rigid_particles.rgb[y_min_particle.index] = ti.Vector([1.0, 1.0, 1.0])
		# self.ps.rigid_particles.rgb[y_max_particle.index] = ti.Vector([1.0, 1.0, 1.0])
		# self.ps.rigid_particles.rgb[z_min_particle.index] = ti.Vector([1.0, 1.0, 1.0])
		# self.ps.rigid_particles.rgb[z_max_particle.index] = ti.Vector([1.0, 1.0, 1.0])

		if ti.abs(x_min) > ti.abs(center2boundary_min.x):
			pass

		# Just consider rotation around the x-axis
		y_d = self.ps.rigid_particles.pos[y_max_particle.index] - self.ps.rigid_centriod[None]
		y_d.x = 0
		y_rad_max = ti.math.acos(y_max / y_d.norm())
		y_rad_min = ti.math.acos(center2boundary_max.y / y_d.norm())
		y_rad = y_rad_max - y_rad_min
		y_rad_com = 2 * ti.math.pi - 2 * y_rad_min - y_rad
		z = ti.sqrt(y_d.norm() ** 2 - center2boundary_max.y ** 2)
		# Don't change sign
		vec1 = ti.Vector([y_d.z, y_d.y])
		vec2 = ti.Vector([-z, center2boundary_min.y])
		if attitude.x < 0:
			vec2 = ti.Vector([z, center2boundary_min.y])
		clockwise = self.clockwise_orientation(vec1, vec2)
		y_rotation_clockwise = y_rad
		y_rotation_clockwise_inv = -y_rad_com
		if clockwise == 1:
			y_rotation_clockwise = y_rad
			y_rotation_clockwise_inv = - y_rad_com
		elif clockwise == -1:
			y_rotation_clockwise = y_rad_com
			y_rotation_clockwise_inv = - y_rad

		y_d_min = self.ps.rigid_particles.pos[y_min_particle.index] - self.ps.rigid_centriod[None]
		y_d_min.x = 0
		ymin_rad_max = ti.math.acos(ti.abs(y_min) / y_d_min.norm())
		ymin_rad_min = ti.math.acos(ti.abs(center2boundary_min.y) / y_d_min.norm())
		ymin_rad = ymin_rad_max - ymin_rad_min
		ymin_rad_com = 2 * ti.math.pi - 2 * ymin_rad_min - ymin_rad
		z = ti.sqrt(y_d_min.norm() ** 2 - center2boundary_min.y ** 2)
		# change sign
		vec1 = ti.Vector([-y_d_min.z, y_d_min.y])
		vec2 = ti.Vector([z, center2boundary_min.y])
		if attitude.x < 0:
			vec2 = ti.Vector([-z, center2boundary_min.y])
		clockwise = self.clockwise_orientation(vec1, vec2)
		ymin_rotation_clockwise = ymin_rad
		ymin_rotation_clockwise_inv = -ymin_rad_com
		if clockwise == 1:
			ymin_rotation_clockwise = ymin_rad
			ymin_rotation_clockwise_inv = - ymin_rad_com
		elif clockwise == -1:
			ymin_rotation_clockwise = ymin_rad_com
			ymin_rotation_clockwise_inv = - ymin_rad


		z_d = self.ps.rigid_particles.pos[z_max_particle.index] - self.ps.rigid_centriod[None]
		z_d.x = 0
		z_rad_max = ti.math.acos(z_max / z_d.norm())
		z_rad_min = ti.math.acos(center2boundary_max.z / z_d.norm())
		z_rad = z_rad_max - z_rad_min
		z_rad_com = 2 * ti.math.pi - 2 * z_rad_min - z_rad
		y = ti.sqrt(z_d.norm() ** 2 - center2boundary_max.z ** 2)
		# Don't change sign
		vec1 = ti.Vector([-z_d.z, z_d.y])
		vec2 = ti.Vector([-center2boundary_max.z, y])
		if attitude.x < 0:
			vec2 = ti.Vector([-center2boundary_max.z, -y])
		clockwise = self.clockwise_orientation(vec1, vec2)
		# print('clockwise: {}, vec1 {}, vec2 {}'.format(clockwise, vec1, vec2))
		z_rotation_clockwise = z_rad
		z_rotation_clockwise_inv = -z_rad_com
		if clockwise == 1:
			z_rotation_clockwise = z_rad
			z_rotation_clockwise_inv = - z_rad_com
		elif clockwise == -1:
			z_rotation_clockwise = z_rad_com
			z_rotation_clockwise_inv = - z_rad

		z_d_min = self.ps.rigid_particles.pos[z_min_particle.index] - self.ps.rigid_centriod[None]
		z_d_min.x = 0
		zmin_rad_max = ti.math.acos(ti.abs(z_min) / z_d_min.norm())
		zmin_rad_min = ti.math.acos(ti.abs(center2boundary_min.z) / z_d_min.norm())
		zmin_rad = zmin_rad_max - zmin_rad_min
		zmin_rad_com = 2 * ti.math.pi - 2 * zmin_rad_min - zmin_rad
		y = ti.sqrt(z_d_min.norm() ** 2 - center2boundary_min.z ** 2)
		# Don't change sign
		vec1 = ti.Vector([-z_d_min.z, z_d_min.y])
		vec2 = ti.Vector([-center2boundary_min.z, y])
		if attitude.x < 0:
			vec2 = ti.Vector([-center2boundary_min.z, -y])
		clockwise = self.clockwise_orientation(vec1, vec2)
		# print('clockwise: {}, vec1 {}, vec2 {}'.format(clockwise, vec1, vec2))
		zmin_rotation_clockwise = zmin_rad
		zmin_rotation_clockwise_inv = -zmin_rad_com
		if clockwise == 1:
			zmin_rotation_clockwise = zmin_rad
			zmin_rotation_clockwise_inv = - zmin_rad_com
		elif clockwise == -1:
			zmin_rotation_clockwise = zmin_rad_com
			zmin_rotation_clockwise_inv = - zmin_rad


		# if z_min < center2boundary_min.z:
		# 	self.ps.rigid_particles.rgb[z_min_particle.index] = ti.Vector([1.0, 1.0, 1.0])
		# 	# print('Collison !')
		# else:
		# 	self.ps.rigid_particles.rgb[z_min_particle.index] = ti.Vector([1.0, 1.0, 0.0])
			# print('center2boundary_max.y {}, y_max {}'.format(center2boundary_max.y, y_max))
		x_max = ti.min(y_rotation_clockwise, z_rotation_clockwise, zmin_rotation_clockwise, ymin_rotation_clockwise)
		x_min = ti.max(y_rotation_clockwise_inv, z_rotation_clockwise_inv, zmin_rotation_clockwise_inv, ymin_rotation_clockwise_inv)
		# print(y_rad, z_rad, zmin_rad, ymin_rad)
		# print('clock wise {}, {}'.format(y_rotation_clockwise, ymin_rotation_clockwise))
		# print('clock wise inv {}, {}'.format(y_rotation_clockwise_inv, ymin_rotation_clockwise_inv))
		return ti.Vector([x_max, x_min])

	@ti.func
	def clockwise_orientation(self, vec1: ti.math.vec2, vec2: ti.math.vec2) -> ti.i32:
		# d = ti.math.dot(vec1, vec2)
		# c = ti.math.cross(vec1, vec2).norm()
		# ret = 0
		# if d > 0 and c > 0:
		# 	ret = -1
		# elif d < 0 and c > 0:
		# 	ret = 1
		ret = 0
		c = ti.math.cross(vec1, vec2)
		if c > 0:
			ret = -1
		elif c < 0:
			ret = 1
		else:
			pass
			# print('Error!')
		return ret


	@ti.kernel
	def reset(self):
		self.ps.rigid_particles.acc.fill(0.0)

		# TODO Debug
		self.ps.rigid_particles.rgb.fill(ti.Vector([1.0, 0.0, 0.0]))

	def step(self):
		self.simulate_cnt[None] += 1

		self.reset()

		self.kinematic()