# coding=utf-8

import taichi as ti
import logging

ti.init(ti.cuda, debug=False, device_memory_fraction=0.5)

logging.basicConfig(level=logging.DEBUG, format="%(name)s (%(levelname)s): %(message)s")
logger = logging.getLogger("Simulation")
# logger = logging.getLogger(__name__)


N = 1

water_size = ti.Vector([1, 1, 0.7])

particle_radius = 0.1
# particle_num = ti.field(int, shape=())
particle_num = int(water_size.x / particle_radius * water_size.y / particle_radius * water_size.z / particle_radius)
logger.info("Particle count: %d", particle_num)

start_pos = ti.Vector([1, 1, 1])

box_min = ti.Vector([0, 0, 0])
box_max =  ti.Vector([3, 3, 3])

pos = ti.Vector.field(3, ti.f32, shape=particle_num)
vel = ti.Vector.field(3, ti.f32, shape=particle_num)
acc = ti.Vector.field(3, ti.f32, shape=particle_num)


box_vert = ti.Vector.field(3, ti.f32, shape=12)

box_vert[0] = ti.Vector([box_min.x, box_min.y, box_min.z])
box_vert[1] = ti.Vector([box_min.x, box_max.y, box_min.z])
box_vert[2] = ti.Vector([box_max.x, box_min.y, box_min.z])
box_vert[3] = ti.Vector([box_max.x, box_max.y, box_min.z])
box_vert[4] = ti.Vector([box_min.x, box_min.y, box_max.z])
box_vert[5] = ti.Vector([box_min.x, box_max.y, box_max.z])
box_vert[6] = ti.Vector([box_max.x, box_min.y, box_max.z])
box_vert[7] = ti.Vector([box_max.x, box_max.y, box_max.z])

box_lines_indices = ti.field(int, shape=(2 * 12))

for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
	box_lines_indices[i] = val

@ti.kernel
def init_particle():
	for i in range(particle_num):
		x_num = water_size.x / particle_radius
		y_num = water_size.y / particle_radius
		xy_num = x_num * y_num
		x = i % xy_num % x_num
		y = int(i % xy_num / y_num)
		z = int(i / xy_num)
		pos[i] = (ti.Vector([x, y, z]) + start_pos) * particle_radius * 2


@ti.func
def poly_kernel(r, h):
	ret = 0.0
	if r <= h:
		ret = 315 / (64 * ti.math.pi * h ** 9) * ((h ** 2 - r**2) **3)
	return ret


@ti.func
def spiky_kernel(r, h):
	ret = 0.0
	if r <= h:
		ret = 15 / (ti.math.pi * h ** 6) * ((h - r) **3)
	return ret

@ti.func
def viscosity_kernel(r, h):
	ret = 0.0
	if r <= h:
		a = - r ** 3 / (2 * h**3)
		b = r ** 2 / (h **2)
		c = h / (2 * r)
		ret = 15 / (2 * ti.math.pi * h ** 3) * (a + b + c - 1)
	return ret


@ti.kernel
def kernel_test():
	r = 0.5
	h = 1
	poly = poly_kernel(r, h)
	spiky = spiky_kernel(r, h)
	viscosity = viscosity_kernel(r, h)
	print("poly kernel value is", poly)
	print("spiky kernel value is", spiky)
	print("viscosity kernel value is", viscosity)

particles_pos = ti.Vector.field(3, dtype=ti.f32, shape=N)

if __name__ == "__main__":

	print("Simulation Start!")

	window = ti.ui.Window('Window Title', res=(640, 640), pos=(150, 150))

	canvas = window.get_canvas()

	scene = ti.ui.Scene()

	camera = ti.ui.Camera()
	camera.position(5,2,2)
	camera.lookat(-5,-2,-2)
	# camera.lookat(0,-1,0)
	camera.up(0,1,0)
	# ti.ui.ProjectionMode = 0
	# camera.projection_mode(ti.ui.ProjectionMode)
	scene.set_camera(camera)

	init_particle()
	kernel_test()


	while window.running:
		camera.track_user_inputs(window, movement_speed=0.005, hold_key=ti.ui.RMB)
		scene.set_camera(camera)
		scene.ambient_light((0.8, 0.8, 0.8))
		scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

		scene.particles(pos, color=(0.68, 0.26, 0.19), radius=particle_radius)
		scene.lines(box_vert, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=1.0)
		canvas.scene(scene)
		window.show()
