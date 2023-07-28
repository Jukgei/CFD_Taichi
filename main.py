# coding=utf-8

import taichi as ti
import logging
from ParticleSystem import ParticleSystem
from wcsph_solver import wcsph_solver

ti.init(ti.gpu, debug=False, device_memory_fraction=0.7, kernel_profiler=True)

logging.basicConfig(level=logging.DEBUG, format="%(name)s (%(levelname)s): %(message)s")
logger = logging.getLogger("Simulation")
# logger = logging.getLogger(__name__)


N = 1
particle_radius = 0.01

start_pos = ti.Vector([1, 1, 1])

box_min = ti.Vector([0.0, 0.0, 0.0])
box_max = ti.Vector([2.5, 3.0, 2.5])

# pos = ti.Vector.field(3, ti.f32, shape=particle_num)
# vel = ti.Vector.field(3, ti.f32, shape=particle_num)
# acc = ti.Vector.field(3, ti.f32, shape=particle_num)


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



@ti.func
# pressure
def poly_kernel(r, h):
	ret = 0.0
	if r <= h:
		ret = 315 / (64 * ti.math.pi * h ** 9) * ((h ** 2 - r**2) **3)
	return ret


@ti.func
# pressure
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
	r = 0.02
	h = 0.04
	poly = poly_kernel(r, h)
	spiky = spiky_kernel(r, h)
	viscosity = viscosity_kernel(r, h)
	print("poly kernel value is", poly)
	print("spiky kernel value is", spiky * 0.03)
	print("viscosity kernel value is", viscosity)
	# grid = ti.ceil(((box_max - box_min) / particle_radius), ti.i32)
	# for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
	# 	print(offset)
	# for offset in ti.grouped(ti.ndrange((-1,2), (-1, 2), (-1, 2))):
	# 	print(offset)

if __name__ == "__main__":

	print("Simulation Start!")

	window = ti.ui.Window('Window Title', res=(640, 640), pos=(150, 150))

	canvas = window.get_canvas()

	scene = ti.ui.Scene()

	camera = ti.ui.Camera()
	camera.position(10, 3, 4)
	camera.lookat(-5, -2, -2)
	# camera.lookat(0,-1,0)
	camera.up(0, 1, 0)
	# ti.ui.ProjectionMode = 0
	# camera.projection_mode(ti.ui.ProjectionMode)
	scene.set_camera(camera)

	# init_particle()
	# kernel_test()
	ps = ParticleSystem(box_min, box_max, particle_radius)
	ps.test()
	solver = wcsph_solver(ps)
	# solver.sample_a_rho()
	solver.step()
	while window.running:

		camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
		scene.set_camera(camera)
		scene.ambient_light((0.8, 0.8, 0.8))
		scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

		# scene.particles(pos, color=(0.68, 0.26, 0.19), radius=particle_radius)
		scene.particles(ps.pos, color=(0.0, 0.26, 0.68), radius=particle_radius)
		# scene.particles(ps.pos_debug, color=(1.0, 0.0, 0.0), radius=particle_radius)
		for i in range(5):
			solver.step()

			# ti.profiler.print_kernel_profiler_info()
			# ti.profiler.clear_kernel_profiler_info()
			# ti.profiler.print_kernel_profiler_info()
		# if window.is_pressed(ti.ui.ESCAPE):

		# ps.test()
		scene.lines(box_vert, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=2.0)
		canvas.scene(scene)
		window.show()
		# ti.profiler.print_scoped_profiler_info()
