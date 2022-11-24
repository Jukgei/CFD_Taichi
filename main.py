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

start_pos = ti.Vector([0, 0, 0])

pos = ti.Vector.field(3, ti.f32, shape=particle_num)
vel = ti.Vector.field(3, ti.f32, shape=particle_num)
acc = ti.Vector.field(3, ti.f32, shape=particle_num)




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

	# for i in range(water_size.x):
	# 	for j in range(water_size.y):
	# 		for k in range(water_size.z):
	# 			x =


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
	# camera.projection_mode(mode)
	scene.set_camera(camera)
	init_particle()
	while window.running:
		camera.track_user_inputs(window, movement_speed=0.005, hold_key=ti.ui.RMB)
		scene.set_camera(camera)
		scene.ambient_light((0.8, 0.8, 0.8))
		scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

		scene.particles(pos, color=(0.68, 0.26, 0.19), radius=particle_radius)

		canvas.scene(scene)
		window.show()
