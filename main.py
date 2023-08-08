# coding=utf-8

import taichi as ti
import logging
import argparse
import importlib
import utils
from ParticleSystem import ParticleSystem

parser = argparse.ArgumentParser(description='SPH in Taichi')
parser.add_argument('--config', help="Please input a scene config json file.", type=str, default='default.json')
args = parser.parse_args()
config = utils.read_config(args.config)

scene_config = config.get('scene')
solver_config = config.get('solver')
fluid_config = config.get('fluid')

ti.init(ti.gpu, debug=False, device_memory_fraction=0.7, kernel_profiler=True)

logging.basicConfig(level=logging.DEBUG, format="%(name)s (%(levelname)s): %(message)s")
logger = logging.getLogger("Simulation")
# logger = logging.getLogger(__name__)


# Init scene box
box_min = ti.Vector(scene_config.get('box_min'))
box_max = ti.Vector(scene_config.get('box_max'))
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


if __name__ == "__main__":

	print("Simulation Start!")

	window = ti.ui.Window('Window Title', res=(640, 640), pos=(150, 150))

	canvas = window.get_canvas()

	scene = ti.ui.Scene()

	camera = ti.ui.Camera()
	camera.position(7, 3, 5)
	camera.lookat(-5, -2, -2)
	camera.up(0, 1, 0)
	scene.set_camera(camera)

	ps = ParticleSystem(config)
	solver_name = solver_config.get('name')
	module = importlib.import_module(solver_name + '_solver')
	solver_ = getattr(module, solver_name + '_solver')
	solver = solver_(ps, config)
	frame_cnt = 0
	iter_cnt = solver_config.get('iter_cnt')
	# for i in range(800):
	# 	solver.step()
	# 	print("Frame: ", i)
	# a = []
	# for i in range(ps.particle_num):
	# 	a.append(ps.vel[i].norm())
	# 	# print("vel{}, {}".format(i, ps.vel[i]))
	# print("Max {}, Min {}".format(max(a), min(a)))
	while window.running:
		# Debug GUI
		gui = window.get_gui()
		gui.text("frame_cnt: {}".format(frame_cnt))
		gui.text("time: {:.4f}".format(frame_cnt * iter_cnt * solver.delta_time))

		# Cam and light
		camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
		scene.set_camera(camera)
		scene.ambient_light((0.8, 0.8, 0.8))
		scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

		scene.particles(ps.pos, color=(0.0, 0.26, 0.68), radius=ps.particle_radius)

		for i in range(iter_cnt):
			solver.step()
			# ti.profiler.print_kernel_profiler_info()
			# ti.profiler.clear_kernel_profiler_info()
		# ti.profiler.print_kernel_profiler_info()

		scene.lines(box_vert, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=2.0)

		canvas.scene(scene)
		window.show()
		frame_cnt += 1
