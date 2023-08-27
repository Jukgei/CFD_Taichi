# coding=utf-8

import taichi as ti
import logging
import argparse
import importlib
import utils
import numpy as np
from ParticleSystem import ParticleSystem

parser = argparse.ArgumentParser(description='SPH in Taichi')
parser.add_argument('--config', help="Please input a scene config json file.", type=str, default='default.json')
args = parser.parse_args()
config = utils.read_config(args.config)

scene_config = config.get('scene')
solver_config = config.get('solver')
fluid_config = config.get('fluid')

ti.init(ti.gpu, debug=False, device_memory_fraction=0.7, kernel_profiler=True)
# ti.init(ti.cpu, cpu_max_num_threads=1)

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

	video_manager = ti.tools.VideoManager(output_dir="./output", framerate=24, automatic_build=False)

	np_rgba = np.reshape(ps.rgba.to_numpy(), (ps.particle_num, 4))
	series_prefix = './output/output'
	is_output_gif = scene_config.get('is_output_gif', False)
	is_output_ply = scene_config.get('is_output_ply', False)
	is_pause = False
	while window.running:
		# Debug GUI
		gui = window.get_gui()
		gui.text("frame_cnt: {}".format(frame_cnt))
		gui.text("time: {:.4f}".format(frame_cnt * iter_cnt * solver.delta_time[None]))
		gui.text("Pause: {}".format(is_pause))
		if window.is_pressed(ti.GUI.SPACE):
			is_pause = not is_pause

		# Cam and light
		camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
		scene.set_camera(camera)
		scene.ambient_light((0.8, 0.8, 0.8))
		scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
		# if not is_pause:
		# 	solver.visualize_rho()
		scene.particles(ps.pos, color=(0.0, 0.28, 1), radius=ps.particle_radius, per_vertex_color=ps.rgb)

		if not is_pause:
			for i in range(iter_cnt):
				solver.step()
			# is_pause = True
			# ti.profiler.print_kernel_profiler_info()
			# ti.profiler.clear_kernel_profiler_info()
		# ti.profiler.print_kernel_profiler_info()

		scene.lines(box_vert, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=2.0)

		canvas.scene(scene)

		# gui.show(f'frame/{frame_cnt:06d}.png')
		frame_cnt += 1
		if is_output_gif:
			img = window.get_image_buffer_as_numpy()
			video_manager.write_frame(img)

		if is_output_ply:
			np_pos = np.reshape(ps.pos.to_numpy(), (ps.particle_num, 3))
			writer = ti.tools.PLYWriter(num_vertices=ps.particle_num)
			writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
			writer.add_vertex_rgba(
				np_rgba[:, 0], np_rgba[:, 1], np_rgba[:, 2], np_rgba[:, 3])
			writer.export_frame_ascii(frame_cnt, series_prefix)
		# print(frame_cnt)
		window.show()

	if is_output_gif:
		video_manager.make_video(gif=True, mp4=False)