# coding=utf-8

import time
import taichi as ti
import logging
import argparse
import importlib
import utils
import numpy as np
from ParticleSystem import ParticleSystem
from rigid_solver import rigid_solver

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
	start_time = time.time()
	window = ti.ui.Window('Window Title', res=(640, 640), pos=(150, 150))

	canvas = window.get_canvas()

	scene = ti.ui.Scene()

	camera = ti.ui.Camera()

	camera.position(*scene_config.get('cam_pos'))
	camera.lookat(*scene_config.get('cam_look_at'))
	camera.up(*scene_config.get('cam_up'))
	scene.set_camera(camera)

	ps = ParticleSystem(config)
	solver_name = solver_config.get('name')
	module = importlib.import_module(solver_name + '_solver')
	solver_ = getattr(module, solver_name + '_solver')
	solver = solver_(ps, config)
	rs = None
	if config.get('solid', {}):
		rs = rigid_solver(ps, config)

	frame_cnt = 0
	iter_cnt = solver_config.get('iter_cnt')

	np_rgba = np.reshape(ps.rgba.to_numpy(), (ps.particle_num, 4))
	series_prefix = './output/output'
	is_output_gif = scene_config.get('is_output_gif', False)
	is_output_ply = scene_config.get('is_output_ply', False)
	output_fps = scene_config.get('output_fps', 60)
	frame_time = 1.0 / output_fps
	video_manager = ti.tools.VideoManager(output_dir="./output", framerate=output_fps, automatic_build=False)
	is_pause = not scene_config.get('is_simulate', True)
	flag = 0
	render_fluid = True
	render_rigid = True
	output_frame_cnt = 0
	ply_cnt = 0
	t = 0.0
	simulation_time_consuming = 0
	pause_start_time = 0
	pause_time = 0
	while window.running:
		# Debug GUI

		if frame_cnt > 100000:
			break
		# if frame_cnt > 5200:
		# 	ps.active_rigid[None] = 1
		# 	if not flag:
		# 		flag = 1
		# 		ps.reset_grid()
		# 		ps.update_grid()
		# 		ps.init_rigid_particles_data()

		gui = window.get_gui()
		gui.text("frame_cnt: {}".format(frame_cnt))
		gui.text("delta time: {:.5f}".format(solver.delta_time[None]))
		gui.text("time: {:.4f}".format(t))
		if is_pause:
			gui.text("real time: {:.3f}".format(pause_start_time - pause_time - start_time))
		else:
			gui.text("real time: {:.3f}".format(time.time() - start_time - pause_time))
		gui.text("ply_cnt: {}".format(ply_cnt))
		gui.text("Pause: {}".format(is_pause))
		if window.is_pressed(ti.GUI.ESCAPE):
			break
		if window.is_pressed(ti.GUI.SPACE) and is_pause:
			pause_time += time.time() - pause_start_time
			is_pause = False
		if window.is_pressed('p') and not is_pause:
			pause_start_time = time.time()
			is_pause = True
		if window.is_pressed('f'):
			render_fluid = True
		if window.is_pressed('g'):
			render_fluid = False
		if window.is_pressed('r'):
			render_rigid = True
		if window.is_pressed('t'):
			render_rigid = False
		if window.is_pressed('c'):
			print('Camera position [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_position)))
			print('Camera look at [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_lookat)))
			print('Camera up [{}]'.format(', '.join('{:.2f}'.format(x) for x in camera.curr_up)))

		if window.is_pressed('b'):
			camera.position(*scene_config.get('cam_pos'))
			camera.lookat(*scene_config.get('cam_look_at'))
			camera.up(*scene_config.get('cam_up'))

		# Cam and light
		camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
		scene.set_camera(camera)
		scene.ambient_light((0.8, 0.8, 0.8))
		scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
		# if not is_pause:
		# 	solver.visualize_rho()
		if render_fluid:
			scene.particles(ps.fluid_particles.pos, color=(0.0, 0.28, 1), radius=ps.particle_radius, per_vertex_color=ps.fluid_particles.rgb)
		if render_rigid and config.get('solid', {}):
			scene.particles(ps.rigid_particles.pos, color=(1.0, 0.0, 0), radius=ps.particle_radius / 5, per_vertex_color=ps.rigid_particles.rgb)
		# scene.particles(rs.debug_particles, color=(1.0, 1.0, 1.0), radius=ps.particle_radius)
		# scene.particles(ps.boundary_particles.pos, color=(1.0, 1.0, 1.0), radius=ps.particle_radius)

		if not is_pause:
			for i in range(iter_cnt):
				solver.step()

			for i in range(iter_cnt):
				if rs and ps.active_rigid[None] == 1:
					rs.step()
			frame_cnt += 1
			t += iter_cnt * solver.delta_time[None]
			# ti.profiler.print_kernel_profiler_info()
			# ti.profiler.clear_kernel_profiler_info()
		# ti.profiler.print_kernel_profiler_info()

		scene.lines(box_vert, indices=box_lines_indices, color=(0.99, 0.68, 0.28), width=2.0)

		canvas.scene(scene)

		# gui.show(f'frame/{frame_cnt:06d}.png')

		if is_output_gif and (t / frame_time) > output_frame_cnt:
			img = window.get_image_buffer_as_numpy()
			video_manager.write_frame(img)
			output_frame_cnt += 1

		if is_output_ply and (t / frame_time) > ply_cnt:
			np_pos = np.reshape(ps.fluid_particles.pos.to_numpy(), (ps.particle_num, 3))
			writer = ti.tools.PLYWriter(num_vertices=ps.particle_num)
			writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
			writer.add_vertex_rgba(
				np_rgba[:, 0], np_rgba[:, 1], np_rgba[:, 2], np_rgba[:, 3])
			writer.export_frame_ascii(ply_cnt, series_prefix)
			ps.update_mesh_vextics()
			with open(f"output/obj_{ply_cnt:06}.obj", "w") as f:
				e = ps.mesh.export(file_type='obj')
				f.write(e)
			ply_cnt += 1
		# print(frame_cnt)
		window.show()

	if is_output_gif:
		video_manager.make_video(gif=True, mp4=True)

	print("Simulation time: {}".format(time.time() - start_time))