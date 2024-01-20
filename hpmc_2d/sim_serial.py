from __future__ import annotations

import itertools
import math
import os

import gsd.hoomd
import hoomd
import numpy as np
from hoomd.hpmc.integrate import SimplePolygon, HPMCIntegrator
from loguru import logger

from hpmc_2d.polygon import render_plato

_cpu = hoomd.device.CPU()
# assert hoomd.version.mpi_enabled
_rank = _cpu.communicator.rank
_pid = os.getpid()


class Sim2DSerial:

    def __init__(
            self,
            vertices_file: str | os.PathLike,
            shape_area: float,
            m: int = 6, spacing: float = 100, name: str = "sim_2d", n_select: int = 4,
    ):
        """
        hard particle monte carlo simulation with 2D shapes

        :param vertices_file: path to the npy file, it contains all vertices of the shape
        :param shape_area: 2d area of the shape
        :param m: 2 * m**2 is the total number of particles for 2D
        :param spacing: initial spacing between shapes
        :param name: name of the simulation, used as prefixes for gsd files
        :param n_select: number of shapes selected in an update
        """
        self.shape_area = shape_area
        self.vertices_file = vertices_file
        self.name = name
        self.spacing = spacing
        self.m = m

        self.n_particles = 0
        self.box_size = 0

        self.n_select = n_select

        self.vertices = np.load(self.vertices_file)

        self.init_gsd = f"{self.name}_init.gsd"
        self.rand_gsd = f"{self.name}_rand.gsd"
        self.compress_gsd = f"{self.name}_compress.gsd"
        self.eq_gsd = f"{self.name}_eq.gsd"
        self.eq_traj = f"{self.name}_eq_traj.gsd"

        self.init_png = f"{self.name}_init.png"
        self.rand_png = f"{self.name}_rand.png"
        self.compress_png = f"{self.name}_compress.png"
        self.eq_png = f"{self.name}_eq.png"

    @property
    def mc(self):
        """ simple polygon integrator """
        mc = SimplePolygon(nselect=self.n_select)
        mc.shape['A'] = dict(vertices=self.vertices)
        return mc

    def run_init(self):
        """
        run initialization, export a frame to the current work folder,
        box size is determined by number of particles and spacing
        """
        self.n_particles = 2 * self.m ** 2

        k = math.ceil(self.n_particles ** (1 / 2))
        self.box_size = k * self.spacing
        x = np.linspace(-self.box_size / 2, self.box_size / 2, k, endpoint=False)
        position = np.array(list(itertools.product(x, repeat=2)))
        position = np.hstack((position, np.zeros((position.shape[0], 1))))
        position = position[0:self.n_particles]
        orientation = [(1, 0, 0, 0)] * self.n_particles

        frame = gsd.hoomd.Frame()
        frame.particles.N = self.n_particles
        frame.particles.position = position
        frame.particles.orientation = orientation
        frame.particles.typeid = [0] * self.n_particles
        frame.particles.types = ['A']
        frame.configuration.box = [self.box_size, self.box_size, 0, 0, 0, 0]
        with gsd.hoomd.open(name=self.init_gsd, mode='w') as f:
            f.append(frame)
        logger.info(f"initial state written to: {self.init_gsd}")
        logger.info(f"n particles: {self.n_particles}")
        logger.info(f"box size: {self.box_size}")
        logger.info(f"spacing: {self.spacing}")

    def run_rand(self, rand_n_timesteps: int = 1000):
        """ run randomization, export a frame to the current work folder """
        simulation = hoomd.Simulation(device=_cpu, seed=42)
        simulation.operations.integrator = self.mc
        simulation.create_state_from_gsd(filename=self.init_gsd)
        self.state_draw(simulation, self.init_png)

        simulation.run(rand_n_timesteps)
        hoomd.write.GSD.write(state=simulation.state, mode='xb', filename=self.rand_gsd)
        self.mc_report(simulation.operations.integrator)
        self.state_draw(simulation, self.rand_png)
        initial_volume_fraction = (simulation.state.N_particles * self.shape_area / simulation.state.box.volume)
        logger.info(f"init box area: {simulation.state.box.volume}")
        logger.info(f"init area fraction: {initial_volume_fraction}")

    def run_compress(
            self,
            final_volume_fraction: float = 0.61,
            compress_every_n_steps: int = 10,
            compress_movesize_tune_every_n_steps: int = 10,
            compress_n_timesteps: int = 10000,
    ):
        """
        run compression, raise RuntimeError if compression not completed in the given timesteps

        :param final_volume_fraction: target volume,
        the volume fraction is defined as box_size/(particle_area*n_particles)
        :param compress_every_n_steps: frequency of compression operation
        :param compress_movesize_tune_every_n_steps: tuner frequency, should equal `compress_every_n_steps`
        :param compress_n_timesteps: total timesteps for compression
        :return:
        """
        simulation = hoomd.Simulation(device=_cpu, seed=43)
        simulation.operations.integrator = self.mc
        simulation.create_state_from_gsd(filename=self.rand_gsd)
        initial_volume_fraction = (simulation.state.N_particles * self.shape_area / simulation.state.box.volume)

        logger.info(f"init box area: {simulation.state.box.volume}")
        logger.info(f"init area fraction: {initial_volume_fraction}")

        initial_box = simulation.state.box
        final_box = hoomd.Box.from_box(initial_box)
        final_box.volume = simulation.state.N_particles * self.shape_area / final_volume_fraction
        logger.info(f"final area fraction set to: {final_volume_fraction}")
        compress = hoomd.hpmc.update.QuickCompress(
            trigger=hoomd.trigger.Periodic(compress_every_n_steps), target_box=final_box
        )
        simulation.operations.updaters.append(compress)

        periodic = hoomd.trigger.Periodic(compress_movesize_tune_every_n_steps)
        # 20% acceptance rate suggested by doc
        tune = hoomd.hpmc.tune.MoveSize.scale_solver(moves=['a', 'd'],
                                                     target=0.2,
                                                     trigger=periodic,
                                                     max_translation_move=0.2,
                                                     max_rotation_move=0.2)
        simulation.operations.tuners.append(tune)

        while not compress.complete and simulation.timestep < compress_n_timesteps:
            simulation.run(compress_n_timesteps)

        logger.info(f"compress # of timestep: {simulation.timestep}")

        hoomd.write.GSD.write(state=simulation.state, mode='xb', filename=self.compress_gsd)

        self.state_draw(simulation, self.compress_png)

        if not compress.complete:
            raise RuntimeError("Compression failed to complete")

    def run_eq(
            self,
            eq_write_frame_every_n_steps: int = 1000,
            eq_movesize_tune_every_n_steps: int = 100,
            eq_movesize_tune_pre_eq_n_steps: int = 5000,
            eq_n_timesteps: int = 100000,
    ):
        """
        run equilibration

        :param eq_write_frame_every_n_steps:
        :param eq_movesize_tune_every_n_steps:
        :param eq_movesize_tune_pre_eq_n_steps:
        :param eq_n_timesteps:
        :return:
        """

        simulation = hoomd.Simulation(device=_cpu, seed=44)
        simulation.operations.integrator = self.mc
        simulation.create_state_from_gsd(filename=self.compress_gsd)

        gsd_writer = hoomd.write.GSD(filename=self.eq_traj,
                                     trigger=hoomd.trigger.Periodic(eq_write_frame_every_n_steps),
                                     mode='xb')
        simulation.operations.writers.append(gsd_writer)

        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=['a', 'd'],
            target=0.2,
            trigger=hoomd.trigger.And([
                hoomd.trigger.Periodic(eq_movesize_tune_every_n_steps),
                hoomd.trigger.Before(simulation.timestep + eq_movesize_tune_pre_eq_n_steps)
            ]))
        simulation.operations.tuners.append(tune)
        simulation.run(eq_movesize_tune_pre_eq_n_steps)
        logger.info(f"target accept rate is: 0.2")
        logger.info(f"running 100 more steps to verify accept rate")
        simulation.run(100)
        self.mc_report(simulation.operations.integrator)

        logger.info(f"running the actual eq")
        simulation.run(eq_n_timesteps)
        hoomd.write.GSD.write(state=simulation.state, mode='xb', filename=self.eq_gsd)
        self.state_draw(simulation, self.eq_png)
        gsd_writer.flush()

    def pre_run(self):
        for file in [self.init_gsd, self.rand_gsd, self.compress_gsd, self.eq_traj, self.eq_gsd]:
            try:
                os.remove(file)
                logger.info(f"remove existing file at the beginning of a new run: {file}")
            except FileNotFoundError:
                pass

    def run(
            self,

            rand_n_timesteps: int = 1000,

            final_volume_fraction: float = 0.61,
            compress_every_n_steps: int = 10,
            compress_movesize_tune_every_n_steps: int = 10,
            compress_n_timesteps: int = 10000,

            eq_write_frame_every_n_steps: int = 1000,
            eq_movesize_tune_every_n_steps: int = 100,
            eq_movesize_tune_pre_eq_n_steps: int = 5000,
            eq_n_timesteps: int = 100000,
    ):
        self.pre_run()

        logger.info("INIT started")
        self.run_init()
        logger.info("INIT finished")

        logger.info("RAND started")
        self.run_rand(rand_n_timesteps=rand_n_timesteps)
        logger.info("RAND finished")

        logger.info("COMPRESS started")

        self.run_compress(
            final_volume_fraction, compress_every_n_steps, compress_movesize_tune_every_n_steps,
            compress_n_timesteps
        )

        logger.info("COMPRESS finished")

        logger.info("EQUILIBRATE started")
        self.run_eq(
            eq_write_frame_every_n_steps, eq_movesize_tune_every_n_steps,
            eq_movesize_tune_pre_eq_n_steps, eq_n_timesteps
        )
        logger.info("EQUILIBRATE finished")

    @staticmethod
    def mc_report(mc: HPMCIntegrator):
        logger.info("HPMC reporting:")
        n_trans = mc.translate_moves[0]
        n_trans_ratio = n_trans / sum(mc.translate_moves)
        n_rot = mc.rotate_moves[0]
        n_rot_ratio = n_rot / sum(mc.rotate_moves)
        logger.info(f"# of accepted translation moves: {n_trans} ({n_trans_ratio})")
        logger.info(f"# of accepted rotation moves: {n_rot} ({n_rot_ratio})")
        logger.info(f"# of overlaps: {mc.overlaps}")

    def snapshot_draw(self, snapshot: hoomd.Snapshot, filename):
        pos = snapshot.particles.position
        ori = snapshot.particles.orientation
        scene = render_plato(self.vertices, pos, ori)
        scene.save(filename)

    def state_draw(self, simulation: hoomd.Simulation, filename):

        snapshot = simulation.state.get_snapshot()
        pos = snapshot.particles.position
        ori = snapshot.particles.orientation
        # try:
        #     snapshot = simulation.state.get_snapshot()
        # except RuntimeError:
        #     return
        scene = render_plato(self.vertices, pos, ori)
        scene.save(filename)
