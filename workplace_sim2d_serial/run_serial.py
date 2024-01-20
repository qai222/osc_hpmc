from hpmc_2d import VERTICES_FILE_TIPSPn
from hpmc_2d.sim_serial import Sim2DSerial

if __name__ == '__main__':
    SIM = Sim2DSerial(vertices_file=VERTICES_FILE_TIPSPn, shape_area=144.05592, name="serial2d")
    SIM.run(
        rand_n_timesteps=1000,
        final_volume_fraction=0.61,
        compress_every_n_steps=10,
        compress_movesize_tune_every_n_steps=10,
        compress_n_timesteps=10000,
        eq_write_frame_every_n_steps=1000,
        eq_movesize_tune_every_n_steps=100,
        eq_movesize_tune_pre_eq_n_steps=5000,
        eq_n_timesteps=100000,
    )
