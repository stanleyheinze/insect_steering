
import pytry
import nengo
import numpy as np
import sys
sys.path.append('.')
import moth_world
import scipy.signal
import cx_rate
import central_complex
import moth_brain_nengo
import moth_brain_sigmoid


class PathIntegratorTrial(pytry.PlotTrial):
    def params(self):
        self.param('random walk variability', random_walk_sigma=0.1)
        self.param('time to stop random walk', t_switch=15)
        self.param('total simulation time', t_sim=40)
        self.param('sample every dt', sample_every=0.1)
        self.param('world dt', world_dt=0.1)
        self.param('brain (nengo|sigmoid|ideal)', brain='nengo')
        self.param('rotation scaling', rotation_scaling=8.0)
        self.param('path integrator output scaling low value', path_output_rescale_low=0.0)
        self.param('path integrator output scaling high value', path_output_rescale_high=1.0)
        
        
    def evaluate(self, p, plt):
        world = moth_world.World(x=0, y=0, heading=0,
                                 bounds_x=(-1000.0, 1000.0),
                                 bounds_y=(-1000.0, 1000.0),
                                 grid_size=(100,100),
                                 sensor_dist=50,
                                 trail_length=500,
                                 acceleration=0.2,
                                 drag=0.5,
                                 trail_dt=0.1)




        class PathIntegrator(object):
            def __init__(self, velocity_scale=1.0):
                self.velocity_scale = velocity_scale
                self.cx = cx_rate.CXRate(noise=0.1*p.world_dt,
                                         weight_noise=0.0,
                                         tn_prefs=np.pi/4.0)
                self.tb1 = np.zeros(central_complex.N_TB1)
                self.memory = 0.5 * np.ones(central_complex.N_CPU4)

            def update_cells(self, heading, velocity, tb1, memory, cx, filtered_steps=0.0):
                """Generate activity for all cells, based on previous activity and current
                motion."""
                # Compass
                tl2 = cx.tl2_output(heading)
                cl1 = cx.cl1_output(tl2)
                tb1 = cx.tb1_output(cl1, tb1)

                # Speed
                flow = cx.get_flow(heading, velocity, filtered_steps)
                tn1 = cx.tn1_output(flow)
                tn2 = cx.tn2_output(flow)

                # Update memory for distance just travelled
                memory = cx.cpu4_update(memory, tb1, tn1, tn2)
                cpu4 = cx.cpu4_output(memory)

                # Steer based on memory and direction
                cpu1 = cx.cpu1_output(tb1, cpu4)
                motor = cx.motor_output(cpu1)
                return tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor

            def make_path_integrator(self, filtered_steps=0.0):
                def update(t, x):
                    heading = x[0]
                    velocity = x[1]*self.velocity_scale, x[2]*self.velocity_scale
                    tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor = self.update_cells(
                        heading=heading, velocity=(velocity[0]*p.world_dt, velocity[1]*p.world_dt),
                        tb1=self.tb1, memory=self.memory,
                        cx=self.cx, 
                        filtered_steps=filtered_steps)            
                    self.tb1 = tb1
                    self.memory = memory

                    return np.interp(motor, [p.path_output_rescale_low, p.path_output_rescale_high], [0,1])
                return nengo.Node(update, size_in=3, size_out=2)

            def make_memory(self):
                def update(t):
                    return self.memory
                return nengo.Node(update)


        path = PathIntegrator(velocity_scale=1.0)

        model = nengo.Network()
        with model:
            #visualization = world.make_visualize()
            movement = world.make_movement(dt=p.world_dt)
            pos = world.make_position()
            p_pos = nengo.Probe(pos, sample_every=p.sample_every)

            random_walk = nengo.Node(nengo.processes.WhiteNoise(dist=nengo.dists.Gaussian(mean=0, std=p.random_walk_sigma)))

            h_and_v = world.make_heading_and_velocity()
            path_integrator = path.make_path_integrator()
            p_path_int = nengo.Probe(path_integrator, sample_every=p.sample_every)
            path_mem = path.make_memory()
            p_mem = nengo.Probe(path_mem, sample_every=p.sample_every)
            nengo.Connection(h_and_v, path_integrator, synapse=None)
            p_h_and_v = nengo.Probe(h_and_v, sample_every=p.sample_every)

            def mode_control(t, x):
                return x[0]*x[1] + x[2]*x[3]
            mode = nengo.Node(mode_control, size_in=4)
            nengo.Connection(random_walk, mode[0], synapse=None)

            if p.brain == 'nengo':
                brain = moth_brain_nengo.MothBrainNengo(noise=0.02, inhib=0.0)
            elif p.brain == 'sigmoid':
                brain = moth_brain_sigmoid.MothBrainSigmoid(weight_noise=0.01, sigmoid_noise=0.02, inhib=0.0)

            sensor_scaling = 1.0
            if p.brain == 'nengo':
                nengo.Connection(path_integrator[0], brain.inputL,
                                 transform=sensor_scaling, synapse=None)
                nengo.Connection(path_integrator[1], brain.inputR,
                                 transform=sensor_scaling, synapse=None)
                nengo.Connection(brain.turn, mode[2], transform=p.rotation_scaling, synapse=None)
            elif p.brain == 'sigmoid':
                nengo.Connection(path_integrator[0], brain.inputR,
                                 transform=sensor_scaling, synapse=None)
                nengo.Connection(path_integrator[1], brain.inputL,
                                 transform=sensor_scaling, synapse=None)
                nengo.Connection(brain.turn, mode[2], transform=p.rotation_scaling, synapse=None)
            elif p.brain == 'ideal':
                nengo.Connection(path_integrator[0], mode[2],
                                 transform=sensor_scaling*p.rotation_scaling, synapse=None)
                nengo.Connection(path_integrator[1], mode[2],
                                 transform=-sensor_scaling*p.rotation_scaling, synapse=None)
            else:
                raise Exception('unknown brain type: %s' % brain)



            def control_func(t):        
                if t < p.t_switch:
                    return 1,0
                else:
                    return 0,1
            control = nengo.Node(control_func)
            nengo.Connection(control, mode[[1, 3]])
            nengo.Connection(mode, movement, synapse=None, transform=0.1)

        sim = nengo.Simulator(model)
        sim.run(p.t_sim)

        if p.plt:
            path = sim.data[p_pos]
            index = int(p.t_switch / p.sample_every)
            plt.plot(path[:index,0], path[:index,1], c='g')
            plt.plot(path[index:,0], path[index:,1], c='k')
        
        return dict(
            path = sim.data[p_pos],
            h_and_v = sim.data[p_h_and_v],
            path_mem = sim.data[p_mem],
            path_int = sim.data[p_path_int],
        )
            
