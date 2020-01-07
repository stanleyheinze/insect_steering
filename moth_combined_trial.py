import nengo
import numpy as np
import pytry
import scipy.ndimage.filters


import sys
if '.' not in sys.path:
    sys.path.append('.')

import moth_world
import moth_brain_nengo
import moth_brain_sigmoid
import moth_brain_nengo_GIIC
import moth_brain_sigmoid_GIIC

class SimulationFail(Exception):
    pass
class SimulationSuccess(Exception):
    pass

class MothTrial(pytry.PlotTrial):
    def params(self):
        self.param('starting x location', x=0.3)
        self.param('starting y location', y=-0.0)
        self.param('min x value', min_x=-0.5)
        self.param('max x value', max_x=0.5)
        self.param('min y value', min_y=-0.2)
        self.param('max y value', max_y=0.2)
        self.param('number of grid squares for x', grid_count_x=100)
        self.param('number of grid squares for y', grid_count_y=40)
        self.param('source location x', source_x=-0.5)
        self.param('source location y', source_y=0.0)
        self.param('stop simulation when this close to target', stopping_distance=0.01)
        self.param('puff release rate', puff_release_rate=50)
        self.param('puff initial radius', puff_init_rad=0.1)
        self.param('maximum detectable puff concentration', puff_max_conc=100)

        self.param('starting heading', heading=0.0)
        self.param('acceleration', acceleration=0.2)
        self.param('drag', drag=0.5)
        self.param('brain type (nengo or sigmoid or nengo_GIIC or sigmoid_GIIC)', brain='sigmoid_GIIC')
        self.param('sensor scaling', sensor_scaling=3.0)
        self.param('rotation scaling', rotation_scaling=10.0)
        self.param('maximum time to run for', max_time=10.0)
        self.param('noise', noise=0.0)
        self.param('strength of turning into wind', turn_into_wind_strength=0.1)

        self.param('turning bias', turning_bias=0.0)
        self.param('inhibition strength', inhib_strength=3.0)
        
        self.param('precompute time', precompute_time=0.0)
        self.param('save path information', save_path=True)
        self.param('save sensor input', save_sensor=False)

        self.param('number of neurons (per population)', n_neurons=100)

    def evaluate(self, p, plt):

        model = nengo.Network()
        with model:
            world = moth_world.World(x=p.x, y=p.y, heading=p.heading+np.pi,
                                     acceleration=p.acceleration,
                                     puff_release_rate=p.puff_release_rate,
                                     puff_init_rad=p.puff_init_rad,
                                     bounds_x=(p.min_x, p.max_x),
                                     bounds_y=(p.min_y, p.max_y),
                                     max_concentration=p.puff_max_conc,
                                     source_location=(p.source_x, p.source_y),
                                     grid_size=(p.grid_count_x, p.grid_count_y),
                                     precompute_time=p.precompute_time,
                                     drag=p.drag)
            
            rotation = world.make_movement()
            sensor = world.make_sensor()
            pos = world.make_position()
            heading_and_vel = world.make_heading_and_velocity()
            #visual = world.make_visualize()

            turning_bias = nengo.Node(p.turning_bias)
            nengo.Connection(turning_bias, rotation)
            
            if p.brain == 'nengo':
                brain = moth_brain_nengo.MothBrainNengo(noise=p.noise, inhib=p.inhib_strength, N=p.n_neurons)
            elif p.brain == 'sigmoid':
                brain = moth_brain_sigmoid.MothBrainSigmoid(weight_noise=0.01, sigmoid_noise=p.noise, inhib=p.inhib_strength)
            elif p.brain == 'nengo_GIIC':
                brain = moth_brain_nengo_GIIC.MothBrainNengo(noise=p.noise, inhib=p.inhib_strength, N=p.n_neurons)
            elif p.brain == 'sigmoid_GIIC':
                brain = moth_brain_sigmoid_GIIC.MothBrainSigmoid(weight_noise=0.01, sigmoid_noise=p.noise, inhib=p.inhib_strength)
            
            nengo.Connection(sensor[0], brain.inputR, transform=p.sensor_scaling, synapse=None)
            nengo.Connection(sensor[1], brain.inputL, transform=p.sensor_scaling, synapse=None)
            nengo.Connection(brain.turn, rotation, transform=p.rotation_scaling, synapse=None)

            def upwind_func(t):
               heading = world.heading

               wind_source_dir = np.pi
               angle_difference = wind_source_dir - heading
               while angle_difference > np.pi:
                   angle_difference -= np.pi*2
               while angle_difference < -np.pi:
                   angle_difference += np.pi*2
                    
               return angle_difference*p.turn_into_wind_strength
            upwind = nengo.Node(upwind_func)
            nengo.Connection(upwind, rotation)


            def check_bounds(t):
                if world.x < p.min_x or world.x > p.max_x:
                    raise SimulationFail()
                if world.y < p.min_y or world.y > p.max_y:
                    raise SimulationFail()
                delta = [world.x - p.source_x, world.y - p.source_y]
                if np.linalg.norm(delta) < p.stopping_distance:
                    raise SimulationSuccess()

            nengo.Node(check_bounds)

            p_pos = nengo.Probe(pos)
            p_heading = nengo.Probe(heading_and_vel[0])

            if p.save_sensor:
                p_sensor = nengo.Probe(sensor)

        sim = nengo.Simulator(model, progress_bar=False)
        try:
            sim.run(p.max_time, progress_bar=True)
            success = False
        except SimulationSuccess:
            success = True
        except SimulationFail:
            success = False
            
        
        dt = sim.dt
        path = sim.data[p_pos]
        heading = sim.data[p_heading][:,0]
        heading_s = scipy.ndimage.filters.gaussian_filter(heading, sigma=0.5/dt)
        rotation_rate = np.diff(heading_s) * 180 / np.pi / dt

        turn_start = []
        turn_end = []
        turn_angle = []
        turn_duration = []
        turn_velocity = []

        turning = (rotation_rate > 5) | (rotation_rate<-5)
        for i in np.where(np.diff(turning.astype(int))==1)[0]:
            ends = np.where(turning[i+1:]==0)[0]
            if len(ends) == 0:
                j = len(turning)
            else:
                j = i+ends[0]

            if j > i + 500:  # must be 0.5s turn
                d_heading = (heading_s[j] - heading_s[i])*180 / np.pi
                if np.abs(d_heading) > 30:
                    turn_start.append(i*dt)
                    turn_end.append(j*dt)
                    turn_angle.append(d_heading)
                    turn_duration.append((j-i)*dt)
                    turn_velocity.append(turn_angle[-1]/turn_duration[-1])
        turn_duration = np.array(turn_duration)
        turn_angle = np.array(turn_angle)
        turn_velocity = np.array(turn_velocity)
        turn_start = np.array(turn_start)
        turn_end = np.array(turn_end)
        

        if plt:
            plt.subplot(1, 2, 1)
            xy = sim.data[p_pos]
            plt.plot(xy[:,0], xy[:,1])
            plt.xlim(p.min_x, p.max_x)
            plt.ylim(p.min_y, p.max_y)
            plt.scatter([p.x], [p.y], marker='x')
            plt.scatter([p.source_x], [p.source_y], marker='o')
            plt.subplot(1, 2, 2)
            t = np.arange(len(rotation_rate))*dt
            plt.plot(t, rotation_rate)   
            plt.axhline(5, ls='--')
            plt.axhline(-5, ls='--')
            for i in range(len(turn_start)):
                plt.axvline(turn_start[i], c='k')
                plt.axvline(turn_end[i], c='k')
                plt.plot([turn_start[i], turn_end[i]], [turn_velocity[i], turn_velocity[i]], c='k')
                plt.ylabel('rotation rate (deg/sec)')

        time_taken = sim.n_steps * sim.dt
        pos = sim.data[p_pos][-1]
        delta = np.diff(sim.data[p_pos], axis=0)
        travel_distance = np.sum(np.linalg.norm(delta, axis=1))

        distance = np.linalg.norm(pos - (p.source_x, p.source_y))
        
        cw = turn_angle > 0
        acw = turn_angle < 0
        
        return dict(
                time_taken=time_taken,
                pos_x=pos[0],
                pos_y=pos[1],
                distance=distance,
                travel_distance=travel_distance,
                path=sim.data[p_pos] if p.save_path else None,
                path_heading=sim.data[p_heading] if p.save_path else None,
                success=success,
                turn_start=turn_start,
                turn_end=turn_end,
                turn_angle=turn_angle,
                turn_duration=turn_duration,
                turn_velocity=turn_velocity,
                total_turn_duration = np.sum(turn_duration),
                total_turn_duration_cw = np.sum(turn_duration[cw]),
                total_turn_duration_acw = np.sum(turn_duration[acw]),        
                mean_turn_velocity = np.mean(np.abs(turn_velocity)),
                mean_turn_velocity_cw = np.mean(np.abs(turn_velocity[cw])),
                mean_turn_velocity_acw = np.mean(np.abs(turn_velocity[acw])),
                total_turn_angle = np.sum(np.abs(turn_angle)),
                total_turn_angle_cw = np.sum(np.abs(turn_angle[cw])),
                total_turn_angle_acw = np.sum(np.abs(turn_angle[acw])),
                sensor_data = sim.data[p_sensor] if p.save_sensor else None,
                )






