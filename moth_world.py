import nengo
import numpy as np

import pompy.models
import pompy.processors

import bee_simulator

import base64
try:
    from cStringIO import StringIO       # Python 2
except ImportError:
    from io import BytesIO as StringIO   # Python 3
from PIL import Image

class World(object):
    def __init__(self, x, y, heading,
                    bounds_x = (-0.5, 0.5),
                    bounds_y = (-0.2, 0.2),
                    grid_size=(100, 40),
                    source_location = (-0.5, 0),
                    acceleration=0.2,
                    drag=0.5,
                    puff_release_rate=100,
                    puff_init_rad=0.001,
                    centre_rel_diff_scale=1.5,
                    sensor_dist = 0.03,
                    trail_length = 0,
                    trail_dt = 0.1,
                    plume_dt=0.005,
                    precompute_time=0.0,
                    max_concentration=1e5):
        
        self.heading = heading
        self.acceleration = acceleration
        self.velocity = [0,0]
        self.drag = drag        
        self.conc = None
        self.trail_length = trail_length
        self.trail_dt = trail_dt
        self.plume_dt = plume_dt
        self.last_trail_time = None
        self.trail = []
        
        self.x = x
        self.y = y
        
        
        self.max_concentration = max_concentration
        
        self.sensor_angle = 45*np.pi/180
        self.sensor_dist = sensor_dist
        
        
        
        # Define simulation region
        wind_region = pompy.models.Rectangle(0., -2., 10., 2.)
        sim_region = pompy.models.Rectangle(bounds_x[0], bounds_y[0], bounds_x[1], bounds_y[1])
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
      
        
        # Set up wind model
        wind_grid_dim_x = 21
        wind_grid_dim_y = 11
        wind_vel_x_av = 2.
        wind_vel_y_av = 0.
        self.wind_model = pompy.models.WindModel(
            wind_region, wind_grid_dim_x, wind_grid_dim_y, 
            wind_vel_x_av, wind_vel_y_av)
            
        # Set up plume model
        centre_rel_diff_scale = 1.5
        self.plume_model = pompy.models.PlumeModel(
            sim_region,
            (source_location[0], source_location[1], 0),
            self.wind_model,
            centre_rel_diff_scale=centre_rel_diff_scale,
            puff_release_rate=puff_release_rate, 
            puff_init_rad=puff_init_rad)
        
        for i in range(int(precompute_time/plume_dt)):
            self.wind_model.update(plume_dt)
            self.plume_model.update(plume_dt)
        
        # Create a concentration array generator
        array_z = 0.01
        self.grid_size = grid_size
        puff_mol_amount = 1.
        self.array_gen = pompy.processors.ConcentrationArrayGenerator(
            sim_region, array_z, grid_size[0], self.grid_size[1], puff_mol_amount)
    
    def make_sensor(self):
        self.last_update = -9999
        update_rate = self.plume_dt
        dt = self.plume_dt
        def update(t):
            if t >= self.last_update + update_rate:
                self.last_update = t
                self.wind_model.update(dt)
                self.plume_model.update(dt)
                self.conc = self.array_gen.generate_single_array(self.plume_model.puff_array)    
                
            lx, ly, rx, ry = self.sensor_pos()
            lx, ly = self.grid_location(lx, ly)
            rx, ry = self.grid_location(rx, ry)
            lx = int(lx)
            rx = int(rx)
            ly = int(ly)
            ry = int(ry)
            if 0 <= lx < self.grid_size[0] and 0 <= ly < self.grid_size[1]:
                lc = self.conc[lx, ly]
            else:
                lc = 0
            if 0 <= rx < self.grid_size[0] and 0 <= ry < self.grid_size[1]:
                rc = self.conc[rx, ry]
            else:
                rc = 0
            
            return (np.clip(lc/self.max_concentration, 0, 1), 
                    np.clip(rc/self.max_concentration, 0, 1))
        return nengo.Node(update)
    
    def sensor_pos(self):
        left_x = self.x + self.sensor_dist*np.cos(self.heading-self.sensor_angle)
        left_y = self.y + self.sensor_dist*np.sin(self.heading-self.sensor_angle)
        right_x = self.x + self.sensor_dist*np.cos(self.heading+self.sensor_angle)
        right_y = self.y + self.sensor_dist*np.sin(self.heading+self.sensor_angle)
        return left_x, left_y, right_x, right_y
        
    
    def make_visualize(self):
        self.last_visualize = -9999
        visualize_rate = 0.01
        def update(t):
            if t < self.last_visualize + visualize_rate:
                return
            self.last_visualize = t
            #conc = self.array_gen.generate_single_array(self.plume_model.puff_array)    
            conc = self.conc
            if self.conc is not None:
                limits = [0, self.max_concentration]
                conc = (conc - limits[0])/(limits[1] - limits[0])
                conc = np.clip(conc * 255, 0, 255)
                conc = 255 - conc
                conc = conc.astype('uint8')
            
                png = Image.fromarray(conc[:,:].T)
                buffer = StringIO()
                png.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

                img = '''<image width="100%%" height="100%%"
                          xlink:href="data:image/png;base64,%s" 
                          style="image-rendering: pixelated;"/>
                      ''' % img_str
            else:
                img = ''
                                  
            template = '''
                    <svg width="100%%" height="100%%" viewbox="0 0 %d %d">
                        %s
                        %s
                        %s
                    </svg>'''                        
            
            xx, yy = self.grid_location(self.x, self.y)
            lx, ly, rx, ry = self.sensor_pos()
            lx, ly = self.grid_location(lx, ly)
            rx, ry = self.grid_location(rx, ry)
            
            
            body = '''<circle cx="%f" cy="%f" r="1" fill="blue"/>
                      <path d="M%f %f L%f %f L%f %f" stroke="red" fill="none"/>''' % (
                          xx, yy, lx, ly, xx, yy, rx, ry)


            if len(self.trail) > 0:
                pts = ['%f %f' % self.grid_location(t[0], t[1]) for t in self.trail]
                path = '''<path d="M%s" stroke="#ccc" fill="none"/>''' % ' L'.join(pts)
            else:
                path = ''
        
        
            update._nengo_html_ = template % (self.grid_size[0], self.grid_size[1],
                                              img, path, body)    
        return nengo.Node(update)
            
    def grid_location(self, x, y):
        tx = (x - self.bounds_x[0])/(self.bounds_x[1] - self.bounds_x[0])
        ty = (y - self.bounds_y[0])/(self.bounds_y[1] - self.bounds_y[0])
        return tx*self.grid_size[0], ty*self.grid_size[1]
        
    def make_movement(self, dt=0.001):
        def update(t, x):
            self.heading, self.velocity = bee_simulator.get_next_state(
                            heading=self.heading,
                            velocity=self.velocity,
                            rotation=x[0]*dt,
                            acceleration=self.acceleration,
                            drag=self.drag)
            self.x += self.velocity[1]*dt
            self.y += self.velocity[0]*dt

            if self.trail_length > 0:
                if self.last_trail_time is None or t > self.last_trail_time + self.trail_dt:
                    self.trail.append((self.x, self.y))
                    if len(self.trail) > self.trail_length:
                        self.trail = self.trail[1:]
                    self.last_trail_time = t
            return x
        return nengo.Node(update, size_in=1)
        
    def make_position(self):
        def update(t):
            return self.x, self.y
        return nengo.Node(update)

    def make_heading_and_velocity(self):
        def update(t):
            return self.heading, self.velocity[0], self.velocity[1]
        return nengo.Node(update, size_in=0, size_out=3)
