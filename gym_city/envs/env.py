from gym import core, spaces
from gym.utils import seeding
from collections import OrderedDict
import numpy as np
import math

import sys
if sys.version_info[0] >= 3:
    import gi
    gi.require_version('Gtk', '3.0')
    from gi.repository import Gtk as gtk
    from .tilemap import TileMap, zoneFromInt
    from .corecontrol import MicropolisControl
else:
    import gtk
    from tilemap import TileMap, zoneFromInt
    from corecontrol import MicropolisControl
import time

class MicropolisEnv(core.Env):

    def __init__(self, MAP_X=20, MAP_Y=20, PADDING=0):
        self.SHOW_GUI=False
        self.start_time = time.time()
        self.print_map = False
        self.num_episode = 0
        self.max_static = 0
        self.player_step = False
        self.static_player_builds = False
    ### MIXED
        self.city_trgs = OrderedDict({
                'res_pop': 500,
                'com_pop': 50,
                'ind_pop': 50,
                'traffic': 2000,
                # i believe one plant is worth 12, the other 16?
                'num_plants': 14,
                'mayor_rating': 100
                })
        self.trg_param_vals = np.array([v for v in self.city_trgs.values()])
        self.param_bounds = OrderedDict({
                'res_pop': (0, 750),
                'com_pop': (0, 100),
                'ind_pop': (0, 100),
                'traffic': (0, 2000),
                'num_plants': (0, 100),
                'mayor_rating': (0, 100)
                })
        self.weights = OrderedDict({
                'res_pop': 1,
                'com_pop': 1,
                'ind_pop': 1,
                'traffic': 1,
                'num_plants': 0,
                'mayor_rating': 0,
                })

        self.num_params = 6
        # not necessarily true but should take care of most cases
        self.max_loss = 0
        i = 0
        self.param_ranges = []
        for param, (lb, ub) in self.param_bounds.items():
            weight = self.weights[param]
            rng = abs(ub - lb)
            self.param_ranges += [rng]
            if i < self.num_params:
                self.max_loss += rng * weight
                i += 1
   ### MIXED
       #self.city_trgs = {
       #        'res_pop': 1,
       #        'com_pop': 4,
       #        'ind_pop': 4,
       #        'traffic': 0.2,
       #        'num_plants': 0,
       #        'mayor_rating': 0}
  ### Traffic
       #self.city_trgs = {
       #        'res_pop': 1,
       #        'com_pop': 4,
       #        'ind_pop': 4,
       #        'traffic': 5,
       #        'num_plants': 0,
       #        'mayor_rating':0
       #        }
        self.city_metrics = {}
        self.max_reward = 100
       #self.setMapSize((MAP_X, MAP_Y), PADDING)

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        np.random.seed(seed)
        return [seed1, seed2]

    def setMapSize(self, size, **kwargs):
        '''Do most of the actual initialization.
        '''
        self.pre_gui(size, **kwargs)
        #TODO: this better
        if hasattr(self, 'micro'):
            self.micro.reset_params(size)
        else:
            self.micro = MicropolisControl(self, self.MAP_X, self.MAP_Y, self.PADDING,
                rank=self.rank, power_puzzle=self.power_puzzle, gui=self.render_gui)
        self.city_metrics = self.get_city_metrics()
        self.last_city_metrics = self.city_metrics
        self.post_gui()

    def pre_gui(self, size, max_step=None, rank=0, print_map=False,
            PADDING=0, static_builds=True, parallel_gui=False,
            render_gui=False, empty_start=True, simple_reward=False,
            power_puzzle=False, record=False, traffic_only=False, random_builds=False, poet=False, **kwargs):
        self.PADDING = PADDING
        self.rank = rank
        self.render_gui = render_gui
        self.random_builds = random_builds
        self.traffic_only = traffic_only
        if record: raise NotImplementedError
        if max_step is None:
            max_step = size * size
        self.max_step = max_step
        self.empty_start = empty_start
        self.simple_reward = simple_reward
        self.power_puzzle = power_puzzle
        if type(size) == int:
            self.MAP_X = size
            self.MAP_Y = size
        else:
            self.MAP_X = size[0]
            self.MAP_Y = size[1]
        self.obs_width = self.MAP_X + PADDING * 2
        self.static_builds = True
        self.poet = poet
        self.print_map = print_map

    def post_gui(self):
        self.win1 = self.micro.win1
        self.micro.SHOW_GUI=self.SHOW_GUI
        self.num_step = 0
        self.minFunds = 0
        self.initFunds = self.micro.init_funds
        self.num_tools = self.micro.num_tools
        self.num_zones = self.micro.num_zones
        # res, com, ind pop, demand
        self.num_scalars = 6
        self.num_density_maps = 3
        num_user_features = 1 # static builds
        # traffic, power, density
        print('num map features: {}'.format(self.micro.map.num_features))
        self.num_obs_channels = self.micro.map.num_features + self.num_scalars \
                + self.num_density_maps + num_user_features
        if self.poet:
            self.num_obs_channels += len(self.city_trgs)
        #ac_low = np.zeros((3))
       #ac_high = np.array([self.num_tools - 1, self.MAP_X - 1, self.MAP_Y - 1])
       #self.action_space = spaces.Box(low=ac_low, high=ac_high, dtype=int)
        self.action_space = spaces.Discrete(self.num_tools * self.MAP_X * self.MAP_Y)
        self.last_state = None
        self.metadata = {'runtime.vectorized': True}
        low_obs = np.full((self.num_obs_channels, self.MAP_X, self.MAP_Y), fill_value=-1)
        high_obs = np.full((self.num_obs_channels, self.MAP_X, self.MAP_Y), fill_value=1)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype = float)
        self.state = None
        self.intsToActions = {}
        self.actionsToInts = np.zeros((self.num_tools, self.MAP_X, self.MAP_Y))
        self.mapIntsToActions()
        self.last_pop = 0
        self.last_num_roads = 0
#       self.past_actions = np.full((self.num_tools, self.MAP_X, self.MAP_Y), False)
        self.auto_reset = True
        self.mayor_rating = 50
        self.last_mayor_rating = self.mayor_rating
        self.last_priority_road_net_size = 0
        self.display_city_trgs()
        if self.render_gui and self.rank == 0:
            self.render()

    def get_param_bounds(self):
        return self.param_bounds

    def display_city_trgs(self):
        if self.win1 is not None:
            self.win1.agentPanel.displayTrgs(self.city_trgs)
        return self.city_trgs


    def mapIntsToActionsChunk(self):
        ''' Unrolls the action vector into spatial chunks (does this matter empirically?).'''
        w0 = 20
        w1 = 10
        i = 0
        for j0 in range(self.MAP_X // w0):
            for k0 in range(self.MAP_Y // w0):
                for j1 in range(w0 // w1):
                    for k1 in range(w0 // w1):
                        for z in range(self.num_tools):
                            for x in range(j0 * w0 + j1*w1,
                                    j0 * w0 + (j1+1)*w1):
                                for y in range(k0 * w0 + k1*w1,
                                        k0 * w0 + (k1+1)*w1):
                                    self.intsToActions[i] = [z, x, y]
                                    i += 1

    def mapIntsToActions(self):
        ''' Unrolls the action vector in the same order as the pytorch model
        on its forward pass.'''
        chunk_width = 1
        i = 0
        for z in range(self.num_tools):
            for x in range(self.MAP_X):
                for y in range(self.MAP_Y):
                        self.intsToActions[i] = [z, x, y]
                        self.actionsToInts[z, x, y] = i
                        i += 1
        print('len of intsToActions: {}\n num tools: {}'.format(len(self.intsToActions), self.num_tools))

    def randomStep(self):
        self.step(self.action_space.sample())

    def close(self):
        self.micro.close()

    def randomStaticStart(self):
        num_static = int(self.MAP_X * self.MAP_Y / 10)
        lst_epi = 500
#       num_static = math.ceil(((lst_epi - self.num_episode) / lst_epi) * num_static)
#       num_static = max(0, max_static)
        self.micro.setFunds(self.micro.init_funds)
        if num_static > 0:
            num_static = self.np_random.integers(0, num_static + 1)
        for i in range(num_static):
            if i % 2 == 0:
                static_build = True
            else:
                static_build = False
            self.step(self.action_space.sample(), static_build=True)

    def randomStart(self):
        r = self.np_random.integers(0, 100)
        self.micro.setFunds(self.micro.init_funds)
        for i in range(r):
            self.step(self.action_space.sample())
#       i = np.random.randint(0, (self.obs_width * self.obs_width / 3))
#       a = (np.random.randint(0, self.num_tools, i), np.random.randint(0, self.obs_width, i), np.random.randint(0, self.obs_width, i))
#       for j in range(i):
#           self.micro.takeSetupAction((a[0][j], a[1][j], a[2][j]))

    def powerPuzzle(self):
        ''' Set up one plant, one res. If we restrict the agent to building power lines, we can test its ability
        to make long-range associations. '''
        for i in range(5):
            self.micro.doBotTool(np.random.integers(0, self.micro.MAP_X),
                                 np.random.integers(0, self.micro.MAP_Y), 'Residential', static_build=True)
        while self.micro.map.num_plants == 0:
            self.micro.doBotTool(np.random.integers(0, self.micro.MAP_X),
                                  np.random.integers(0, self.micro.MAP_Y),
                                  'NuclearPowerPlant', static_build=True)

    def reset(self):
        self.display_city_trgs()
        if True:
           #if self.render_gui:
            if False:
                self.micro.clearBotBuilds()
            else:
                self.micro.clearMap()
        if not self.empty_start:
            self.micro.newMap()
        self.num_step = 0
        if self.power_puzzle:
            self.powerPuzzle()
        if self.random_builds:
            self.randomStaticStart()
        self.micro.simTick()
        self.city_metrics = self.get_city_metrics()
        self.last_city_metrics = self.city_metrics
        self.micro.setFunds(self.micro.init_funds)
       #curr_funds = self.micro.getFunds()
        self.curr_pop = 0
        self.curr_reward = self.getReward()
        self.state = self.getState()
        self.last_pop=0
        self.micro.num_roads = 0
        self.last_num_roads = 0
       #self.past_actions.fill(False)
        self.num_episode += 1
        return self.state

  # def getRoadPenalty(self):
  #
  #     class roadPenalty(torch.nn.module):
  #         def __init__(self):
  #             super(roadPenalty, self).__init__()

  #             self.
    def getState(self):
        res_pop, com_pop, ind_pop = self.micro.getResPop(), self.micro.getComPop(), self.micro.getIndPop()
        resDemand, comDemand, indDemand = self.micro.engine.getDemands()
        scalars = [res_pop, com_pop, ind_pop, resDemand, comDemand, indDemand]
        if self.poet:
            for j in range(3):
                scalars[j] = scalars[j] / self.param_ranges[j]
            trg_metrics = [v for k, v in self.city_trgs.items()]
            for i in range(len(trg_metrics)):
                trg_metrics[i] = trg_metrics[i] / self.param_ranges[i]
            scalars += trg_metrics
        return self.observation(scalars)


    def observation(self, scalars):
        state = self.micro.map.getMapState()
        density_maps = self.micro.getDensityMaps()
       #if self.render_gui:
       #    print(density_maps[2])
        road_networks = self.micro.map.road_networks
        if self.render_gui:
           #print(road_networks, self.micro.map.road_net_sizes)
            pass
        scalar_layers = np.zeros((len(scalars), self.MAP_X, self.MAP_Y))
        for si in range(len(scalars)):
            fill_val = scalars[si]
            if not type(fill_val) == str:
                scalar_layers[si].fill(scalars[si])
        state = np.concatenate((state, density_maps, scalar_layers), 0)
        if self.static_builds:
            state = np.concatenate((state, self.micro.map.static_builds), 0)
        return state

    def getPop(self):
        self.resPop, self.comPop, self.indPop = self.micro.getResPop(), \
                                     self.micro.getComPop(), \
                                     self.micro.getIndPop()

        curr_pop = self.resPop + \
                   self.comPop + \
                   self.indPop

        return curr_pop

    def getReward(self):
        '''Calculate reward.
        '''
        if True:
            reward = 0
            for metric, trg in self.city_trgs.items():
                last_val = self.last_city_metrics[metric]
                trg_change = trg - last_val
                val = self.city_metrics[metric]
                change = val - last_val
                if np.sign(change) != np.sign(trg_change):
                    metric_rew = -abs(change)
                elif abs(change) < abs(trg_change):
                    metric_rew = abs(change)
                else:
                    metric_rew = abs(trg_change) - abs(trg_change - change)
                reward += metric_rew * self.weights[metric]
       #if self.render_gui and reward != 0:
       #    print(self.city_metrics)
       #    print(self.city_trgs)
       #    print(reward)
       #    print()

       #if False:
       #    max_reward = self.max_reward
       #    loss = 0
       #    i = 0
       #    for k, v in self.city_trgs.items():
       #        if i == self.num_params:
       #            break
       #        else:
       #            if True:
       #                reward = 0
       #                for metric_name, trg in self.city_trgs.items():

       #            weight = self.weights[k]
       #            loss += abs(v - self.city_metrics[k]) * weight
       #            i += 1

       #    reward = (self.max_loss - loss) * max_reward / self.max_loss
       #    reward = self.getPopReward()
       #self.curr_reward = reward
        return reward


    def getPopReward(self):
        if False:
            pop_reward = self.micro.getTotPop()

        else:
            resPop, comPop, indPop = (1/4) * self.micro.getResPop(), self.micro.getComPop(), self.micro.getIndPop()
            pop_reward = resPop + comPop + indPop
            # population density per 16x16 section of map
            pop_reward = pop_reward / (self.MAP_X*self.MAP_Y / 16**2)
            zone_variety = 0
            if resPop > 0:
                zone_variety += 1
            if comPop > 0:
                zone_variety += 1
            if indPop > 0:
                zone_variety += 1
            zone_bonus = (zone_variety - 1) * 50
            pop_reward += max(0, zone_bonus)
        if False:
            pop_reward = (resPop + 1) * (comPop + 1) * (indPop + 1) - 1
        return 0
        return pop_reward

    def set_param_bounds(self, bounds):
        print('setting visual param bounds (TODO: forreal')
        if self.win1:
            self.win1.agentPanel.setMetricRanges(bounds)


    def set_params(self, trgs):
        for k, v in trgs.items():
            self.city_trgs[k] = v
        self.trg_param_vals = np.array([v for v in self.city_trgs.values()])
        self.display_city_trgs()
       #print('set city trgs of env {} to: {}'.format(self.rank, self.city_trgs))

    def get_city_metrics(self):
        res_pop, com_pop, ind_pop = self.micro.getResPop(), \
                                     self.micro.getComPop(), \
                                     self.micro.getIndPop()
        traffic = self.micro.total_traffic
        mayor_rating = self.getRating()
        num_plants = self.micro.map.num_plants
        city_metrics = {
                'res_pop': res_pop,
                'com_pop': com_pop,
                'ind_pop': ind_pop,
                'traffic': traffic, 'num_plants': num_plants,
                'mayor_rating': mayor_rating
                }
        return city_metrics

    def display_city_metrics(self):
        if self.win1 is not None:
            self.win1.agentPanel.displayMetrics(self.city_metrics)



    def step(self, a, static_build=False):
       #self.micro.engine.setPasses(np.random.randint(1, 101))
        if self.player_step:
           #if self.player_step == a:
           #    static_build=False
           #static_build = True
            if self.static_player_builds:
                static_build=True
            a = self.player_step
            self.player_step = False
       #else:
       #    a = 0
        a = self.intsToActions[a]
        self.micro.takeAction(a, static_build)
        return self.postact()

    def postact(self):
        # never let the agent go broke, for now
        self.micro.setFunds(self.micro.init_funds)
       #print('rank {} tickin'.format(self.rank))
        # TODO: BROKEN!
        self.micro.simTick()
        self.state = self.getState()
       #print(self.state[-2])
        self.curr_pop = self.getPop()
        self.last_city_metrics = self.city_metrics
        self.city_metrics = self.get_city_metrics()
        if self.render_gui:
            self.display_city_metrics()

       #if self.traffic_only:
       #    self.curr_pop = self.getPopReward() / 1
       #   #self.curr_pop = 0
       #else:
       #    self.curr_pop = self.getPop() #** 2
       #   #self.curr_pop = self.getPopReward() #** 2
       #pop_reward = self.curr_pop
       #self.curr_mayor_rating = self.getRating()
       #if not self.simple_reward:
       #   #if self.micro.total_traffic > 0:
       #   #    print(self.micro.total_traffic)
       #    if self.traffic_only:
       #        traffic_reward = self.micro.total_traffic * 10
       #       #traffic_reward = 0
       #    else:
       #       #traffic_reward = self.micro.total_traffic / 100
       #        traffic_reward = self.reward_weights[3] * self.micro.total_traffic
       #    if self.player_step:
       #        print('pop reward: {}\n'
       #        'traffic reward: {}'.format(pop_reward, traffic_reward))
       #        self.player_step = None
       #    if pop_reward > 0 and traffic_reward > 0:
       #       #print(pop_reward, traffic_reward)
       #        pass
       #    reward = pop_reward  + traffic_reward
       #    if reward > 0 and self.micro.map.num_roads > 0 and not self.traffic_only: # to avoid one-road minima in early training
       #        max_net_1 = 0
       #        max_net_2 = 0
       #        for n in  self.micro.map.road_net_sizes.values():
       #            if n > max_net_1:
       #                max_net_1 = n
       #           #    max_net_2 = max_net_1
       #           #elif n > max_net_2:
       #           #    max_net_2 = n
        reward = 0


        reward = self.getReward()
       #reward = reward / (self.max_step)
        self.curr_funds = curr_funds = self.micro.getFunds()
        bankrupt = curr_funds < self.minFunds
        terminal = (bankrupt or self.num_step >= self.max_step) and\
            self.auto_reset
        if self.print_map:
           #if static_build:
           #    print('STATIC BUILD')
            self.printMap()
        if self.render_gui:
           #pass
            self.micro.render()
        infos = {}
        # Get the next player-build ready, if there is one in the queue
        if self.micro.player_builds:
            b = self.micro.player_builds[0]
            a = self.actionsToInts[b]
            infos['player_move'] = int(a)
            self.micro.player_builds = self.micro.player_builds[1:]
            self.player_step = a
        self.num_step += 1
       ## Override Reward
       #reward = self.city_metrics['res_pop'] + self.city_metrics['com_pop']\
       #         + self.city_metrics['ind_pop'] + self.city_metrics['traffic']
        return (self.state, reward, terminal, infos)

    def getRating(self):
        return self.micro.engine.cityYes

    def printMap(self, static_builds=True):
           #if static_builds:
           #    static_map = self.micro.map.static_builds
           #else:
           #    static_map = None
            np.set_printoptions(threshold=np.inf)
            zone_map = self.micro.map.zoneMap[-1]
            zone_map = zone_map.transpose(1,0)
            zone_map = np.array_repr(zone_map).replace(',  ','  ').replace('],\n', ']\n').replace(',\n', ',').replace(', ', ' ').replace('        ',' ').replace('         ','  ')
            print('{} \n population: {}, traffic: {}, episode: {}, step: {}, reward: {} \n'.format(zone_map, self.curr_pop, self.micro.total_traffic, self.num_episode, self.num_step, self.curr_reward#, static_map
                ))
           #print(self.micro.map.centers)


    def get_zone_grid(self, abbreviate=True):
        '''
        Returns the city zone map as a 2D grid with zone names.
        
        Args:
            abbreviate: If True, uses short names (R, C, I, W, D, etc.)
                       If False, uses full names (Residential, Commercial, etc.)
        
        Returns:
            List[List[str]]: 2D grid of zone names
        '''
        zone_map = self.micro.map.zoneMap[-1]  # Shape: (MAP_X, MAP_Y)
        zone_map = zone_map.transpose(1, 0)  # Transpose to get (Y, X) order for display
        
        # Get zone name mapping
        zone_names = self.micro.map.zones
        
        # Abbreviation mapping for compact display
        abbreviations = {
            'Residential': 'R', 'Commercial': 'C', 'Industrial': 'I',
            'Road': 'D', 'Rail': 'L', 'Wire': 'W', 'Forest': 'F',
            'Land': 'L', 'Water': 'W', 'Rubble': 'B', 'CoalPowerPlant': 'P',
            'NuclearPowerPlant': 'N', 'Airport': 'A', 'Seaport': 'S',
            'Stadium': 'T', 'Park': 'K', 'FireDept': 'Fire', 'PoliceDept': 'Pol',
            'Net': 'Net', 'Church': 'Ch', 'Hospital': 'H',
            'RoadWire': 'RW', 'RailWire': 'RW', 'Bridge': 'Br',
            'RoadRail': 'RL', 'WaterWire': 'WW', 'Radar': 'Rd',
            'RailBridge': 'RB'
        }
        
        grid = []
        for y in range(self.MAP_Y):
            row = []
            for x in range(self.MAP_X):
                zone_int = int(zone_map[y, x])
                if zone_int < len(zone_names):
                    zone_name = zone_names[zone_int]
                    if abbreviate and zone_name in abbreviations:
                        row.append(abbreviations[zone_name])
                    else:
                        row.append(zone_name)
                else:
                    row.append('U')  # Unknown
            grid.append(row)
        return grid

    def zone_grid_to_text(self, abbreviate=True):
        '''
        Returns the zone grid as a formatted text string.
        
        Args:
            abbreviate: If True, uses short zone names
        
        Returns:
            str: Formatted text representation of the zone grid
        '''
        grid = self.get_zone_grid(abbreviate=abbreviate)
        
        # Create column headers
        header = '   ' + ' '.join(f'{i:2d}' for i in range(self.MAP_X))
        lines = [header, '   ' + '--' * self.MAP_X]
        
        # Add each row with row number
        for y, row in enumerate(grid):
            row_str = f'{y:2d}|' + ' '.join(f'{s:>2s}' for s in row)
            lines.append(row_str)
        
        # Add legend for abbreviations
        if abbreviate:
            legend = [
                '',
                'Legend:',
                '  R=Residential, C=Commercial, I=Industrial, D=Road,',
                '  L=Rail, W=Wire, F=Forest, P=Coal Plant, N=Nuclear Plant,',
                '  A=Airport, S=Seaport, T=Stadium, K=Park, Fire=Fire Dept,',
                '  Pol=Police Dept, Ch=Church, H=Hospital, U=Unknown'
            ]
            lines.extend(legend)
        
        return '\n'.join(lines)

    def get_zone_counts(self):
        '''
        Returns a dictionary of zone types and their counts.
        
        Returns:
            Dict[str, int]: Zone name to count mapping
        '''
        zone_map = self.micro.map.zoneMap[-1]
        zone_names = self.micro.map.zones
        counts = {}
        
        for zone_int in range(len(zone_names)):
            count = int(np.sum(zone_map == zone_int))
            if count > 0:
                counts[zone_names[zone_int]] = count
        
        return counts

    def get_road_networks(self):
        '''
        Returns information about connected road networks.
        
        Returns:
            Dict[int, int]: Network ID to size mapping
        '''
        return dict(self.micro.map.road_net_sizes)

    def get_road_network_text(self):
        '''
        Returns a text description of road networks.
        
        Returns:
            str: Formatted description of road networks
        '''
        networks = self.get_road_networks()
        if not networks:
            return "No road networks detected."
        
        lines = ["ROAD NETWORKS:"]
        total_road_tiles = 0
        for net_id, size in sorted(networks.items(), key=lambda x: -x[1]):
            lines.append(f"  Network {net_id}: {size} tiles")
            total_road_tiles += size
        lines.append(f"  Total: {total_road_tiles} road tiles in {len(networks)} network(s)")
        
        return "\n".join(lines)

    def get_density_info(self):
        '''
        Returns a text description of traffic and population density.
        
        Returns:
            str: Formatted density information
        '''
        traffic = self.micro.total_traffic
        pop = self.micro.getResPop() + self.micro.getComPop() + self.micro.getIndPop()
        
        lines = [
            "DENSITY INFO:",
            f"  Total Population: {pop:,}",
            f"  Total Traffic: {traffic:,}",
        ]
        
        if traffic > 0 and pop > 0:
            traffic_per_capita = traffic / pop
            lines.append(f"  Traffic per capita: {traffic_per_capita:.2f}")
        
        return "\n".join(lines)

    def get_map_ascii(self, include_stats=True):
        '''
        Returns the city map as an ASCII grid with zone abbreviations.
        This provides tile-level accuracy for LLM spatial awareness.
        
        Args:
            include_stats: If True, adds stats line at top (pop, funds, etc.)
        
        Returns:
            str: Formatted ASCII grid with one character per tile
        '''
        # Tile character mapping
        tile_map = {
            'Residential': 'R',
            'Commercial': 'C',
            'Industrial': 'I',
            'Road': '+',
            'Rail': '=',
            'Wire': 'w',
            'FireDept': 'F',
            'PoliceDept': 'P',
            'CoalPowerPlant': 'E',
            'NuclearPowerPlant': 'N',
            'Stadium': 'S',
            'Airport': 'A',
            'Seaport': 'H',
            'Park': 'k',
            'Water': '~',
            'Forest': 't',
            'Land': '.',
            'Rubble': 'X',
            'Fire': 'f',
            'Hospital': 'h',
            'Church': 'c',
            'Net': 'n',
        }
        
        rows = []
        for y in range(self.MAP_Y):
            row = ''
            for x in range(self.MAP_X):
                tile_int = self.micro.getTile(x, y) & 1023
                zone_name = zoneFromInt(tile_int)
                char = tile_map.get(zone_name, '?')
                row += char
            rows.append(row)
        
        result = '\n'.join(rows)
        
        if include_stats:
            # Add stats line at top
            pop = self.micro.getResPop() + self.micro.getComPop() + self.micro.getIndPop()
            funds = self.micro.getFunds()
            traffic = self.micro.total_traffic
            
            # Count unpowered zones (zones without power)
            power_map = self.micro.getDensityMaps()[0]
            unpowered = int(np.sum(power_map == 0))
            
            stats = f"Stats: Pop={pop:,}, Funds=${funds:,}, Traffic={traffic:,}, Unpowered={unpowered}"
            result = stats + "\n" + result
        
        return result

    def describe_city(self, verbose=False):
        '''
        Returns a plain text description of the current city state.
        
        Args:
            verbose: If True, includes more detailed information including
                    zone counts and density statistics
        
        Returns:
            str: A human-readable description of the city state
        '''
        # Get basic metrics
        res_pop = self.micro.getResPop()
        com_pop = self.micro.getComPop()
        ind_pop = self.micro.getIndPop()
        total_pop = res_pop + com_pop + ind_pop
        traffic = self.micro.total_traffic
        funds = self.micro.getFunds()
        mayor_rating = self.getRating()
        num_plants = self.micro.map.num_plants
        
        # Get zone counts
        zone_map = self.micro.map.zoneMap[-1]  # Shape: (MAP_X, MAP_Y)
        zone_counts = {}
        for zone_name, zone_int in self.micro.map.zoneInts.items():
            count = np.sum(zone_map == zone_int)
            if count > 0:
                zone_counts[zone_name] = int(count)
        
        # Build description
        lines = []
        lines.append("=" * 60)
        lines.append("CITY STATE DESCRIPTION")
        lines.append("=" * 60)
        lines.append(f"")
        lines.append("POPULATION:")
        lines.append(f"  Total: {total_pop:,}")
        lines.append(f"  Residential: {res_pop:,}")
        lines.append(f"  Commercial: {com_pop:,}")
        lines.append(f"  Industrial: {ind_pop:,}")
        lines.append(f"")
        lines.append("INFRASTRUCTURE:")
        lines.append(f"  Traffic: {traffic:,}")
        lines.append(f"  Power Plants: {num_plants}")
        lines.append(f"  Funds: ${funds:,}")
        lines.append(f"  Mayor Rating: {mayor_rating}/100")
        lines.append(f"")
        lines.append("ZONE DISTRIBUTION:")
        for zone_name, count in sorted(zone_counts.items(), key=lambda x: -x[1]):
            percentage = (count / (self.MAP_X * self.MAP_Y)) * 100
            lines.append(f"  {zone_name}: {count} tiles ({percentage:.1f}%)")
        
        if verbose:
            lines.append(f"")
            lines.append("METRICS:")
            lines.append(f"  Step: {self.num_step}")
            lines.append(f"  Episode: {self.num_episode}")
            lines.append(f"  Current Reward: {self.curr_reward:.2f}")
            lines.append(f"  City Time: {self.micro.engine.cityTime} ticks")
            lines.append(f"  City Month: {self.micro.engine.cityMonth}")
            lines.append(f"  Year: {(self.micro.engine.cityTime // 48) + 1900}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


    def render(self, mode='human'):
        self.micro.render()

    def test(self):
        env = MicropolisEnv()
        for i in range(5000):
            env.step(env.action_space.sample())

    def set_res_weight(self, val):
        self.city_trgs['res_pop']= val

    def set_com_weight(self, val):
        self.city_trgs['com_pop'] = val

    def set_ind_weight(self, val):
        self.city_trgs['ind_pop'] = val

    def set_traffic_weight(self, val):
        self.city_trgs['traffic'] = val

    def set_plants_weight(self, val):
        self.city_trgs['num_plants'] = val

    def set_rating_weight(self,val):
        self.city_trgs['mayor_rating'] = val
