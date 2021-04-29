#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from __future__ import division

import math
import rospy
from geometry_msgs.msg import Twist
from util import Vector2, angle_diff, MAFilter


def get_agent_velocity(agent):
    """Return agent velocity as Vector2 instance."""
    vel = Vector2()
    vel.x = agent.twist.twist.linear.x
    vel.y = agent.twist.twist.linear.y
    return vel


def get_agent_position(agent):
    """Return agent position as Vector2 instance."""
    pos = Vector2()
    pos.x = agent.pose.pose.position.x
    pos.y = agent.pose.pose.position.y
    return pos


def get_obst_position(obst):
    """Return obstacle position as Vector2 instance."""
    pos = Vector2()
    pos.x = obst.position.x
    pos.y = obst.position.y
    return pos


class Boid(object):
    """
    An implementation of Craig Reynolds' flocking rules and boid objects.

    Each boid (bird-oid object) maneuvers based on the positions and velocities
    of its nearby flockmates. Computation is based on three components:
    1) alignment: steer towards the average heading of local flockmates
    2) cohesion: steer to move toward the average position of local flockmates
    3) separation: steer to avoid crowding local flockmates.
    Additionally, 4th component, avoid, is implemented where boids steer away
    from obstacles in their search radius.

    Each component yields a force on boid. Total force then gives the
    acceleration which is integrated to boid's velocity. Force and velocity are
    limited to the specified amount.

    State:
        position (Vector2): Boid's position
        velocity (Vector2): Boid's velocity

    Parameters:
        mass (double): Boid's mass
        alignment_factor (double): Weight for alignment component
        cohesion_factor (double): Weight for cohesion component
        separation_factor (double): Weight for separation component
        avoid_factor (double): Weight for obstacle avoiding component
        max_speed (double): Velocity upper limit
        max_force (double): Force upper limit
        friction (double): Constant friction force
        crowd_radius (double): Radius to avoid crowding
        search_radius (double): Boid's sensing radius
        avoid_radius (double): Radius to avoid obstacles

    Methods:
        update_parameters(self): Save parameters in class variables
        compute_alignment(self, nearest_agents): Return alignment component
        compute_cohesion(self, nearest_agents): Return cohesion component
        compute_separation(self, nearest_agents): Return separation component
        compute_avoids(self, avoids): Return avoid component
        compute_velocity(self, my_agent, nearest_agents, avoids):
            Compute total velocity based on all components
    """

    def __init__(self, initial_velocity_x, initial_velocity_y, wait_count, start_count, frequency):
        """Create an empty boid and update parameters."""
        self.position = Vector2()
        self.velocity = Vector2()
        self.mass = 0.18                # Mass of Sphero robot in kilograms
        self.wait_count = wait_count    # Waiting time before starting
        self.start_count = start_count  # Time during which initial velocity is being sent
        self.frequency = frequency      # Control loop frequency

        # Set initial velocity
        self.initial_velocity = Twist()
        self.initial_velocity.linear.x = initial_velocity_x
        self.initial_velocity.linear.y = initial_velocity_y

        # UNCOMMENT IF USING FILTER
        # # Initialize moving average filters.
        # self.x_filter = MAFilter(3)
        # self.y_filter = MAFilter(3)

        # This dictionary holds values of each flocking components and is used
        # to pass them to the visualization markers publisher.
        self.viz_components = {}

    def update_parameters(self, params):
        """Save Reynolds controller parameters in class variables."""
        self.alignment_factor = params['alignment_factor']
        self.cohesion_factor = params['cohesion_factor']
        self.separation_factor = params['separation_factor']
        self.avoid_factor = params['avoid_factor']
        self.max_speed = params['max_speed']
        self.max_force = params['max_force']
        self.friction = params['friction']
        self.crowd_radius = params['crowd_radius']
        self.search_radius = params['search_radius']
        self.avoid_radius = params['avoid_radius']

        # Scaling is calculated so that force is maximal when agent is
        # 0.85 * search_radius away from obstacle.
        self.avoid_scaling = 1 / ((0.85 * self.search_radius) ** 2 * self.max_force)

        # Scaling is calculated so that cohesion and separation forces
        # are equal when agents are crowd_radius apart.
        self.separation_scaling = self.search_radius / self.crowd_radius ** 3 / self.max_force

        rospy.loginfo(rospy.get_caller_id() + " -> Parameters updated")
        rospy.logdebug('alignment_factor:  %s', self.alignment_factor)
        rospy.logdebug('cohesion_factor:  %s', self.cohesion_factor)
        rospy.logdebug('separation_factor:  %s', self.separation_factor)
        rospy.logdebug('avoid_factor:  %s', self.avoid_factor)
        rospy.logdebug('max_speed:  %s', self.max_speed)
        rospy.logdebug('max_force:  %s', self.max_force)
        rospy.logdebug('friction:  %s', self.friction)
        rospy.logdebug('crowd_radius:  %s', self.crowd_radius)
        rospy.logdebug('search_radius:  %s', self.search_radius)
        rospy.logdebug('avoid_radius:  %s', self.avoid_radius)

    def compute_alignment(self, nearest_agents):
        """Return alignment component."""
        mean_velocity = Vector2()
        steer = Vector2()
        # Find mean direction of neighboring agents.
        for agent in nearest_agents:
            agent_velocity = get_agent_velocity(agent)
            mean_velocity += agent_velocity
        rospy.logdebug("alignment*:   %s", mean_velocity)

        # Steer toward calculated mean direction with maximum velocity.
        if nearest_agents:
            mean_velocity.set_mag(self.max_speed)
            steer = mean_velocity - self.velocity
            steer.limit(self.max_force)
        return steer

    def compute_cohesion(self, nearest_agents):
        """Return cohesion component."""
        mean_position = Vector2()
        direction = Vector2()
        # Find mean position of neighboring agents.
        for agent in nearest_agents:
            agent_position = get_agent_position(agent)
            mean_position += agent_position

        # Apply force in the direction of calculated mean position.
        # Force is proportional to agents' distance from the mean.
        if nearest_agents:
            direction = mean_position / len(nearest_agents)
            rospy.logdebug("cohesion*:    %s", direction)
            d = direction.norm()
            direction.set_mag((self.max_force * (d / self.search_radius)))
        return direction

    def compute_separation(self, nearest_agents):
        """Return separation component."""
        direction = Vector2()
        count = 0

        # Calculate repulsive force for each neighboring agent in sight.
        for agent in nearest_agents:
            agent_position = get_agent_position(agent)
            d = agent_position.norm()
            if d < self.crowd_radius:
                count += 1
                agent_position *= -1        # Make vector point away from other agent.
                agent_position.normalize()  # Normalize to get only direction.
                # Vector's magnitude is proportional to inverse square of the distance between agents.
                agent_position = agent_position / (self.separation_scaling * d**2)
                direction += agent_position

        if count:
            # Devide by number of close-by agents to get average force.
            direction /= count
            direction.limit(2 * self.max_force)  # 2 * max_force gives this rule a slight priority.
        rospy.logdebug("separation*:  %s", direction)
        return direction

    def compute_avoids(self, avoids):
        """
        Return avoid component.

        This rule consists of two components. The first is active for all
        obstacles within range and depends on agent's distance from obstacle as
        well as its aproach angle. Force is maximum when agent approaches the
        obstacle head-on and minimum if obstacle is to the side of an agent.
        Second component depends only on distance and only obstacles closer than
        avoid_radius are taken into account. This is to ensure minimal distance
        from obstacles at all times.
        """
        main_direction = Vector2()
        safety_direction = Vector2()
        count = 0

        # Calculate repulsive force for each obstacle in sight.
        for obst in avoids:
            obst_position = get_obst_position(obst)
            d = obst_position.norm()
            obst_position *= -1        # Make vector point away from obstacle.
            obst_position.normalize()  # Normalize to get only direction.
            # Additionally, if obstacle is very close...
            if d < self.avoid_radius:
                # Scale lineary so that there is no force when agent is on the
                # edge of minimum avoiding distance and force is maximum if the
                # distance from the obstacle is zero.
                safety_scaling = -2 * self.max_force / self.avoid_radius * d + 2 * self.max_force
                safety_direction += obst_position * safety_scaling
                count += 1

            # For all other obstacles: scale with inverse square law.
            obst_position = obst_position / (self.avoid_scaling * d**2)
            main_direction += obst_position

        if avoids:
            # Calculate the approach vector.
            a = angle_diff(self.old_heading, main_direction.arg() + 180)
            # We mustn't allow scaling to be negative.
            side_scaling = max(math.cos(math.radians(a)), 0)
            # Divicde by number of obstacles to get average.
            main_direction = main_direction / len(avoids) * side_scaling
            safety_direction /= count

        rospy.logdebug("avoids*:      %s", main_direction)
        # Final force is sum of two componets.
        # Force is not limited so this rule has highest priority.
        return main_direction + safety_direction

    def compute_velocity(self, my_agent, nearest_agents, avoids):
        """Compute total velocity based on all components."""

        # While waiting to start, send zero velocity and decrease counter.
        if self.wait_count > 0:
            self.wait_count -= 1
            rospy.logdebug("wait " + '{}'.format(self.wait_count))
            rospy.logdebug("velocity:\n%s", Twist().linear)
            return Twist(), None

        # Send initial velocity and decrease counter.
        elif self.start_count > 0:
            self.start_count -= 1
            rospy.logdebug("start " + '{}'.format(self.start_count))
            rospy.logdebug("velocity:\n%s", self.initial_velocity.linear)
            return self.initial_velocity, None

        # Normal operation, velocity is determined using Reynolds' rules.
        else:
            self.velocity = get_agent_velocity(my_agent)
            self.old_heading = self.velocity.arg()
            self.old_velocity = Vector2(self.velocity.x, self.velocity.y)
            rospy.logdebug("old_velocity: %s", self.velocity)

            # Compute all the components.
            alignment = self.compute_alignment(nearest_agents)
            cohesion = self.compute_cohesion(nearest_agents)
            separation = self.compute_separation(nearest_agents)
            avoid = self.compute_avoids(avoids)

            rospy.logdebug("alignment:    %s", alignment)
            rospy.logdebug("cohesion:     %s", cohesion)
            rospy.logdebug("separation:   %s", separation)
            rospy.logdebug("avoid:        %s", avoid)

            # Add components together and limit the output.
            force = Vector2()
            force += alignment * self.alignment_factor
            force += cohesion * self.cohesion_factor
            force += separation * self.separation_factor
            force += avoid * self.avoid_factor
            force.limit(self.max_force)

            # If agent is moving, apply constant friction force.
            # If agent's velocity is less then friction / 2, it would get
            # negative velocity. In this case, just stop it.
            if self.velocity.norm() > self.friction / 2:
                force += self.friction * -1 * self.velocity.normalize(ret=True)
            else:
                self.velocity = Vector2()

            acceleration = force / self.mass

            # Calculate total velocity (delta_velocity = acceleration * delta_time).
            self.velocity += acceleration / self.frequency
            
            # OVO JE NOVO DODANO
            #self.velocity.limit(self.max_speed)
            #self.velocity = Vector2(x = 0.0, y = -1.0)
            self.velocity.normalize()
            self.velocity = self.velocity * self.max_speed

            rospy.logdebug("force:        %s", force)
            rospy.logdebug("acceleration: %s", acceleration / self.frequency)
            rospy.logdebug("velocity:     %s\n", self.velocity)

            # UNCOMMENT IF USING FILTER
            # # Apply moving average filter on calculated velocity
            # filtered = Twist()
            # filtered.linear.x = self.x_filter.step(self.velocity.x)
            # filtered.linear.y = self.y_filter.step(self.velocity.y)

            # # Store raw and filtered velocity in a list for later analysis
            # self.data_list.append([self.velocity.norm(), self.velocity.arg(),
            #                        filtered.norm(), filtered.arg()])

            # Return the the velocity as Twist message.
            vel = Twist()
            vel.linear.x = self.velocity.x
            vel.linear.y = self.velocity.y

            # Pack all components for Rviz visualization.
            # Make sure these keys are the same as the ones in `util.py`.
            self.viz_components['alignment'] = alignment * self.alignment_factor
            self.viz_components['cohesion'] = cohesion * self.cohesion_factor
            self.viz_components['separation'] = separation * self.separation_factor
            self.viz_components['avoid'] = avoid * self.avoid_factor
            self.viz_components['acceleration'] = acceleration / self.frequency
            self.viz_components['velocity'] = self.velocity
            self.viz_components['estimated'] = self.old_velocity
            return vel, self.viz_components
