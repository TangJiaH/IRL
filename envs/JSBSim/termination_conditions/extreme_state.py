from .termination_condition_base import BaseTerminationCondition
from ..core.catalog import Catalog as c, JsbsimCatalog, ExtraCatalog
from numpy.linalg import norm

class ExtremeState(BaseTerminationCondition):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        End up the simulation if the aircraft is on an extreme state.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        done = bool(env.agents[agent_id].get_property_value(c.detect_extreme_state))
        if done:
            env.agents[agent_id].crash()
            self.log(f'{agent_id} is on an extreme state! Total Steps={env.current_step}')
            print(f'{agent_id} is on an extreme state! Total Steps={env.current_step}')
            self.get_info(env.agents[agent_id], agent_id)
        success = False
        return done, success, info

    def get_info(self, sim, agent_id):
        extreme_velocity = sim.get_property_value(JsbsimCatalog.velocities_eci_velocity_mag_fps) >= 1e10
        extreme_rotation = (
                norm(
                    sim.get_property_values(
                        [
                            JsbsimCatalog.velocities_p_rad_sec,
                            JsbsimCatalog.velocities_q_rad_sec,
                            JsbsimCatalog.velocities_r_rad_sec,
                        ]
                    )
                ) >= 1000
        )
        extreme_altitude = sim.get_property_value(JsbsimCatalog.position_h_sl_ft) >= 1e10
        extreme_acceleration = (
                max(
                    [
                        abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_x_norm)),
                        abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_y_norm)),
                        abs(sim.get_property_value(JsbsimCatalog.accelerations_n_pilot_z_norm)),
                    ]
                ) > 1e1
        )  # acceleration larger than 10G
        sim.set_property_value(
            ExtraCatalog.detect_extreme_state,
            extreme_altitude or extreme_rotation or extreme_velocity or extreme_acceleration,
        )
        if extreme_velocity:
            print(f'{agent_id} is on an extreme_velocity!')
        if extreme_rotation:
            print(f'{agent_id} is on an extreme_rotation!')
        if extreme_altitude:
            print(f'{agent_id} is on an extreme_altitude!')
        if extreme_acceleration:
            print(f'{agent_id} is on an extreme_acceleration!')
