# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An Agent Factory."""

import datetime
from concordia import components as generic_components
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.clocks import game_clock
from concordia.components import agent as agent_components
from concordia.factory.agent import basic_agent__main_role
from concordia.language_model import language_model
from concordia.typing import component

# TODO: Change characteristic here
extra_instructions = 'The agent always acts on its own self-interest, never caring about the goals or feelings of others.'
additional_characteristic = 'Selfish Individual'

def get_characteristic_component(
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    agent_name: str,
    characteristic_name: str,
    state_clock_now: None = None,
    extra_instructions: str = '',
    num_memories_to_retrieve: int = 25,
    verbose: bool = False
) -> component.Component:
  """Component that can override the agent's traits."""
  return agent_components.characteristic.Characteristic(
    model=model,
    memory=memory,
    agent_name=agent_name,
    characteristic_name=characteristic_name,
    state_clock_now=state_clock_now,
    extra_instructions=extra_instructions,
    num_memories_to_retrieve=num_memories_to_retrieve,
    verbose=verbose
  )


def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,
    memory: associative_memory.AssociativeMemory,
    clock: game_clock.MultiIntervalClock,
    update_time_interval: datetime.timedelta,
) -> basic_agent.BasicAgent:
  """Build an agent.

  Args:
    config: The agent config to use.
    model: The language model to use.
    memory: The agent's memory object.
    clock: The clock to use.
    update_time_interval: Agent calls update every time this interval passes.

  Returns:
    An agent.
  """
  if not config.extras.get('main_character', False):
    raise ValueError('This function is meant for a main character '
                     'but it was called on a supporting character.')

  agent_name = config.name

  instructions = basic_agent__main_role.get_instructions(agent_name)

  time = generic_components.report_function.ReportFunction(
      name='Current time',
      function=clock.current_time_interval_str,
  )

  overarching_goal = generic_components.constant.ConstantComponent(
      state=config.goal, name='overarching goal')

  current_obs = agent_components.observation.Observation(
      agent_name=agent_name,
      clock_now=clock.now,
      memory=memory,
      timeframe=clock.get_step_size(),
      component_name='current observations',
  )
  summary_obs = agent_components.observation.ObservationSummary(
      agent_name=agent_name,
      model=model,
      clock_now=clock.now,
      memory=memory,
      components=[current_obs],
      timeframe_delta_from=datetime.timedelta(hours=4),
      timeframe_delta_until=datetime.timedelta(hours=1),
      component_name='summary of observations',
  )

  relevant_memories = agent_components.all_similar_memories.AllSimilarMemories(
      name='relevant memories',
      model=model,
      memory=memory,
      agent_name=agent_name,
      components=[summary_obs],
      clock_now=clock.now,
      num_memories_to_retrieve=10,
  )

  options_perception = (
      agent_components.options_perception.AvailableOptionsPerception(
          name=(f'\nQuestion: Which options are available to {agent_name} '
                'right now?\nAnswer'),
          model=model,
          memory=memory,
          agent_name=agent_name,
          components=[overarching_goal,
                      current_obs,
                      summary_obs,
                      relevant_memories],
          clock_now=clock.now,
      )
  )
  best_option_perception = (
      agent_components.options_perception.BestOptionPerception(
          name=(f'\nQuestion: Of the options available to {agent_name}, and '
                'given their goal, which choice of action or strategy is '
                f'best for {agent_name} to take right now?\nAnswer'),
          model=model,
          memory=memory,
          agent_name=agent_name,
          components=[overarching_goal,
                      current_obs,
                      summary_obs,
                      relevant_memories,
                      options_perception],
          clock_now=clock.now,
      )
  )

  characteristic = get_characteristic_component(
    model=model,
    memory=memory,
    agent_name=agent_name,
    characteristic_name=additional_characteristic,
    extra_instructions=extra_instructions,
  )

  information = generic_components.sequential.Sequential(
      name='information',
      components=[
          time,
          current_obs,
          summary_obs,
          relevant_memories,
          options_perception,
          best_option_perception,
          characteristic,
      ]
  )

  agent = basic_agent.BasicAgent(
      model=model,
      agent_name=agent_name,
      clock=clock,
      verbose=False,
      components=[instructions,
                  overarching_goal,
                  information],
      update_interval=update_time_interval
  )

  return agent
