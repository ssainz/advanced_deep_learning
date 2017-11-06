import numpy as np
import math


class State:

    def __init__(self, size_x, size_y, default_reward):

        self.size_x = size_x
        self.size_y = size_y

        self.m_state_v = np.zeros((size_x, size_y))
        self.m_state_rewards = np.zeros((size_x, size_y))
        self.m_probability = np.zeros((size_x, size_y))
        self.m_policy = np.zeros((size_x, size_y, 2))

        # To encourage the bot to go to rewards in the least amount of time, set rewards to -1:
        self.m_state_rewards[:] = default_reward

        # For the rewards we say that (1,4) is +10
        self.m_state_rewards[0,3] = 10.0

        # For the rewards we say that (2,4) is -10
        self.m_state_rewards[1, 3] = -10.0

        # Probability, we saw is always 1:
        self.m_probability[:] = 1.0


        # For the rewards we say that (3,2) loops, it returns no value and if any neighbor choose to come here, the result is returned to previous state.
        self.m_state_rewards[2, 1] = None
        self.m_state_v[2,1] = None
        self.m_probability[2, 1] = None

    def get_potential_actions(self, index_x, index_y):

        potential_states = []
        action_name = []
        if index_x < self.size_x - 1:
            action_name.append("DOWN ")
            potential_states.append(( (index_x + 1), index_y ) )

        if index_y < self.size_y - 1:
            action_name.append("RIGHT")
            potential_states.append( ( index_x, (index_y + 1)) )

        if index_x > 0:
            action_name.append("UP   ")
            potential_states.append( ((index_x - 1), index_y) )

        if index_y > 0:
            action_name.append("LEFT ")
            potential_states.append(  ( index_x, (index_y - 1)) )

        # We remove the invalid cell (2, 1):
        i = 0
        potential_states_cleaned = []
        action_name_cleaned = []
        for (t_x, t_y) in potential_states:
            if not ( t_x == 2 and t_y == 1 ):
                potential_states_cleaned.append((t_x, t_y))
                action_name_cleaned.append(action_name[i])
            i += 1

        return potential_states_cleaned, action_name_cleaned


    def get_new_v_values_table(self):
        return np.copy(self.m_state_v)


    def set_v_values_table(self, new_state):
        self.m_state_v = new_state


    def get_v_value(self, index_x, index_y):
        if np.isnan(self.m_state_v[index_x, index_y]):
            return 0
        else:
            return self.m_state_v[index_x, index_y]

    def get_reward_value(self, index_x, index_y):
        if np.isnan(self.m_state_rewards[index_x, index_y]):
            return 0
        else:
            return self.m_state_rewards[index_x, index_y]

    def get_probability_of_succesful_jump(self, index_x, index_y):
        if np.isnan(self.m_probability[index_x, index_y]):
            return 0
        else:
            return self.m_probability[index_x, index_y]

    def set_v_value(self,  index_x, index_y, val, temp_state = None):
        if temp_state is None :
            temp_state = self.m_state_v

        temp_state[index_x, index_y] = val

    def get_policy(self, from_x, from_y):

        return self.m_policy[from_x, from_y, :]

    def set_policy(self, from_x, from_y, to_x, to_y):

        self.m_policy[from_x, from_y, 0] = to_x
        self.m_policy[from_x, from_y, 1] = to_y

    def sum_all_values(self):

        # Remove None away:
        a = np.copy(self.m_state_v)
        where_are_NaNs = np.isnan(a)
        a[where_are_NaNs] = 0

        return np.sum(a)

    def print_policy(self):
        result = ""
        i = 0
        for row in self.m_policy:
            j = 0
            for col in row:
                result = result + " | (%s,%s)->(%s,%s) | " % (i,j, int(col[0]), int(col[1]))
                j += 1

            result = result + "\n"
            i += 1

        return result


size_x = 3
size_y = 4

# Create state manager class that keeps track of values, rewards and finding next actions.
state_mgr = State(size_x, size_y, default_reward=0.0)

# Initialize v value of all states in ZERO.
# Actually the State class already init states in zero, but just for completeness
for x in range(size_x):
    for y in range(size_y):
        state_mgr.set_v_value(x,y, 0)

# Also initialize first policy (pick the first action (priority: DOWN, RIGHT, UP, LEFT)):
# We CANNOT just go up, because some states cannot go up...
for x in range(size_x):
    for y in range(size_y):
        state_list, action_name = state_mgr.get_potential_actions(x,y )
        state_mgr.set_policy(x,y, state_list[0][0], state_list[0][1])



# We go over the two states, value iteration and policy improvement:
first_time = True
delta_between_sum_of_all_values_policy = 0.0
gamma = 0.9
q = 0
policy_iteration_sum = []
probability_of_deterministic_jump = 1.0
convergence_criteria = 0.01
while first_time == True or (delta_between_sum_of_all_values_policy > convergence_criteria):
    first_time = False

    # Get current value sum
    sum_before_change_policy = state_mgr.sum_all_values()

    # ---------------------------
    # VALUE ITERATION: BEGIN
    # ---------------------------
    delta_between_sum_of_all_values_val_iteration = 0.0
    first_time_value_iteration = True
    while first_time_value_iteration == True or (delta_between_sum_of_all_values_val_iteration > convergence_criteria):
        first_time_value_iteration = False
        # Get current value sum
        sum_before_change_value_iteration = state_mgr.sum_all_values()

        # Get new V_t new values, meanwhile old state, V_(t-1) is still set in the state manager.
        V_t = state_mgr.get_new_v_values_table()

        # For each state:
        for _x in range(size_x):
            for _y in range(size_y):

                # We do not process invalid state (2, 1), or end states (0,3) or (1,3)
                if not (_x == 2 and _y == 1) and not (_x == 0 and _y == 3) and not (_x == 1 and _y == 3):
                    state_list, actions_name = state_mgr.get_potential_actions(_x, _y)

                    # Non deterministic, we will check all posible states for each action:

                    # Find the likely state:
                    policy_a = state_mgr.get_policy(_x, _y)
                    _x_likely = policy_a[0]
                    _y_likely = policy_a[1]

                    # Calculate the probability for the other states (including current node!)
                    random_states = list(state_list)
                    random_states.append((_x, _y))

                    # Find probability of random jump.
                    probability_of_random_jump = 1.0 - probability_of_deterministic_jump
                    number_of_non_likely_states = len(random_states) - 1  # minus one because of the likely state
                    probability_of_random_jump_to_one_non_likely_state = probability_of_random_jump / number_of_non_likely_states

                    # Now we iterate over all the random states for this action (which includes itself):
                    sum_new_v_value = 0
                    for (_x_random, _y_random) in random_states:
                        if _x_random == _x_likely and _y_random == _y_likely:
                            pr = probability_of_deterministic_jump
                        else:
                            pr = probability_of_random_jump_to_one_non_likely_state

                        # Get its V values:
                        v_value = state_mgr.get_v_value(_x_random, _y_random)

                        # Also get its rewards
                        reward = state_mgr.get_reward_value(_x_random, _y_random)

                        # Get the new v_value for this probability
                        new_v_value = pr * (reward + (gamma * v_value))
                        sum_new_v_value += new_v_value


                    # Now we set the new V value and Pi policy
                    state_mgr.set_v_value(_x, _y, sum_new_v_value, temp_state=V_t)

        # after all updates, set the new state into the state manager: V_t becomes V_(t-1):
        state_mgr.set_v_values_table(V_t)

        # Get value sum after update.
        sum_after_change_value_iteration = state_mgr.sum_all_values()
        delta_between_sum_of_all_values_val_iteration = math.fabs(sum_before_change_value_iteration - sum_after_change_value_iteration)

    # ---------------------------
    # VALUE ITERATION: END
    # ---------------------------

    # ---------------------------
    # POLICY UPGRADE: BEGIN
    # ---------------------------

    # For each state:
    for _x in range(size_x):
        for _y in range(size_y):

            # We do not process invalid state (2, 1), or end states (0,3) or (1,3)
            if not (_x == 2 and _y == 1) and not (_x == 0 and _y == 3) and not (_x == 1 and _y == 3):
                state_list, actions_name = state_mgr.get_potential_actions(_x, _y)
                max_a_value = 0
                policy_a = state_mgr.get_policy(_x, _y)
                z = 0
                for action_name in actions_name:
                    # For this action/state, we find expected value of all the potential actions.

                    # Non deterministic, we will check all possible states for each action:

                    # Find the likely state:
                    (_x_likely, _y_likely) = state_list[z]

                    # Calculate the probability for the other states (including current node!)
                    random_states = list(state_list)
                    random_states.append((_x, _y))

                    # Find probability of random jump.
                    probability_of_random_jump = 1.0 - probability_of_deterministic_jump
                    number_of_non_likely_states = len(random_states) - 1  # minus one because of the likely state
                    probability_of_random_jump_to_one_non_likely_state = probability_of_random_jump / number_of_non_likely_states

                    # Now we iterate over all the random states for this action (which includes itself):
                    sum_new_v_value = 0
                    for (_x_random, _y_random) in random_states:
                        if _x_random == _x_likely and _y_random == _y_likely:
                            pr = probability_of_deterministic_jump
                        else:
                            pr = probability_of_random_jump_to_one_non_likely_state

                        # Get its V values:
                        v_value = state_mgr.get_v_value(_x_random, _y_random)

                        # Also get its rewards
                        reward = state_mgr.get_reward_value(_x_random, _y_random)

                        # Get the new v_value for this probability
                        new_v_value = pr * (reward + (gamma * v_value))
                        sum_new_v_value += new_v_value

                    # Check if it is the largest value
                    if sum_new_v_value > max_a_value:
                        max_a_value = sum_new_v_value
                        policy_a = [_x_likely, _y_likely]

                    z += 1

                # Now we set the new Pi policy
                state_mgr.set_policy(_x, _y, policy_a[0], policy_a[1])

    sum_after_change_policy = state_mgr.sum_all_values()
    delta_between_sum_of_all_values_policy = math.fabs(sum_before_change_policy - sum_after_change_policy)

    print("Policy iteration: it(%s): sum of V-values: (%s)" %(q, sum_after_change_policy))
    policy_iteration_sum.append(sum_after_change_policy)
    q += 1
    # ---------------------------
    # POLICY UPGRADE: END
    # ---------------------------

# plt.subplot(1, 1, 1)
# policy_iteration_convergence, = plt.plot(np.array(policy_iteration_sum), label='Policy iteration (sum of v-values across iterations')
# plt.legend(handles=[policy_iteration_convergence])
# plt.draw()
# plt.show()

print("[POLICY ITERATION] V values:")
print(state_mgr.m_state_v)
print("[POLICY ITERATION] Policy:")
print(state_mgr.print_policy())