import numpy as np


def q_learning(alpha, gamma, number_of_iterations, file_path):

    # READING THE FILE

    # action and state dictionaries, they will help us translate between state name and state index
    # Also these dictionaries help to find right shape of Q-values table.
    actions = {}
    action_count = 0
    states = {}
    state_count = 0

    transitions = []
    with open(file_path, "rt") as f:
        for line in f:
            tokens = line.split()
            s_t = tokens[0]

            a_t = tokens[1]
            s_next = tokens[2]
            r_t = tokens[3]
            transitions.append([s_t, a_t, s_next, r_t])
            # print("s_t: %s, a_t:%s, s_next:%s, r_t:%s " % (s_t, a_t, s_next, r_t))

            if not s_t in states:
                states[s_t] = state_count
                state_count += 1
            if not a_t in actions:
                actions[a_t] = action_count
                action_count += 1
            if not s_next in states:
                states[s_next] = state_count
                state_count += 1

    transitions = np.stack(transitions, axis=0)

    # Generate Q-Learning table
    count_of_distinct_states = len(states.keys())
    count_of_distinct_actions = len(actions.keys())
    q = np.zeros((count_of_distinct_states,count_of_distinct_actions), dtype=np.float64)

    for i in range(number_of_iterations):


        # Generate new sampling, no replacement :
        indexes = np.array(range(transitions.shape[0]))
        np.random.shuffle(indexes)

        # Go over the random samples
        for ind in indexes:
            if (ind + len(indexes) * i) != 0:
                alpha = 1 / (ind + len(indexes) * i)
            # Initialize the initial state, action state or state
            s_t = transitions[ind,0]
            a_t = transitions[ind,1]
            s_next = transitions[ind,2]
            r_t = transitions[ind,3]
            # We fetch all the Q-values for s_next.
            s_next_q_values = q[states[s_next],:]
            max_s_next_q_value = None
            for s_next_q_value in s_next_q_values:
                if max_s_next_q_value is None or s_next_q_value > max_s_next_q_value:
                    max_s_next_q_value = s_next_q_value

            # we calculate the new sample:
            sample = float(r_t) + gamma * max_s_next_q_value
            # print(r_t)
            # print(float(r_t))
            # print(sample)
            # print("-----")

            # We apply exponential moving average to the sample
            # Here we need to do q[state[s_t], actions[a_t] ] because the data has non-zero based integers as
            # actions and state identifiers. Otherwise, if states and actions id were zero based, we
            # would just use q[s_t, a_t]
            q[states[s_t], actions[a_t]] = ((1 - alpha) * q[states[s_t],actions[a_t]]) + (alpha * sample)

            print(alpha)



    # we print Q- table:

    # we must invert the state-index and action-index dictionaries for us to print from the table zero-based index
    index_to_state = {}
    for state in states.keys():
        index_to_state[states[state]] = state

    index_to_action = {}
    for action in actions.keys():
        index_to_action[actions[action]] = action

    buffer = ""
    for state_index in range(q.shape[0]):
        for action_index in range(q.shape[1]):
            buffer += "Q(%s, %s)=%s\t" % (index_to_state[state_index], index_to_action[action_index], q[state_index, action_index])

        buffer += "\n"


    print(buffer)


alpha = 1.0
gamma = 0.9
number_of_iterations = 1000
file_path = "/srv/datasets/qlearning/q-learning.dat"

q_learning(alpha, gamma, number_of_iterations, file_path)