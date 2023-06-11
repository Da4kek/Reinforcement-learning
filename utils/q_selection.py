def computeQ(old_q_value,reward,next_optimal_q,learning_rate,gamma):
    """_summary_

    Args:
        old_q_value (_type_): _description_
        reward (_type_): _description_
        next_optimal_q (_type_): _description_
        learning_rate (_type_): _description_
        gamma (_type_): _description_
    """
    return old_q_value + learning_rate * (reward + gamma * next_optimal_q - old_q_value)