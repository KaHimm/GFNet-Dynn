from gfnet_dynn import TrainingPhase



def switch_training_phase(current_phase):
    if current_phase == "gate":
        return "classifier"
    elif current_phase == "classifier":
        return "gate"
    elif current_phase == "warm_up":
        return "gate"


        