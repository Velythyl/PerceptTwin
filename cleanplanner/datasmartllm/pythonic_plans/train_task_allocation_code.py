# EXAMPLE 1 - Task Description: Put tomato in fridge
# GENERAL TASK DECOMPOSITION
# Independent subtasks:
# SubTask 1: Put Tomato in Fridge. (Predicted skills required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject)
# We can perform SubTask 1.

objects = ['Apple-9213#jk', 'Bowl|aekgj|o', 'CounterTop|2|0', 'Tomato-#211a-be',
           'Fridge|2|1', 'GarbageBin-2199-ae98']

# CODE
def put_tomato_in_fridge():
    # 0: SubTask 1: Put Tomato in Fridge
    # 1: Go to the Tomato.
    GoToObject('Tomato')
    # 2: Pick up the Tomato.
    PickupObject('Tomato')
    # 3: Go to the Fridge.
    GoToObject('Fridge')
    # 4: Open the Fridge.
    OpenObject('Fridge')
    # 5: Put the Tomato in the Fridge.
    PutObject('Tomato', 'Fridge')
    # 6: Close the Fridge.
    CloseObject('Fridge')

# Perform SubTask 1
put_tomato_in_fridge()
# Task Put tomato in fridge is done

# EXAMPLE 2 - Task Description: Slice the Potato
# GENERAL TASK DECOMPOSITION
# Independent subtasks:
# SubTask 1: Slice the Potato. (Skills Required: GoToObject, PickupObject, SliceObject, PutObject)
# We can execute SubTask 1 first.

objects = ['Knife-9213#jk', 'Bowl|aekgj|o', 'CounterTop|2|0', 'Tomato-#211a-be',
           'Fridge|2|1', 'Potato-2199-ae98', "Dolphin-2919|I@1", "Jeremy|219|9"]

# CODE
def slice_potato():
    # 0: SubTask 1: Slice the Potato
    # 1: Go to the Knife.
    GoToObject('Knife')
    # 2: Pick up the Knife.
    PickupObject('Knife')
    # 3: Go to the Potato.
    GoToObject('Potato')
    # 4: Slice the Potato.
    SliceObject('Potato')
    # 5: Go to the countertop.
    GoToObject('CounterTop')
    # 6: Put the Knife back on the CounterTop.
    PutObject('Knife', 'CounterTop')
# Execute SubTask 1
slice_potato()
# Task sliced potato is done


# EXAMPLE 3 - Task Description: Throw the fork in the trash
# GENERAL TASK DECOMPOSITION
# Independent subtasks:
# SubTask 1: Pick up the Fork. (Skills Required: GoToObject, PickupObject)
# SubTask 2: Throw the Fork in the Trash. (Skills Required: GoToObject, ThrowObject)
# We can execute SubTask 1 first and then SubTask 2.

objects = ['Cylinder', 'Eagle', 'Celery|29|a', 'Cigar-21ae-86f8',
           'GarbageCan|2|1', 'Potato-2199-ae98', "Fork-2919|I@1", "Dog|219|9"]

# CODE
def pick_up_fork():
    # 0: SubTask 1: Pick up the Fork
    # 1: Go to the Fork.
    GoToObject('Fork')
    # 2: Pick up the Fork.
    PickupObject('Fork')

def throw_fork_in_trash():
    # 0: SubTask 2: Throw the Fork in the Trash
    # 1: Go to the GarbageCan.
    GoToObject('GarbageCan')
    # 2: Throw the Fork in the GarbageCan.
    ThrowObject('Fork', 'GarbageCan')

# Execute SubTask 1
pick_up_fork()

# Execute SubTask 2
throw_fork_in_trash()

# Task throw the fork in the trash is done
