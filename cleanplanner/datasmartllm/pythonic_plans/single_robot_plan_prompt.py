# Note: skills that change object states (SliceObject, BreakObject, Open/CloseObject, CleanObject, etc) DO NOT change the object name.
# So, SliceObject('Potato') simply results in Potato having the slice property. This must be inferred from context, from previous actions in the plan.

# EXAMPLE TASK: Put Tomato in Fridge
# OBJECTS IN SCENE:
objects = ['Apple-9213#jk', 'Bowl|aekgj|o', 'CounterTop|2|0', 'Tomato-#211a-be',
           'Fridge|2|1: {"hasInsideOf": ["Apple-3981#0"]}', 'GarbageBin-2199-ae98']

# REASONING: We must pickup the tomato and place it inside the fridge. We'll need GoToObject, PickupObject, OpenObject, CloseObject, and PutObject skills.

# CODE
def put_tomato_in_fridge():
    # 0: Task : Put Tomato in Fridge
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

# Perform task
put_tomato_in_fridge()
# Task Put tomato in fridge is done

# EXAMPLE TASK: Slice the Potato
# OBJECTS IN SCENE:
objects = ['Knife-9213#jk', 'Bowl|aekgj|o', 'CounterTop|2|0', 'Tomato-#211a-be',
           'Fridge|2|1', 'Potato-2199-ae98', "Dolphin-2919|I@1", "Jeremy|219|9"]

# REASONING: We must slice the potato. We'll need GoToObject, PickupObject, SliceObject, and PutObject skills.

# CODE
def slice_potato():
    # 0: Task: Slice the Potato
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

# EXAMPLE TASK: Slice the Lettuce
# OBJECTS IN SCENE:
objects = ['Bowl|aekgj|o', 'CounterTop|2|0', 'Tomato-#211a-be',
           'Fridge|2|1', 'Lettuce-2199-ae98', "Dolphin-2919|I@1", "Jeremy|219|9"]

# REASONING: We must slice the lettuce. But! There's no knife! We cannot accomplish this plan

AbortPlan("We must use SliceObject, but there is no knife in the scene.")


# EXAMPLE TASK: Throw the fork in the trash
# OBJECTS IN SCENE:
objects = ['Cylinder', 'Eagle', 'Celery|29|a', 'Cigar-21ae-86f8',
           'GarbageCan|2|1', 'Potato-2199-ae98', "Fork-2919|I@1", "Dog|219|9"]

# REASONING: We must first pickup the fork. Then we put the fork in the trash. We'll need GoToObject, PickupObject, and ThrowObject skills.

# CODE
def pick_up_fork():
    # 0: Subtask: Pick up the Fork
    # 1: Go to the Fork.
    GoToObject('Fork')
    # 2: Pick up the Fork.
    PickupObject('Fork')

def throw_fork_in_trash():
    # 0: Subtask: Throw the Fork in the Trash
    # 1: Go to the GarbageCan.
    GoToObject('GarbageCan')
    # 2: Throw the Fork in the GarbageCan.
    ThrowObject('Fork', 'GarbageCan')

# Execute SubTask 1
pick_up_fork()

# Execute SubTask 2
throw_fork_in_trash()

# Task throw the fork in the trash is done

# ---------
# The above were just examples. You can reuse the imports, but do not reuse the functions and objects above!
# ---------
# Now, for the real code generation:
