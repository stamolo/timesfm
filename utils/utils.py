
def progress_bar(progress, max_progress, bar_size=12):
    f_bar = '='
    p_bar = '>'
    space = ' '
    percentage = progress/(max_progress-1)
    completed = round(percentage * bar_size)
    remaining = bar_size - completed
    if percentage == 1:
        completed_bar = f_bar * completed
    elif completed == 0:
        completed_bar = ''
    else:
        completed_bar = f_bar * (completed-1)
        completed_bar += p_bar
    progress_bar = f'{completed_bar}{space * remaining}'
    print(f'{int(percentage*100):3d}%', f'[{progress_bar}]', end='')
    
    if percentage == 1:
        print(end='\n')
    else:
        print(end='\r')