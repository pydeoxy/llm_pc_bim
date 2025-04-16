import numpy as np

# Classes in the S3DIS dataset
# class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
#               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')

# Define a fixed color map for 13 labels
color_map_dict = {
    0:[[1.0, 0.0, 0.0],'ceiling'],  # Label 0: Red 
    1:[[0.0, 1.0, 0.0],'floor'],  # Label 1: Green
    2:[[0.0, 0.0, 1.0],'wall'],  # Label 2: Blue
    3:[[1.0, 1.0, 0.0],'beam'],  # Label 3: Yellow
    4:[[1.0, 0.0, 1.0],'column'],  # Label 4: Magenta
    5:[[0.0, 1.0, 1.0],'window'],  # Label 5: Cyan
    6:[[0.5, 0.5, 0.5],'door'],  # Label 6: Gray
    7:[[1.0, 0.5, 0.0],'table'],  # Label 7: Orange
    8:[[0.5, 0.0, 1.0],'chair'],  # Label 8: Purple
    9:[[0.5, 1.0, 0.5],'sofa'],  # Label 9: Light Green
    10:[[0.5, 0.5, 1.0],'bookcase'],  # Label 10: Light Blue
    11:[[1.0, 0.5, 0.5],'board'],  # Label 11: Pink
    12:[[0.0, 0.0, 0.0],'clutter']   # Label 12: Black
}

colors = [value[0] for value in color_map_dict.values()]

color_map = np.array(colors)

num_classes = len(colors)

if __name__ == '__main__':
    print(colors)