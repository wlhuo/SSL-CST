import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

def read_gradient(file, nucleus_mask, patch_size, bin_size, dia_estimate):
    watershed2x = {}
    watershed2y = {}
    for i in range(nucleus_mask.shape[0]):
        for j in range(nucleus_mask.shape[1]):
            if nucleus_mask[i, j] == 0:
                continue
            if nucleus_mask[i, j] in watershed2x:
                watershed2x[nucleus_mask[i, j]].append(i)
                watershed2y[nucleus_mask[i, j]].append(j)
            else:
                watershed2x[nucleus_mask[i, j]] = [i]
                watershed2y[nucleus_mask[i, j]] = [j]

    watershed2center = {}
    sizes = []
    for nucleus in watershed2x:
        watershed2center[nucleus] = [np.mean(watershed2x[nucleus]), np.mean(watershed2y[nucleus])]
        sizes.append(len(watershed2x[nucleus]))


    class2dir = class_to_gradient()
    intensity = np.zeros(patch_size)
    dx = np.zeros(patch_size)
    dy = np.zeros(patch_size)
    class_num = 16
    pred_U = np.zeros(patch_size)
    pred_V = np.zeros(patch_size)
    pred_C = np.zeros(patch_size)
    with open(file) as fr:
        for line in fr:
            x, y, b, class_logit = line.split()
            class_prior = np.zeros(class_num)
            class_prior_ext = np.zeros(class_num)
            for center in watershed2center:
                dis_c = euclidean_distance(watershed2center[center][0], watershed2center[center][1], int(x), int(y))
                if dis_c <= int(dia_estimate):
                    class_prior += dir_to_class(watershed2center[center][0] - int(x) , watershed2center[center][1] - int(y), class_num, watershed2center[center][0], watershed2center[center][1])
                if dis_c <= int(dia_estimate) * 1.25:
                    class_prior_ext += dir_to_class(watershed2center[center][0] - int(x) , watershed2center[center][1] - int(y), class_num, watershed2center[center][0], watershed2center[center][1])
            if np.sum(class_prior) > 0:
                 class_prior = class_prior_ext
            class_prior_ori = np.ones(class_num) / class_num
            if np.sum(class_prior) == 0:
                class_prior = class_prior_ori
            else:
                class_prior = class_prior / np.sum(class_prior)
            class_prior = class_prior / np.sum(class_prior)
            class_logit = [float(logit) for logit in class_logit.split(":")]
            class_p = np.exp(class_logit) / sum(np.exp(class_logit))
            for k in range(class_num):
                class_p[k] = class_p[k] / class_prior_ori[k] * class_prior[k]
            class_p = class_p / np.sum(class_p)
            cla = np.argmax(class_p)
            cx, cy = class2dir[cla]
            dx[int(x), int(y)] = cx
            dy[int(x), int(y)] = cy
            intensity[int(x), int(y)] = float(b)
            pred_U[int(x), int(y)] = float(cy) * 10
            pred_V[int(x), int(y)] = - float(cx) * 10 
            pred_C[int(x), int(y)] = float(b)

    def diffuse(bin_size):
        dx_new = dx.copy()
        dy_new = dy.copy()
        intensity_new = intensity.copy()
        for i in range(dx_new.shape[0]):
            if i % bin_size == 0:
                for j in range(dx_new.shape[1]):
                    if j % bin_size == 0:
                        neighbor_grad_x = []
                        neighbor_grad_y = []
                        neighbor_intensity = []
                        for k in [-bin_size, 0, bin_size]:
                            for n in [-bin_size, 0, bin_size]:
                                if i + k >= 0 and i + k < dx_new.shape[0] and j + n >= 0 and j + n < dx_new.shape[1]:
                                    neighbor_grad_x.append(dx[i + k, j + n])
                                    neighbor_grad_y.append(dy[i + k, j + n])
                                    neighbor_intensity.append(intensity[i + k, j + n])
                        dx_new[i, j] = np.mean(neighbor_grad_x)
                        dy_new[i, j] = np.mean(neighbor_grad_y)
                        intensity_new[i, j] = np.mean(neighbor_intensity)
        return dx_new, dy_new, intensity_new

    for i in range(2):
        dx, dy, intensity = diffuse(bin_size)

    return intensity, dx, dy, pred_U, pred_V, pred_C

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 +  (y1 - y2) ** 2)

def dir_to_class(x, y, class_num, center_x, center_y):
    if y == 0 and x > 0:
        deg = np.arctan(float('inf'))
    elif y == 0 and x < 0:
        deg = np.arctan(-float('inf'))
    elif y == 0 and x == 0:
        deg = np.arctan(0)
    else:
        deg = np.arctan((x/y))
    if (x > 0 and y < 0) or (x <= 0 and y < 0):
        deg += np.pi
    elif x < 0 and y >= 0:
        deg += 2 * np.pi
    cla = int(deg / (2 * np.pi / class_num))
    prior_d = np.zeros(class_num)
    prior_d[cla] = 1
    return prior_d

def class_to_gradient():
    class2dir = {}
    for cla in range(16):
        xyratio = np.tan((cla) * 2 * np.pi/16 + np.pi/16)
        if cla >= 4 and cla <= 11:
            y = -1
        else:
            y = 1
        x = y * xyratio
        x_new = x / np.sqrt(x * x + y * y)
        y_new = y / np.sqrt(x * x + y * y)
        class2dir[cla] = (x_new, y_new)

    return class2dir

def gvf_tracking(dx, dy, intensity, bin_size, K=50, Diffusions=10, Mu=5, Lambda=5, Iterations=10,
                 dT=0.05):
    """
    References
    ----------
    G. Li et al "3D cell nuclei segmentation based on gradient flow
    tracking" in BMC Cell Biology,vol.40,no.8, 2007.
    """
    M = intensity.shape[0]
    N = intensity.shape[1]
    Mask = (intensity > 0.1).astype(int)

    intensity_values = []
    for i in range(intensity.shape[0]):
        for j in range(intensity.shape[1]):
            if intensity[i, j] > 0:
                intensity_values.append(intensity[i, j])

    plt.rcParams["figure.figsize"] = (15,15)

    Mag = np.sqrt(((dx**2 + dy**2) + np.finfo(float).eps))
    dy = dy / Mag
    dx = dx / Mag

    Mapped = np.zeros(intensity.shape)
    Segmentation = np.zeros(intensity.shape)
    Sinks = []

    i, j = np.nonzero(Mask)

    for index, (x, y) in enumerate(zip(i, j)):

        cosphi = 1
        points = 1
        novel = 1
        alloc = 1

        Trajectory = np.zeros((K, 2))
        Trajectory[0, 0] = int(x)
        Trajectory[0, 1] = int(y)
        last_xstep = 0
        last_ystep = 0

        while(cosphi > 0):
            xStep = round_float(dx[int(Trajectory[points-1, 0]), int(Trajectory[points-1, 1])]) * bin_size
            yStep = round_float(dy[int(Trajectory[points-1, 0]), int(Trajectory[points-1, 1])]) * bin_size
            if xStep == 0 and yStep == 0 and last_xstep==0 and last_ystep==0:
                novel = -1
                break
            elif xStep == 0 and yStep == 0:
                xStep = last_xstep
                yStep = last_ystep

            last_xstep = xStep
            last_ystep = yStep

            if ((Trajectory[points-1, 0] + xStep < 0) or
                    (Trajectory[points-1, 0] + xStep > M-1) or
                    (Trajectory[points-1, 1] + yStep < 0) or
                    (Trajectory[points-1, 1] + yStep > N-1)):
                break

            if points < K: 
                Trajectory[points, 0] = Trajectory[points-1, 0] + xStep
                Trajectory[points, 1] = Trajectory[points-1, 1] + yStep

            else: 

                cycle = detect_cycle(Trajectory, points)

                if cycle == points: 

                    temp = Trajectory
                    Trajectory = np.zeros((K*(alloc + 1), 2))
                    Trajectory[:K*alloc, ] = temp
                    alloc += 1

                    Trajectory[points, 0] = Trajectory[points-1, 0] + xStep
                    Trajectory[points, 1] = Trajectory[points-1, 1] + yStep

                else: 
                    points = cycle
                    break

            if Mapped[int(Trajectory[points, 0]), int(Trajectory[points, 1])] == 1:
                novel = 0
                cosphi = -1
            elif Mask[int(Trajectory[points, 0]), int(Trajectory[points, 1])] == 0:
                cosphi = -1
            else:
                cosphi = dy[int(Trajectory[points-1, 0]), int(Trajectory[points-1, 1])] * dy[int(Trajectory[points, 0]), int(Trajectory[points, 1])] + dx[int(Trajectory[points-1, 0]), int(Trajectory[points-1, 1])] * dx[int(Trajectory[points, 0]), int(Trajectory[points, 1])]

            points += 1

        if novel == 1:

            Sinks.append(Trajectory[points-1, ])

            for j in range(points):
                Segmentation[int(Trajectory[j, 0]), int(Trajectory[j, 1])] = len(Sinks)
                Mapped[int(Trajectory[j, 0]), int(Trajectory[j, 1])] = 1

        elif novel == 0:

            for j in range(points):
                Segmentation[int(Trajectory[j, 0]), int(Trajectory[j, 1])] = \
                    Segmentation[int(Trajectory[points-1, 0]),
                                 int(Trajectory[points-1, 1])]

    Sinks = np.asarray(Sinks)
    Sinks_select = []
    Segmentation_new = np.zeros(Segmentation.shape)
    for i, (x,y) in enumerate(Sinks):
        idx = np.where(Segmentation == i + 1)
        if len(idx[0]) <= 3:
            Segmentation[idx] = 0
        else:
            Sinks_select.append([x,y])
            Segmentation_new[np.where(Segmentation == i + 1)] = len(Sinks_select)
    Sinks = np.asarray(Sinks_select)

    return Segmentation_new, Sinks, dx, dy, Mask

def merge_sinks(Label, Sinks, downrs, Radius=3.5):
    import skimage.morphology as mp
    from skimage import measure as ms
    SeedImage = np.zeros(Label.shape)
    for i in range(Sinks.shape[0]):
        SeedImage[int(Sinks[i, 0]), int(Sinks[i, 1])] = i+1

    Dilated = mp.binary_dilation(SeedImage, mp.disk(Radius))
    Labels = ms.label(Dilated)
    New = Labels[Sinks[:, 0].astype(int), Sinks[:, 1].astype(int)]
    New = New + 1

    Unique = np.arange(1, New.max()+1)

    Merged = np.zeros(Label.shape)

    Props = ms.regionprops(Label.astype(int))
    for i in Unique:
        Indices = np.nonzero(New == i)[0]
        for j in Indices:
            Coords = Props[j].coords
            Merged[Coords[:, 0], Coords[:, 1]] = i

    filled = 1
    while(filled > 0):
        Merged, filled = fill_holes(Merged, int(downrs))

    return Merged

def fill_holes(label_mat, downrs):
    filled = 0
    label_mat_new = label_mat.copy()
    for i in range(label_mat.shape[0]):
        for j in range(label_mat.shape[1]):
            if i % downrs == 0 and j % downrs == 0 and label_mat[i, j] == 0:
                neighbor_labels = []
                for d in [[-downrs, -downrs], [-downrs, 0], [-downrs, downrs], [downrs, -downrs], [downrs, 0], [downrs, downrs], [0, -downrs], [0, downrs]]:
                    if i + d[0] >= 0 and i + d[0] < label_mat.shape[0] and j + d[1] >= 0 and j + d[1] < label_mat.shape[1]:
                        if label_mat[i + d[0], j + d[1]] > 0:
                            neighbor_labels.append(label_mat[i + d[0], j + d[1]])
                if len(neighbor_labels) >= 7 and len(set(neighbor_labels)) == 1:
                    label_mat_new[i, j] = neighbor_labels[0]
                    filled += 1
    return label_mat_new, filled

def remove_small_cells(label_mat):
    label_size = {}
    for i in range(label_mat.shape[0]):
        for j in range(label_mat.shape[1]):
            if label_mat[i , j] > 0:
                if label_mat[i , j] not in label_size:
                    label_size[label_mat[i , j]] = 1
                else:
                    label_size[label_mat[i , j]] += 1
    label_mat_new = np.zeros(label_mat.shape)
    for i in range(label_mat.shape[0]):
        for j in range(label_mat.shape[1]):
            if label_mat[i, j] > 0 and label_size[label_mat[i, j]] >= 200:
                label_mat_new[i, j] = label_mat[i, j]
    return label_mat_new

def detect_cycle(Trajectory, points):
    length = 0
    xMin = np.min(Trajectory[0:points, 0])
    xMax = np.max(Trajectory[0:points, 0])
    xRange = xMax - xMin + 1
    yMin = np.min(Trajectory[0:points, 1])
    yMax = np.max(Trajectory[0:points, 1])
    yRange = yMax - yMin + 1

    Map = np.zeros((int(xRange), int(yRange)))
    for i in range(points):
        if Map[int(Trajectory[i, 0]-xMin), int(Trajectory[i, 1]-yMin)] == 1:
            break
        else:
            Map[int(Trajectory[i, 0]-xMin), int(Trajectory[i, 1]-yMin)] = 1
        length += 1

    return length

def round_float(x):
    if x >= 0.0:
        t = np.ceil(x)
        if t - x > 0.5:
            t -= 1.0
        return t
    else:
        t = np.ceil(-x)
        if t + x > 0.5:
            t -= 1.0
        return -t

def result_stats(merged, startx, starty, patchsize,result_fold):
    cell2nspots = {}
    for i in range(merged.shape[0]):
        for j in range(merged.shape[1]):
            if merged[i, j] > 0:
                if merged[i, j] not in cell2nspots:
                    cell2nspots[merged[i, j]] = 1
                else:
                    cell2nspots[merged[i, j]] += 1
    all_sizes = [cell2nspots[cell] for cell in cell2nspots]

    with open(os.path.join(result_fold,'cell_stats_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.txt'), 'w') as fw:
        fw.write('Number of cells: ' + str(len(cell2nspots)) + '\n')
        fw.write('Patch size (spots x spots): ' + str(merged.shape[0]) + ' x ' + str(merged.shape[1]) + '\n')
        fw.write('Average cell size (spots): ' + str(np.mean(all_sizes)) + '\n')
        fw.write('Standard deviation of cell sizes: ' + str(np.std(all_sizes)) + '\n')

def postprocess(startx, starty, patchsize, bin_size, dia_estimate,nucleus_mask,result_fold):
    downrs = bin_size
    startx = str(startx)
    starty = str(starty)
    patchsize = str(patchsize)
    patchsizex = nucleus_mask.shape[0]
    patchsizey = nucleus_mask.shape[1]

    print('Adjust spot prediction priors...')
    intensity, dx, dy, pred_U, pred_V, pred_C = read_gradient(os.path.join(result_fold,'spot_prediction_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.txt'), nucleus_mask, (int(patchsizex), int(patchsizey)), bin_size, dia_estimate)
    print('Gradient flow tracking...')
    Segmentation, Sinks, dx, dy, mask = gvf_tracking(dx, dy, intensity, bin_size)
    print('Merge basins...')
    merged = merge_sinks(Segmentation, Sinks, downrs)
    sink_map = np.zeros(Segmentation.shape)
    for i, (x,y) in enumerate(Sinks):
        sink_map[int(x), int(y)] = i + 1

    expand = np.zeros(merged.shape)
    for i in range(merged.shape[0]):
        for j in range(merged.shape[1]):
            if merged[i, j] > 0 and expand[i, j] == 0:
                for m in range(int(downrs)):
                    for n in range(int(downrs)):
                        if i + m < int(patchsizex) and j + n < int(patchsizey):
                            merged[i + m, j + n] = merged[i, j]
                            expand[i + m, j + n] = 1
    merged = remove_small_cells(merged)

    edges = np.zeros(merged.shape)
    for i in range(merged.shape[0]):
        for j in range(merged.shape[1]):
            for k in [-1, 0, 1]:
                for n in [-1, 0, 1]:
                    if i + k >= 0 and i + k < merged.shape[0] and j + n >= 0 and j + n < merged.shape[1]:
                        if merged[i + k, j + n] != merged[i, j]:
                            edges[i, j] = 1
                            break

    fig, ax = plt.subplots(figsize=(32, 32), tight_layout=True)
    fw = open(os.path.join(result_fold,'spot2cell_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.txt'), 'w')
    for i in range(merged.shape[0]):
        for j in range(merged.shape[1]):
            if merged[i, j] > 0:
                fw.write(str(i) + ':' + str(j) + '\t' + str(merged[i, j]) + '\n')
    fw.close()

    nucl_labels = nucleus_mask
    fw = open(os.path.join(result_fold,'spot2nucl_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.txt'), 'w')
    for i in range(nucl_labels.shape[0]):
        for j in range(nucl_labels.shape[1]):
            if nucl_labels[i, j] > 0:
                fw.write(str(i) + ':' + str(j) + '\t' + str(nucl_labels[i, j]) + '\n')
    fw.close()

    result_stats(merged, startx, starty, patchsize,result_fold)

    idx = np.where(merged == 0)
    merged = merged % 9 + 1
    merged[idx] = 0
    tifffile.imwrite(os.path.join(result_fold,"merge.tif"),merged)
    plt.imshow(merged, alpha=0.6, cmap='tab10')
    plt.imshow(edges, alpha=0.2, cmap='Greys')
    q = ax.quiver(dy * mask * 10, - dx * mask * 10, intensity, scale=5, width=0.2, units='x')
    plt.savefig(os.path.join(result_fold,'cell_masks_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.png'))
