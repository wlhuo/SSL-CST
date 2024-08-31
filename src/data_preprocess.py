import os
import cv2
import math
import tifffile
import numpy as np
import pandas as pd
import scanpy as sc
import spateo as st
import anndata as ad
from PIL import Image
from numba import cuda
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix, vstack


Image.MAX_IMAGE_PIXELS = None



def gem_to_txt():
    gem_path = "/data/wlhuo/Stereopy/Stereopy_Demo_Data/Demo_MouseBrain/SS200000135TL_D1.tissue.gem.gz"
    new_path = "/data/wlhuo/wlhuo_research/data/profile.txt"
    import gzip
    with gzip.open(gem_path, 'rb') as f, open(new_path, 'wb') as f_out:
        f_out.write(f.read())

def split_image(img_path, output_dir, tile_width, tile_height):

    img = Image.open(img_path)
    img_width, img_height = img.size
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    columns = img_width // tile_width
    rows = img_height // tile_height
    
    for i in range(columns):
        for j in range(rows):
            left = i * tile_width
            upper = j * tile_height
            right = left + tile_width
            lower = upper + tile_height
            tile = img.crop((left, upper, right, lower))
            tile.save(os.path.join(output_dir, f"tile_{i}_{j}.tif"))

            
@cuda.jit
def calculate_distance_kernel(nucleus_mask, cellpose2center, distance_matrix):
    i, j = cuda.grid(2)
    if i < nucleus_mask[0] and j < nucleus_mask[1]:
        nucleus_distance = (i - cellpose2center[0])**2 + (j - cellpose2center[1])**2
        if nucleus_distance > 900:
            distance_matrix[i,j] = -math.sqrt(nucleus_distance - 900)
        else:
            distance_matrix[i,j] = math.sqrt(900 - nucleus_distance)
            
def seg_cell_cellpose(img_path):
    from .cell_pose import CellPose
    output = "/data/wlhuo/wlhuo_research/Mouse_brain_Adult_sub_cellpose.tif"
    cp = CellPose(
            img_path=img_path,
            out_path=output,
            model_type='cyto2',
            dmin=15,   # min cell diameter
            dmax=45,   # max cell diameter
            step=10
            )
    nucleus_mask = cp.segment_cells()
    return nucleus_mask
      
def preprocess(bin_file, image_file, prealigned, align, startx, starty, patchsize, bin_size, n_neighbor,result_fold):
    if prealigned:
        adatasub = st.io.read_bgi_agg(bin_file, image_file, prealigned=True)
    else:
        adatasub = st.io.read_bgi_agg(bin_file, image_file)
    if int(patchsize) > 0:
        adatasub = adatasub[int(startx):int(startx)+int(patchsize),int(starty):int(starty)+int(patchsize)]

    startx = str(startx)
    starty = str(starty)
    patchsize = str(patchsize)

    adatasub.layers['unspliced'] = adatasub.X
    patchsizex = adatasub.X.shape[0]
    patchsizey = adatasub.X.shape[1]

    before = adatasub.layers['stain'].copy()
    if align:
        st.cs.refine_alignment(adatasub, mode=align, transform_layers=['stain'])
    try:
        os.mkdir('fig/')
    except FileExistsError:
        print('fig folder exists')

    plt.savefig('fig/alignment' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.png')

    try:
        os.mkdir(result_fold)
    except FileExistsError:
        print('result_fold folder exists')


    tifffile.imsave(os.path.join(result_fold,"wlhuo_refine.tif"), adatasub.layers['stain'])
    nucleus_mask = seg_cell_cellpose("wlhuo_refine.tif")
    
    tifffile.imsave(os.path.join(result_fold,"wlhuo_after.tif"), nucleus_mask)
    
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(nucleus_mask)
    nucleus_mask_2x = {}
    nucleus_mask_2y = {}
    nucleus_mask_coo = {}
    for label in range(1, np.max(labels)+1):
        object_pixels = np.argwhere(labels == label)
        nucleus_mask_2x[label] = object_pixels[:,0]
        nucleus_mask_2y[label] = object_pixels[:,1]
        nucleus_mask_coo[label] = object_pixels

    cellpose2center = {}
    sizes = []
    for nucleus in nucleus_mask_2x:
        cellpose2center[nucleus] = [np.mean(nucleus_mask_2x[nucleus]), np.mean(nucleus_mask_2y[nucleus])]
        sizes.append(len(nucleus_mask_2x[nucleus]))
    distance_matrix_tmp = np.zeros((nucleus_mask.shape[0], nucleus_mask.shape[1]))
    distance_matrix = np.full((nucleus_mask.shape[0], nucleus_mask.shape[1]),int(math.sqrt(nucleus_mask.shape[0]**2 + nucleus_mask.shape[1]**2)))
    nucleus_mask_gpu = cuda.to_device(nucleus_mask.shape)
    distance_matrix_gpu = cuda.to_device(distance_matrix_tmp)

    threadsperblock = (16, 16)
    blockspergrid_x = (nucleus_mask.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (nucleus_mask.shape[1] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    for nucleus in range(len(cellpose2center)):
        cellpose2center_gpu = cuda.to_device(cellpose2center[nucleus+1])
        calculate_distance_kernel[blockspergrid, threadsperblock](nucleus_mask_gpu, cellpose2center_gpu, distance_matrix_gpu)
        distance_matrix_tmp = distance_matrix_gpu.copy_to_host()
        distance_matrix = np.minimum.reduce([distance_matrix_tmp, distance_matrix])


    xall = []
    yall = []
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            xall.append(int(x))
            yall.append(int(y))
    xmin = np.min(xall)
    ymin = np.min(yall)
    geneid = {}
    genecnt = 0
    id2gene = {}
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            if gene not in geneid:
                geneid[gene] = genecnt  
                id2gene[genecnt] = gene
                genecnt += 1

    idx2exp = {}
    downrs = bin_size
    with open(bin_file) as fr:
        header = fr.readline()
        for line in fr:
            gene, x, y, count = line.split()
            x = int(x) - xmin
            y = int(y) - ymin
            if gene not in geneid:
                continue
            if int(x) < int(startx) or int(x) >= int(startx) + int(patchsizex) or int(y) < int(starty) or int(y) >= int(starty) + int(patchsizey):
                continue
            idx = int(math.floor((int(x) - int(startx)) / downrs) * math.ceil(patchsizey / downrs) + math.floor((int(y) - int(starty)) / downrs))
            if idx not in idx2exp:
                idx2exp[idx] = {}
                idx2exp[idx][geneid[gene]] = int(count)
            elif geneid[gene] not in idx2exp[idx]:
                idx2exp[idx][geneid[gene]] = int(count)
            else:
                idx2exp[idx][geneid[gene]] += int(count)

    all_exp_merged_bins = lil_matrix((int(math.ceil(patchsizex / downrs) * math.ceil(patchsizey / downrs)), genecnt), dtype=np.int8)
    for idx in idx2exp:
        for gid in idx2exp[idx]:
            all_exp_merged_bins[idx, gid] = idx2exp[idx][gid]
    all_exp_merged_bins = all_exp_merged_bins.tocsr()

    all_exp_merged_bins_ad = ad.AnnData(
        all_exp_merged_bins,
        obs=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[0])]),
        var=pd.DataFrame(index=[i for i in range(all_exp_merged_bins.shape[1])]),
    )
    sc.pp.highly_variable_genes(all_exp_merged_bins_ad, n_top_genes=2000, flavor='seurat_v3', span=1.0)
    selected_index = all_exp_merged_bins_ad.var[all_exp_merged_bins_ad.var.highly_variable].index
    selected_index = list(selected_index)
    selected_index = [int(i) for i in selected_index]

    with open('data/variable_genes' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.txt', 'w') as fw:
        for id in selected_index:
            fw.write(id2gene[id] + '\n')


    all_exp_merged_bins = all_exp_merged_bins.toarray()[:, selected_index]
    x_train_tmp = []
    x_train = []
    x_train_pos = []
    x_train_distance = []
    y_train = []
    y_binary_train = []
    x_train_bg_tmp = []
    x_train_bg = []
    x_train_pos_bg = []
    x_train_distance_bg = []
    y_train_bg = []
    y_binary_train_bg = []
    x_test_tmp = []
    x_test= []
    x_test_pos = []
    x_test_distance = []
    offsets = []
    for dis in range(1, 11):
        for dy in range(-dis, dis + 1):
            offsets.append([-dis * downrs, dy * downrs])
        for dy in range(-dis, dis + 1):
            offsets.append([dis * downrs, dy * downrs])
        for dx in range(-dis + 1, dis):
            offsets.append([dx * downrs, -dis * downrs])
        for dx in range(-dis + 1, dis):
            offsets.append([dx * downrs, dis * downrs])
    for i in range(nucleus_mask.shape[0]):
        if (i + 1) % 100 == 0:
            print("finished {0:.0%}".format(i / nucleus_mask.shape[0]))
        for j in range(nucleus_mask.shape[1]):
            if (not i % downrs == 0) or (not j % downrs == 0):
                continue
            idx = int(math.floor(i / downrs) * math.ceil(patchsizey / downrs) + math.floor(j / downrs))
            if nucleus_mask[i, j] > 0:
                if idx >= 0 and idx < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx, :]) > 0:
                    x_train_sample = [all_exp_merged_bins[idx, :]]
                    x_train_pos_sample = [[i, j]]
                    for key,value in nucleus_mask_coo.items():
                        if [i, j] in value.tolist():
                            y_train_sample = cellpose2center[key]
                            break
                    if len(y_train_sample) == 0:
                        print('{} not in nucleus_mask_coo'.format([i,j]))
                    x_train_distance_sample = [[distance_matrix[i,j]]]
                    for dx, dy in offsets:
                        if len(x_train_sample) == n_neighbor:
                            break
                        x = i + dx
                        y = j + dy
                        if x < 0 or x >= nucleus_mask.shape[0] or y < 0 or y >= nucleus_mask.shape[1]:
                            continue
                        idx_nb = int(math.floor(x / downrs) * math.ceil(patchsizey / downrs) + math.floor(y / downrs))
                        if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx_nb, :]) > 0:
                            x_train_sample.append(all_exp_merged_bins[idx_nb, :])
                            x_train_pos_sample.append([x, y])
                            x_train_distance_sample.append([distance_matrix[x,y]])
                    if len(x_train_sample) < n_neighbor:
                        continue
                    x_train_tmp.append(x_train_sample)
                    if len(x_train_tmp) > 500:
                        x_train.extend(x_train_tmp)
                        x_train_tmp = []
                    x_train_pos.append(x_train_pos_sample)
                    x_train_distance.append(x_train_distance_sample)
                    y_train.append(y_train_sample)
                    y_binary_train.append(1)
            else: 
                if idx >= 0 and idx < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx, :]) > 0:
                    backgroud = True
                    for nucleus in cellpose2center:
                        if (i - cellpose2center[nucleus][0]) ** 2 + (j - cellpose2center[nucleus][1]) ** 2 <= 900 or adatasub.layers['stain'][i, j] > 10:
                            backgroud = False
                            break
                    if backgroud:
                        if len(x_train_bg) + len(x_train_bg_tmp) >= len(x_train) + len(x_train_tmp):
                            continue
                        x_train_sample = [all_exp_merged_bins[idx, :]]
                        x_train_pos_sample = [[i, j]]
                        x_train_distance_sample = [[distance_matrix[i,j]]]
                        y_train_sample = [[-1, -1]]
                        for dx, dy in offsets:
                            if len(x_train_sample) == n_neighbor:
                                break
                            x = i + dx
                            y = j + dy
                            if x < 0 or x >= nucleus_mask.shape[0] or y < 0 or y >= nucleus_mask.shape[1]:
                                continue
                            idx_nb = int(math.floor(x / downrs) * math.ceil(patchsizey / downrs) + math.floor(y / downrs))
                            if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx_nb, :]) > 0:
                                x_train_sample.append(all_exp_merged_bins[idx_nb, :])
                                x_train_pos_sample.append([x, y])
                                x_train_distance_sample.append([distance_matrix[x,y]])
                        if len(x_train_sample) < n_neighbor:
                            continue
                        x_train_bg_tmp.append(x_train_sample)
                        if len(x_train_bg_tmp) > 500:
                            x_train_bg.extend(x_train_bg_tmp)
                            x_train_bg_tmp = []
                        x_train_pos_bg.append(x_train_pos_sample)
                        x_train_distance_bg.append(x_train_distance_sample)
                        y_train_bg.append(y_train_sample)
                        y_binary_train_bg.append(0)
                    else:
                        x_test_sample = [all_exp_merged_bins[idx, :]]
                        x_test_pos_sample = [[i, j]]
                        x_test_distance_sample = [distance_matrix[i,j]]
                        for dx, dy in offsets:
                            if len(x_test_sample) == n_neighbor:
                                break
                            x = i + dx
                            y = j + dy
                            if x < 0 or x >= nucleus_mask.shape[0] or y < 0 or y >= nucleus_mask.shape[1]:
                                continue
                            idx_nb = int(math.floor(x / downrs) * math.ceil(patchsizey / downrs) + math.floor(y / downrs))
                            if idx_nb >= 0 and idx_nb < all_exp_merged_bins.shape[0] and np.sum(all_exp_merged_bins[idx_nb, :]) > 0:
                                x_test_sample.append(all_exp_merged_bins[idx_nb, :])
                                x_test_pos_sample.append([x, y])
                                x_test_distance_sample.append(distance_matrix[x,y])
                        if len(x_test_sample) < n_neighbor:
                            continue
                        x_test_tmp.append(x_test_sample)
                        if len(x_test_tmp) > 500:
                            x_test.extend(x_test_tmp)
                            x_test_tmp = []
                        x_test_pos.append(x_test_pos_sample)
                        x_test_distance.append(x_test_distance_sample)
    x_train.extend(x_train_tmp)
    x_train_bg.extend(x_train_bg_tmp)
    x_test.extend(x_test_tmp)

    x_train = np.array(x_train)
    x_train_pos = np.array(x_train_pos)
    x_train_distance = np.array(x_train_distance)
    y_train = np.vstack(y_train)
    y_binary_train = np.array(y_binary_train)
    x_train_bg = np.array(x_train_bg)
    x_train_pos_bg = np.array(x_train_pos_bg)
    x_train_distance_bg = np.array(x_train_distance_bg)
    y_train_bg = np.vstack(y_train_bg)
    y_binary_train_bg = np.array(y_binary_train_bg)

    bg_index = np.arange(len(x_train_bg))
    np.random.shuffle(bg_index)
    x_train = np.vstack((x_train, x_train_bg[bg_index[:len(x_train)]]))
    x_train_pos = np.vstack((x_train_pos, x_train_pos_bg[bg_index[:len(x_train_pos)]]))
    x_train_distance = np.vstack((x_train_distance, x_train_distance_bg[bg_index[:len(x_train_distance)]]))
    y_train = np.vstack((y_train, y_train_bg[bg_index[:len(y_train)]]))
    y_binary_train = np.hstack((y_binary_train, y_binary_train_bg[bg_index[:len(y_binary_train)]]))

    x_test= np.array(x_test)
    x_test_pos = np.array(x_test_pos)
    x_test_distance = np.array(x_test_distance)

    np.savez_compressed('data/x_train_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_train=x_train)
    np.savez_compressed('data/x_train_pos_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_train_pos=x_train_pos)
    np.savez_compressed('data/x_train_distance_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_train_distance=x_train_distance)
    np.savez_compressed('data/y_train_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', y_train=y_train)
    np.savez_compressed('data/y_binary_train_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', y_binary_train=y_binary_train)
    np.savez_compressed('data/x_test_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_test=x_test)
    np.savez_compressed('data/x_test_pos_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_test_pos=x_test_pos)
    np.savez_compressed('data/x_test_distance_' + startx + ':' + starty + ':' + patchsize + ':' + patchsize + '.npz', x_test_distance=x_test_distance)
    
    return nucleus_mask
    

            
if __name__ == "__main__":
    nucleus_mask = seg_cell_cellpose("/data/wlhuo/SSL-CST/data/Mouse_brain_Adult_sub.tif")