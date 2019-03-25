from copy import deepcopy
import os

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from timagetk.components import SpatialImage
from timagetk.io import imread, imsave

from timagetk.wrapping.bal_trsf import TRSF_TYPE_DICT, TRSF_UNIT_DICT
from timagetk.algorithms.trsf import allocate_c_bal_matrix, apply_trsf, create_trsf
from timagetk.algorithms.reconstruction import pts2transfo

from vplants.tissue_nukem_3d.nuclei_mesh_tools import nuclei_image_surface_topomesh
from vplants.tissue_nukem_3d.nuclei_mesh_tools import cut_surface_topomesh, up_facing_surface_topomesh

from vplants.tissue_nukem_3d.signal_map import SignalMap, plot_signal_map
from vplants.tissue_nukem_3d.signal_map_analysis import signal_map_regions

from vplants.cellcomplex.property_topomesh.property_topomesh_io import save_ply_property_topomesh
from vplants.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces
from vplants.cellcomplex.property_topomesh.property_topomesh_extraction import topomesh_connected_components, clean_topomesh

from vplants.cellcomplex.property_topomesh.utils.pandas_tools import topomesh_to_dataframe


def extract_curvature_circle(signal_data, position_name='center', curvature_threshold_range=np.linspace(0.06, 0.09, 4), cell_radius=1, density_k=0.66):
    if not "mean_curvature" in signal_data.columns:
        raise KeyError("mean_curvature signal is not defined!")
    signal_map = SignalMap(signal_data, extent=50, position_name=position_name, resolution=0.5, polar=False, radius=cell_radius, density_k=density_k)

    centers = []
    radii = []
    for curvature_threshold in curvature_threshold_range:
        curvature_regions = signal_map_regions(signal_map, 'mean_curvature', threshold=curvature_threshold)
        print(curvature_regions)
        if len(curvature_regions) > 0:
            curvature_region = curvature_regions.iloc[np.argmin(np.linalg.norm(curvature_regions[['center_x', 'center_y']].values, axis=1))]
            curvature_radius = np.sqrt(curvature_region['area'] / np.pi)
            curvature_center = curvature_region[['center_x', 'center_y']].values
            centers += [curvature_center]
            radii += [curvature_radius]
            print(curvature_center, curvature_radius)

    curvature_center = np.mean(centers, axis=0)
    curvature_radius = np.mean(radii)

    return curvature_center, curvature_radius


def optimize_vertical_axis(positions, angle_max=0.2, angle_resolution=0.01, r_max=80):
    """
    """

    angles = np.linspace(-angle_max, angle_max, 2 * (angle_max / angle_resolution) + 1)

    psis, phis = np.meshgrid(angles, angles)

    rotation_mse = []

    for dome_phi in angles:
        phi_mse = []
        for dome_psi in angles:
            rotation_matrix_psi = np.array([[1, 0, 0], [0, np.cos(dome_psi), -np.sin(dome_psi)], [0, np.sin(dome_psi), np.cos(dome_psi)]])
            rotation_matrix_phi = np.array([[np.cos(dome_phi), 0, -np.sin(dome_phi)], [0, 1, 0], [np.sin(dome_phi), 0, np.cos(dome_phi)]])
            rotated_positions = np.einsum('...ij,...j->...i', rotation_matrix_psi, positions)
            rotated_positions = np.einsum('...ij,...j->...i', rotation_matrix_phi, rotated_positions)

            rotated_r = np.linalg.norm(rotated_positions[:, :2], axis=1)
            rotated_z = rotated_positions[:, 2]
            r_weights = np.exp(-np.power(rotated_r, 2) / np.power(20, 2))
            p = np.polyfit(rotated_r, rotated_z, deg=2, w=r_weights)

            r = np.linspace(0, r_max, r_max + 1)
            mse = (r_weights * np.power(rotated_z - np.polyval(p, rotated_r), 2)).sum() / (r_weights.sum())
            phi_mse += [mse]
        rotation_mse += [phi_mse]

    optimal_rotation = np.where(rotation_mse == np.array(rotation_mse).min())
    optimal_phi = (phis[optimal_rotation]).mean()
    optimal_psi = (psis[optimal_rotation]).mean()

    rotation_matrix_psi = np.array([[1, 0, 0], [0, np.cos(optimal_psi), -np.sin(optimal_psi)], [0, np.sin(optimal_psi), np.cos(optimal_psi)]])
    rotation_matrix_phi = np.array([[np.cos(optimal_phi), 0, -np.sin(optimal_phi)], [0, 1, 0], [np.sin(optimal_phi), 0, np.cos(optimal_phi)]])
    rotated_positions = np.einsum('...ij,...j->...i', rotation_matrix_psi, positions)
    rotated_positions = np.einsum('...ij,...j->...i', rotation_matrix_phi, rotated_positions)

    return rotated_positions, np.dot(rotation_matrix_phi, rotation_matrix_psi)


dirname = "/Users/gcerutti/Projects/Imago/Gabriella_Mesh/"

filenames = []
filenames += ["EM_C_214"]
filenames += ["EM_C_136"]
filenames += ["EM_C_140"]
filenames += ["EM_C_160"]

for filename in filenames:

    signal_filename = dirname + "/" + filename + "/" + filename + "_signal.tif"
    if os.path.exists(signal_filename):
        signal_img = imread(signal_filename)
    else:
        signal_img = None

    seg_filename = dirname + "/" + filename + "/" + filename + "_seg.tif"
    if os.path.exists(seg_filename):
        seg_img = imread(seg_filename)
        voxelsize = np.array(seg_img.voxelsize)

        surface_topomesh = nuclei_image_surface_topomesh(SpatialImage(20000*(seg_img.get_array()>0).astype(np.uint16),voxelsize=seg_img.voxelsize),density_voxelsize=0.5,maximal_length=5.)

        compute_topomesh_property(surface_topomesh,'vertices',2)
        compute_topomesh_property(surface_topomesh,'normal',2,normal_method='orientation')
        compute_topomesh_vertex_property_from_faces(surface_topomesh,'normal',neighborhood=5,adjacency_sigma=3.)
        compute_topomesh_property(surface_topomesh,'mean_curvature',2)
        compute_topomesh_vertex_property_from_faces(surface_topomesh,'mean_curvature',neighborhood=5,adjacency_sigma=3.)

        face_vertex_curvature = surface_topomesh.wisp_property('mean_curvature',0).values(surface_topomesh.wisp_property('vertices',2).values(list(surface_topomesh.wisps(2))))

        curvature_threshold = 0.05
        compactness_threshold = 0.5
        surface_topomesh.update_wisp_property('bulge',2,dict(zip(surface_topomesh.wisps(2),np.all(face_vertex_curvature>curvature_threshold,axis=1).astype(int))))

        save_ply_property_topomesh(surface_topomesh,dirname + "/" + filename + "/" + filename + "_surface_topomesh.ply",properties_to_save={0:['mean_curvature','normal'],1:[],2:['mean_curvature','normal','bulge'],3:[]})


        bulge_topomesh = deepcopy(surface_topomesh)
        faces_to_remove = np.array(list(bulge_topomesh.wisps(2)))[np.logical_not(bulge_topomesh.wisp_property('bulge',2).values(list(bulge_topomesh.wisps(2))))]
        for f in faces_to_remove:
            bulge_topomesh.remove_wisp(2,f)
        bulges = topomesh_connected_components(bulge_topomesh)

        bulge_areas = {}
        bulge_perimeters = {}
        bulge_compactness = {}
        for i_b, b in enumerate(bulges):
            b = clean_topomesh(b)
            compute_topomesh_property(b,'area',2)
            bulge_areas[i_b] = np.sum(b.wisp_property('area',2).values())

            compute_topomesh_property(b,'edges',2)
            compute_topomesh_property(b,'length',1)

            b.update_wisp_property('boundary', 1, dict(zip(b.wisps(1), [b.nb_regions(1, e) < 2 for e in b.wisps(1)])))
            bulge_perimeters[i_b] = np.sum(b.wisp_property('length',1).values()*b.wisp_property('boundary',1).values())

            bulge_compactness[i_b] = 2*np.sqrt(np.pi*bulge_areas[i_b]) / bulge_perimeters[i_b]


        candidate_bulges = [i_b for i_b, b in enumerate(bulges)]
        candidate_bulges = [i_b for i_b in candidate_bulges if bulge_compactness[i_b]>compactness_threshold]
        candidate_bulge = np.array(candidate_bulges)[np.argmax([bulge_areas[i_b] for i_b in candidate_bulges])]

        bulge_topomesh = bulges[candidate_bulge]

        tip_vertex = np.array(list(bulge_topomesh.wisps(0)))[np.argmax(bulge_topomesh.wisp_property('mean_curvature',0).values(list(bulge_topomesh.wisps(0))))]

        surface_topomesh.update_wisp_property('tip',2,dict(zip(surface_topomesh.wisps(2),[int(tip_vertex in surface_topomesh.borders(2,f,2)) for f in surface_topomesh.wisps(2)])))

        save_ply_property_topomesh(surface_topomesh,dirname + "/" + filename + "/" + filename + "_surface_topomesh.ply",properties_to_save={0:['mean_curvature','normal'],1:[],2:['mean_curvature','normal','bulge','tip'],3:[]})

        tip_point = surface_topomesh.wisp_property('barycenter',0)[tip_vertex]
        tip_normal = surface_topomesh.wisp_property('normal',0)[tip_vertex]

        points = surface_topomesh.wisp_property('barycenter',0).values()

        centered_points = points - tip_point

        # axis_phi = np.arcsin([0])
        # axis_psi = np.arcsin(tip_normal[1])
        #
        # rotation_matrix_psi = np.array([[1, 0, 0], [0, np.cos(axis_psi), -np.sin(axis_psi)], [0, np.sin(axis_psi), np.cos(axis_psi)]])
        # rotation_matrix_phi = np.array([[np.cos(axis_phi), 0, -np.sin(axis_phi)], [0, 1, 0], [np.sin(axis_phi), 0, np.cos(axis_phi)]])
        # rotated_points = np.einsum('...ij,...j->...i', rotation_matrix_psi, centered_points)
        # rotated_points = np.einsum('...ij,...j->...i', rotation_matrix_phi, rotated_points)

        cross_vector = np.cross(tip_normal,[0,0,1])
        rotation_sin = np.linalg.norm(cross_vector)
        rotation_cos = np.dot(tip_normal,[0,0,1])

        v1,v2,v3 = cross_vector
        skew_symmetric_cross_matrix = np.array([[0,-v3,v2],[v3,0,-v1],[-v2,v1,0]])
        rotation_matrix = np.identity(3) + skew_symmetric_cross_matrix + np.dot(skew_symmetric_cross_matrix, skew_symmetric_cross_matrix) * (1 - rotation_cos) / np.power(rotation_sin, 2)
        rotated_points = np.einsum('...ij,...j->...i', rotation_matrix, centered_points)

        # v1, v2 = tip_normal, np.array([0.,0.,1.])
        # rotation_matrix = 2*np.dot((v1+v2)[:,np.newaxis],(v1+v2)[np.newaxis])/np.dot((v1+v2)[np.newaxis],(v1+v2)[:,np.newaxis]) - np.identity(3)
        # rotated_points = np.einsum('...ij,...j->...i', rotation_matrix, centered_points)

        figure = plt.figure(0)
        figure.clf()

        figure.add_subplot(1, 2, 1)

        figure.gca().scatter(points[:, 0], points[:, 2], color='k', alpha=0.1, s=10)
        figure.gca().scatter([tip_point[0]], [tip_point[2]], color='r', s=20)
        figure.gca().arrow(tip_point[0], tip_point[2], tip_normal[0], tip_normal[2], color='r')
        figure.gca().plot([tip_point[0] - 10. * tip_normal[0], tip_point[0] + 10. * tip_normal[0]], [tip_point[2] - 10. * tip_normal[2], tip_point[2] + 10. * tip_normal[2]], color='r', alpha=0.1)

        figure.add_subplot(1,2,2)

        figure.gca().scatter(rotated_points[:,0],rotated_points[:,2],color='k',alpha=0.1,s=10)
        figure.gca().scatter([0],[0],color='r',s=20)
        figure.gca().arrow(0,0,0,1,color='r')
        figure.gca().plot([0,0],[-10.,10.],color='r',alpha=0.1)

        figure.set_size_inches(20,10)
        figure.savefig(dirname + "/" + filename + "/" + filename + "_ovule_surface_axis_rotation.png")

        rotated_surface_topomesh = deepcopy(surface_topomesh)
        rotated_surface_topomesh.update_wisp_property('barycenter',0,dict(zip(surface_topomesh.wisps(0),rotated_points)))
        rotated_surface_topomesh = up_facing_surface_topomesh(rotated_surface_topomesh,normal_method='orientation')

        compute_topomesh_property(rotated_surface_topomesh,'vertices',2)
        compute_topomesh_property(rotated_surface_topomesh,'normal',2,normal_method='orientation')
        compute_topomesh_vertex_property_from_faces(rotated_surface_topomesh,'normal',neighborhood=4,adjacency_sigma=1.5)

        save_ply_property_topomesh(rotated_surface_topomesh,dirname + "/" + filename + "/" + filename + "_rotated_surface_topomesh.ply",properties_to_save={0:['mean_curvature','normal'],1:[],2:['mean_curvature','normal'],3:[]})

        surface_data = topomesh_to_dataframe(rotated_surface_topomesh,0)

        surface_data['image_x'] = surface_topomesh.wisp_property('barycenter',0).values(list(rotated_surface_topomesh.wisps(0)))[:,0]
        surface_data['image_y'] = surface_topomesh.wisp_property('barycenter',0).values(list(rotated_surface_topomesh.wisps(0)))[:,1]
        surface_data['image_z'] = surface_topomesh.wisp_property('barycenter',0).values(list(rotated_surface_topomesh.wisps(0)))[:,2]

        surface_data['radial_distance'] = np.linalg.norm(surface_data[['center_x','center_y']].values,axis=1)
        surface_data['aligned_theta'] = 180. / np.pi * np.sign(surface_data['center_y']) * np.arccos(surface_data['center_x'] / surface_data['radial_distance'])
        surface_data['aligned_theta'][surface_data['radial_distance']==0] = 0.

        curvature_center, curvature_radius = extract_curvature_circle(surface_data)

        surface_map = SignalMap(surface_data,extent=20,resolution=0.5,density_k=0.66,radius=1)
        surface_map.compute_signal_map('mean_curvature')
        surface_map.compute_signal_map('center_z')

        figure = plt.figure(1)
        figure.clf()

        figure.add_subplot(1,3,1)
        figure.gca().scatter(surface_data['center_x'],surface_data['center_y'],c=surface_data['mean_curvature'],cmap='RdBu_r',vmin=-0.1,vmax=0.1)
        figure.gca().set_xlim(-20,20)
        figure.gca().set_ylim(-20,20)

        figure.add_subplot(1,3,2)
        plot_signal_map(surface_map,'mean_curvature',figure,colormap='RdBu_r',signal_range=(-0.5,0.5),signal_lut_range=(-0.1,0.1),distance_rings=False)
        figure.gca().scatter(curvature_center[0],curvature_center[1],color='k')
        figure.gca().axis('on')
        figure.gca().set_xlim(-20,20)
        figure.gca().set_ylim(-20,20)

        figure.add_subplot(1,3,3)
        plot_signal_map(surface_map,'center_z',figure,colormap='viridis',signal_range=(-80,20),signal_lut_range=(-40,0),distance_rings=False)
        figure.gca().axis('on')
        figure.gca().set_xlim(-20,20)
        figure.gca().set_ylim(-20,20)

        figure.set_size_inches(15,5)
        figure.savefig(dirname + "/" + filename + "/" + filename + "_ovule_surface_curvature_projection.png")

        recentered_points = surface_data[['center_x','center_y','center_z']].values - np.array(list(curvature_center)+[0])

        tilted_points, tilting_matrix = optimize_vertical_axis(recentered_points, r_max=10.)

        figure = plt.figure(2)
        figure.clf()

        figure.add_subplot(1,3,1)
        figure.gca().scatter(surface_data['radial_distance'], surface_data['center_z'], alpha=0.1)
        figure.gca().set_xlim(0, 20)
        figure.gca().set_ylim(-35, 5)

        figure.add_subplot(1,3,2)
        figure.gca().scatter(np.linalg.norm(recentered_points[:,:2],axis=1), recentered_points[:,2], alpha=0.1)
        figure.gca().set_xlim(0, 20)
        figure.gca().set_ylim(-35, 5)

        figure.add_subplot(1,3,3)
        figure.gca().scatter(np.linalg.norm(tilted_points[:,:2],axis=1), tilted_points[:,2], alpha=0.1)
        figure.gca().set_xlim(0, 20)
        figure.gca().set_ylim(-35, 5)

        figure.canvas.draw()
        figure.set_size_inches(15,10)
        figure.savefig(dirname + "/" + filename + "/" + filename + "_ovule_surface_curve_dispersion.png")

        rotated_surface_topomesh.update_wisp_property('barycenter',0,dict(zip(rotated_surface_topomesh.wisps(0),tilted_points)))

        compute_topomesh_property(rotated_surface_topomesh,'normal',2,normal_method='orientation')
        compute_topomesh_vertex_property_from_faces(rotated_surface_topomesh,'normal',neighborhood=4,adjacency_sigma=1.5)

        save_ply_property_topomesh(rotated_surface_topomesh,dirname + "/" + filename + "/" + filename + "_tilted_surface_topomesh.ply",properties_to_save={0:['mean_curvature','normal'],1:[],2:['mean_curvature','normal'],3:[]})

        surface_data['rotated_x'] = tilted_points[:,0]
        surface_data['rotated_y'] = tilted_points[:,1]
        surface_data['rotated_z'] = tilted_points[:,2]

        surface_map = SignalMap(surface_data,extent=20,resolution=0.5,position_name='rotated',density_k=0.66,radius=1)
        surface_map.compute_signal_map('mean_curvature')
        surface_map.compute_signal_map('center_z')

        base_curvature_threshold = 0.02
        base_mask = (surface_map.signal_map('mean_curvature') < base_curvature_threshold)

        base_x = np.mean(surface_map.xx[base_mask])
        base_y = np.mean(surface_map.yy[base_mask])
        base_z = np.mean(surface_map.signal_map('rotated_z')[base_mask])
        base_r = np.linalg.norm([base_x,base_y])
        base_theta = np.sign(base_y)*np.arccos(base_x/base_r)

        base_thetas = surface_map.tt[base_mask]
        base_theta = base_theta + np.mean(((base_thetas-base_theta + np.pi)%(2*np.pi))-np.pi)

        figure = plt.figure(3)
        figure.clf()

        figure.add_subplot(1,3,1)
        figure.gca().scatter(surface_data['rotated_x'],surface_data['rotated_y'],c=surface_data['mean_curvature'],cmap='RdBu_r',vmin=-0.1,vmax=0.1)
        figure.gca().set_xlim(-20,20)
        figure.gca().set_ylim(-20,20)

        figure.add_subplot(1,3,2)
        plot_signal_map(surface_map,'mean_curvature',figure,colormap='RdBu_r',signal_range=(-0.5,0.5),signal_lut_range=(-0.1,0.1),distance_rings=False)
        figure.gca().contour(surface_map.xx,surface_map.yy,base_mask,[0.5],colors=['k'],alpha=0.25)
        figure.gca().plot([0,20*np.cos(base_theta)],[0,20*np.sin(base_theta)], color='k', alpha=0.5)
        figure.gca().axis('on')
        figure.gca().set_xlim(-20,20)
        figure.gca().set_ylim(-20,20)

        figure.add_subplot(1,3,3)
        plot_signal_map(surface_map,'center_z',figure,colormap='viridis',signal_range=(-80,20),signal_lut_range=(-40,0),distance_rings=False)
        figure.gca().axis('on')
        figure.gca().set_xlim(-20,20)
        figure.gca().set_ylim(-20,20)

        figure.set_size_inches(15,5)
        figure.savefig(dirname + "/" + filename + "/" + filename + "_ovule_surface_curvature_tilted_projection.png")

        base_rotation_matrix = np.array([[np.cos(base_theta),np.sin(base_theta),0.],[-np.sin(base_theta),np.cos(base_theta),0.],[0.,0.,1.]])

        aligned_points = deepcopy(tilted_points)
        aligned_points[:,2] -= base_z
        aligned_points = np.einsum('...ij,...j->...i', base_rotation_matrix, aligned_points)

        rotated_surface_topomesh.update_wisp_property('barycenter',0,dict(zip(rotated_surface_topomesh.wisps(0),aligned_points)))
        save_ply_property_topomesh(rotated_surface_topomesh,dirname + "/" + filename + "/" + filename + "_aligned_surface_topomesh.ply",properties_to_save={0:['mean_curvature','normal'],1:[],2:['mean_curvature','normal'],3:[]})

        surface_data['aligned_x'] = aligned_points[:,0]
        surface_data['aligned_y'] = aligned_points[:,1]
        surface_data['aligned_z'] = aligned_points[:,2]

        surface_data.to_csv(dirname + "/" + filename + "/" + filename + "_aligned_ouvle_surface_data.csv")

        image_points = surface_data[['image_x', 'image_y', 'image_z']].values
        rigid_matrix = pts2transfo(image_points, aligned_points)
        np.savetxt(dirname + "/" + filename + "/" + filename + "_ovule_alignment_transform.csv",rigid_matrix)

        surface_map = SignalMap(surface_data,extent=20,resolution=0.5,position_name='aligned',density_k=0.66,radius=1)
        surface_map.compute_signal_map('mean_curvature')
        surface_map.compute_signal_map('center_z')

        figure = plt.figure(4)
        figure.clf()

        figure.add_subplot(1,3,1)
        figure.gca().scatter(surface_data['aligned_x'],surface_data['aligned_y'],c=surface_data['mean_curvature'],cmap='RdBu_r',vmin=-0.1,vmax=0.1)
        figure.gca().set_xlim(-20,20)
        figure.gca().set_ylim(-20,20)

        figure.add_subplot(1,3,2)
        plot_signal_map(surface_map,'mean_curvature',figure,colormap='RdBu_r',signal_range=(-0.5,0.5),signal_lut_range=(-0.1,0.1),distance_rings=False)
        figure.gca().axis('on')
        figure.gca().set_xlim(-20,20)
        figure.gca().set_ylim(-20,20)

        figure.add_subplot(1,3,3)
        plot_signal_map(surface_map,'center_z',figure,colormap='viridis',signal_range=(-80,20),signal_lut_range=(-40,0),distance_rings=False)
        figure.gca().axis('on')
        figure.gca().set_xlim(-20,20)
        figure.gca().set_ylim(-20,20)

        figure.set_size_inches(15,5)
        figure.savefig(dirname + "/" + filename + "/" + filename + "_ovule_surface_curvature_aligned_projection.png")

        reference_img = SpatialImage(np.zeros((256,256,256)).astype(seg_img.dtype),voxelsize=(0.167,0.167,0.167))
        img_center = (np.array(reference_img.shape) * np.array(reference_img.voxelsize)) / 2.
        img_center[2] -= reference_img.shape[2]*reference_img.voxelsize[2]/4.
        alignment_transformation = pts2transfo(img_center + aligned_points, image_points)
        alignment_trsf = create_trsf(param_str_2='-identity', trsf_type=TRSF_TYPE_DICT['RIGID_3D'], trsf_unit=TRSF_UNIT_DICT['REAL_UNIT'])
        allocate_c_bal_matrix(alignment_trsf.mat.c_struct, alignment_transformation)

        if signal_img is not None:
            aligned_signal_img = apply_trsf(signal_img, alignment_trsf, template_img=reference_img, param_str_2='-interpolation linear')
            imsave(dirname + "/" + filename + "/" + filename + "_signal_aligned.tif",aligned_signal_img)

        aligned_seg_img = apply_trsf(seg_img, alignment_trsf, template_img=reference_img, param_str_2='-interpolation nearest')
        imsave(dirname + "/" + filename + "/" + filename + "_seg_aligned.tif",aligned_seg_img)
