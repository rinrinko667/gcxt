import os 
import subprocess 
import trimesh 
import imageio 
import numpy as np 
import shlex


def _get_blender_exec():
    return os.environ.get("BLENDER_PATH", "blender")


def _run_blender(script_path, args_str):
    blender_exec = _get_blender_exec()
    command = f"{shlex.quote(blender_exec)} -P {shlex.quote(script_path)} -b -- {args_str}"
    ret = subprocess.call(command, shell=True)
    if ret != 0:
        raise RuntimeError(
            f"Blender rendering failed (exit code {ret}). "
            "Set BLENDER_PATH to your blender binary and retry."
        )

def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    # command = [
    #     'ffmpeg', '-r', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%05d.png', output_vid_file,
    # ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

def images_to_video_w_imageio(img_folder, output_vid_file):
    img_files = os.listdir(img_folder)
    img_files.sort()
    if len(img_files) == 0:
        raise RuntimeError(
            f"No rendered images were found in {img_folder}. "
            "Please check BLENDER_PATH and Blender execution logs."
        )
    im_arr = []
    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        im = imageio.imread(img_path)
        im_arr.append(im)

    im_arr = np.asarray(im_arr)
    imageio.mimwrite(output_vid_file, im_arr, fps=30, quality=8) 

def run_blender_rendering_and_save2video(obj_folder_path, out_folder_path, out_vid_path, \
    scene_blend_path="utils/blender_utils/for_demo.blend", \
    mat_color="blue"):
    
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    vid_folder = "/".join(out_vid_path.split("/")[:-1])
    if not os.path.exists(vid_folder):
        os.makedirs(vid_folder)
    
    _run_blender(
        "egoego/vis/blender_vis_human_utils.py",
        f"--folder {shlex.quote(obj_folder_path)} "
        f"--scene {shlex.quote(scene_blend_path)} "
        f"--out-folder {shlex.quote(out_folder_path)} "
        f"--material-color {shlex.quote(mat_color)}",
    )

    use_ffmpeg = False
    if use_ffmpeg:
        images_to_video(out_folder_path, out_vid_path)
    else:
        images_to_video_w_imageio(out_folder_path, out_vid_path)

def run_blender_rendering_and_save2video_cmp(obj_folder_path, gt_obj_folder_path, out_folder_path, out_vid_path, \
    scene_blend_path="utils/blender_utils/floor_colorful_mat_human_w_head_pose_hres.blend", \
    mat_color="blue"):
    
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    vid_folder = "/".join(out_vid_path.split("/")[:-1])
    if not os.path.exists(vid_folder):
        os.makedirs(vid_folder)

    _run_blender(
        "egoego/vis/blender_vis_cmp_human_utils.py",
        f"--folder {shlex.quote(obj_folder_path)} "
        f"--gt-folder {shlex.quote(gt_obj_folder_path)} "
        f"--scene {shlex.quote(scene_blend_path)} "
        f"--out-folder {shlex.quote(out_folder_path)} "
        f"--material-color {shlex.quote(mat_color)}",
    )

    images_to_video_w_imageio(out_folder_path, out_vid_path)

def run_blender_rendering_and_save2video_head_pose(npy_path, out_folder_path, out_vid_path, vis_head_only=False, \
    scene_blend_path="utils/blender_utils/floor_colorful_mat_human_w_head_pose_hres.blend"):
    
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    vid_folder = "/".join(out_vid_path.split("/")[:-1])
    if not os.path.exists(vid_folder):
        os.makedirs(vid_folder)

    img_out_folder_path = out_folder_path.replace("objs", "imgs")

    args_str = (
        f"--folder {shlex.quote(out_folder_path)} "
        f"--scene {shlex.quote(scene_blend_path)} "
        f"--out-folder {shlex.quote(img_out_folder_path)} "
        f"--head-path {shlex.quote(npy_path)}"
    )

    if vis_head_only:
        args_str += " --vis_head_only"

    _run_blender(
        "egoego/vis/blender_vis_human_and_headpose_utils.py",
        args_str,
    )

    use_ffmpeg = False
    if use_ffmpeg:
        images_to_video(img_out_folder_path, out_vid_path)
    else:
        images_to_video_w_imageio(img_out_folder_path, out_vid_path)

def save_verts_faces_to_mesh_file(mesh_verts, mesh_faces, save_mesh_folder, save_gt=False):
    # mesh_verts: T X Nv X 3 
    # mesh_faces: Nf X 3 
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx],
                        faces=mesh_faces)
        if save_gt:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+"_gt.obj")
        else:
            curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+".obj")
        mesh.export(curr_mesh_path)
