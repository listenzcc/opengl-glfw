import pydicom
import pydicom.fileset
import nibabel
import numpy as np
import pyvista as pv

from pathlib import Path
from tqdm.auto import tqdm

# Brain data, fMRI
img = nibabel.load('./brain-f.nii')
data = img.get_fdata()
print(data.shape)

# Brain data, T1
# img = nibabel.load('./brain-T1.mgz')
# data = img.get_fdata()
# print(data.shape)

# Head data
# data_folder = Path(f'D:\\郭小龙\\口腔CT数据\\CBCT数据1, FOV 直径15cm乘以高度12cm，Voxel 0.3mm')
# files = list(data_folder.iterdir())
# dcm_file_path = files[-1]
# ds = pydicom.dcmread(dcm_file_path)
# fs = pydicom.fileset.FileSet(ds)
# data = np.array([pydicom.dcmread(
#     inst.path).pixel_array for inst in tqdm(fs, 'Reading .dcm files')])
# print(data.shape)

# Plot
grid = pv.ImageData(dimensions=data.shape)
grid.point_data["values"] = data.flatten(order="F")

# 创建绘图器并添加体积
plotter = pv.Plotter(line_smoothing=True)
# plotter.add_volume(grid, cmap="gray", opacity="sigmoid")
plotter.add_volume(grid, cmap="plasma", opacity="sigmoid")

# 显示结果
plotter.show()
