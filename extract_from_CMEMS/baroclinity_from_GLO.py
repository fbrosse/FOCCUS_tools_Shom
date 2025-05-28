import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

glo_dir = 'data/'
files_read = 'cmems_mod_glo_phy_anfc_merged-sl*nc'
os.makedirs('figs', exist_ok=True)
os.makedirs(f'{glo_dir}/out', exist_ok=True)
sl = xr.open_mfdataset(f'{glo_dir}/{files_read}')
# Filter the surface values
sl = sl.isel(depth=0)
# Compute the new column as the sum of relevant columns
new_col = 'Water levels correction from baroclinity'
sl[new_col] = sl['global_mean_steric_variation'] + sl['global_mean_mass_volume_variation'] + sl['sea_surface_height']

for t in sl.time.values:
    data = sl.sel(time=t)
    iteration = pd.to_datetime(t).strftime('%Y%m%d_%H')
    filename = f'ssh_corrections_from_GLO_{iteration}'
    data[new_col].to_netcdf(f'{glo_dir}/out/{filename}.nc')

    # Plot to see the correction to be made
    cn = data[new_col].plot.contourf(robust=True, cmap='vlag')
    plt.tight_layout()
    plt.savefig(f'figs/{filename}.png')
    plt.close()

