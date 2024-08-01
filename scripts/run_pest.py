import pyemu
import shutil
import os

master = True

temp = r"pestrun"
num_workers = 5
m_d = 'master' #the manager directory
tmp_d = 'template'



if not os.path.exists(tmp_d):
	print('copying template')	
	shutil.copytree(temp, tmp_d)
	
# shutil.copytree(temp, tmp_d)
assert os.path.exists(tmp_d)

if master:
	print('copying to master folder')
	if not os.path.exists(m_d):
		shutil.copytree(temp, m_d)
		assert os.path.exists(m_d)
	local = True
else:
	print('not copying to master')
	m_d = None
	local = '172.30.4.36'
	local = '0.0.0.0'
	# local = 'AVWRP01'


# from pathlib import Path
# import pyemu
 
# p = "/bin"
# bindir = Path(p)
# bindir.mkdir(exist_ok=True)

# pyemu.utils.get_pestpp(p)
# list(bindir.iterdir())

# Or use an auto-select option
# pyemu.utils.get_pestpp(":windowsapps")

# pyemu.os_utils.run("pestpp-glm pest_ies.pst", cwd=tmp_d)

pyemu.os_utils.start_workers(tmp_d, # the folder which contains the "template" PEST dataset
                            'pestpp-ies', #the PEST software version we want to run
                            'pest_ies.pst', # the control file to use with PEST
                            num_workers=num_workers, #how many agents to deploy
                            worker_root='.', #where to deploy the agent directories; relative to where python is running
                            master_dir=m_d, #the manager directory
                            )
							
# pyemu.os_utils.start_workers(tmp_d, # the folder which contains the "template" PEST dataset
                            # 'pestpp-ies', #the PEST software version we want to run
                            # 'pest_ies.pst', # the control file to use with PEST
							# local = local,
                            # num_workers=num_workers, #how many agents to deploy
                            # worker_root='.', #where to deploy the agent directories; relative to where python is running
                            # master_dir=m_d, #the manager directory
							# verbose = True
                            # )