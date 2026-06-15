call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;%PATH%
set DISTUTILS_USE_SDK=1
set MAX_JOBS=1
cd mamba_repo
git checkout v2.2.4
C:\Users\HP\miniconda3\envs\pii_v2\Scripts\pip.exe install . --no-build-isolation --no-cache-dir -v
