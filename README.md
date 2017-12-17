# 576FinalProject

### Workflow 
- ``GIT_LFS_SKIP_SMUDGE=1 git clone``to clone repo without the large files tracked by git lfs. Helps with catching "repo data quota exceeded" exception raised when cloning full repo.
- ``wget http://mattmahoney.net/dc/text8.zip`` and ``unzip text8.zip`` to enable ``-c text8`` in ``run_models.sh``.
