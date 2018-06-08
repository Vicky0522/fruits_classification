import os
import torch
from torch.utils.ffi import create_extension


sources = ['src/roi_interp.c']
headers = ['src/roi_interp.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/roi_interp_cuda.c']
    headers += ['src/roi_interp_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ['build/roi_interp_kernel.so']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    'roi_interp',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
