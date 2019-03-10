# in top level dir, run following cmd:
# . nv_prof.sh

cd ./build
make
cd ../

mkdir -p ./build/nv_prof
rm -rf ./build/nv_prof/*
nvprof -o ./build/nv_prof/prof.nvvp ./build/pose_refine_test
nvvp ./build/nv_prof/prof.nvvp
