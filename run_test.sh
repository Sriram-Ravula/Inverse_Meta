cd

cd Inverse_Meta/

conda activate inverse_meta


python3 main_new.py --config configs/new_tests/ddrm_brain_hor.yml --doc TEST_ddrm_brain_hor --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config configs/new_tests/ddrm_brain_rand.yml --doc TEST_ddrm_brain_rand --eta 0.8 --etaB 0.9 --timesteps 1000

python3 main_new.py --config configs/new_tests/ddrm_brain_vert.yml --doc TEST_ddrm_brain_vert --eta 0.8 --etaB 0.9 --timesteps 1000



python3 main_new.py --config configs/new_tests/ddrm_knee_hor.yml --doc TEST_ddrm_knee_hor --eta 0.7 --etaB 1.0 --timesteps 1000  

python3 main_new.py --config configs/new_tests/ddrm_knee_rand.yml --doc TEST_ddrm_knee_rand --eta 0.7 --etaB 1.0 --timesteps 1000  

python3 main_new.py --config configs/new_tests/ddrm_knee_vert.yml --doc TEST_ddrm_knee_vert --eta 0.7 --etaB 1.0 --timesteps 1000  




python3 main_new.py --config configs/new_tests/wavelet_brain_hor.yml --doc TEST_wavelet_brain_hor 

python3 main_new.py --config configs/new_tests/wavelet_brain_rand.yml --doc TEST_wavelet_brain_rand 

python3 main_new.py --config configs/new_tests/wavelet_brain_vert.yml --doc TEST_wavelet_brain_vert



python3 main_new.py --config configs/new_tests/wavelet_knee_hor.yml --doc TEST_wavelet_knee_hor 

python3 main_new.py --config configs/new_tests/wavelet_knee_rand.yml --doc TEST_wavelet_knee_rand 

python3 main_new.py --config configs/new_tests/wavelet_knee_vert.yml --doc TEST_wavelet_knee_vert 